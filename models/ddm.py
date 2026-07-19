import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNet
from models.mods import HFRM
import torch.nn as nn
from torch.nn import functional as F
import torch.optim
import clip_loss as clip_loss
from collections import OrderedDict
import clip
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
c_model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"))
c_model.to(device)


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(
            dim=-1)] @ self.text_projection
        return x


class Prompts(nn.Module):
    def __init__(self, initials=None):
        super(Prompts, self).__init__()
        self.text_encoder = TextEncoder(c_model)
        if isinstance(initials, list):
            text = clip.tokenize(initials).cuda()
            print(text)
            self.embedding_prompt = nn.Parameter(
                c_model.token_embedding(text).requires_grad_()).cuda()
        elif isinstance(initials, str):
            prompt_path = initials
            state_dict = torch.load(prompt_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            self.embedding_prompt = nn.Parameter(
                new_state_dict['embedding_prompt']).cuda()
            self.embedding_prompt.requires_grad = True
        else:
            self.embedding_prompt = torch.nn.init.xavier_normal_(nn.Parameter(
                c_model.token_embedding([" ".join(["X"] * 16), " ".join(["X"] * 16)]).requires_grad_())).cuda()

    def forward(self, tensor, flag=1):

        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in [" ".join(["X"] * 16)]])
        text_features = self.text_encoder(
            self.embedding_prompt, tokenized_prompts)
        for i in range(tensor.shape[0]):
            image_features = tensor[i]
            nor = torch.norm(text_features, dim=-1, keepdim=True)
            if flag == 0:
                similarity = (100.0 * image_features @
                              (text_features / nor).T)
                if (i == 0):
                    probs = similarity
                else:
                    probs = torch.cat([probs, similarity], dim=0)
            else:
                similarity = (100.0 * image_features @
                              (text_features / nor).T).softmax(dim=-1)
                if (i == 0):
                    probs = similarity[:, 0]
                else:
                    probs = torch.cat([probs, similarity[:, 0]], dim=0)
        return probs


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                                                 1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                             num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end,
                            num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps,
                                  1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.high_enhance0 = HFRM(in_channels=2, out_channels=64)
        self.high_enhance1 = HFRM(in_channels=2, out_channels=64)
        self.Unet = DiffusionUNet(config)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):

        ir_img = x_cond[:, :1, :, :]
        visr_img = x_cond[:, 1:2, :, :]

        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = ir_img.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([xt, x_cond, ir_img, visr_img], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        # return xs[-1]
        return xs

    def forward(self, x):
        data_dict = {}
        input_img = x[:, :2, :, :]

        ir_img = x[:, :1, :, :]
        visr_img = x[:, 1:2, :, :]

        b = self.betas.to(input_img.device)

        t = torch.randint(low=0, high=self.num_timesteps, size=(
            input_img.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1],
                      dim=0)[:input_img.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(ir_img)

        if self.training:
            x1 = data_transform(x[:, :1, :, :])
            irX = x1 * a.sqrt() + e * (1.0 - a).sqrt()

            x2 = data_transform(x[:, 1:2, :, :])
            visX = x2 * a.sqrt() + e * (1.0 - a).sqrt()

            fusex = (irX + visX) / 2
            cond = torch.cat([ir_img, visr_img], dim=1)
            noise_output = self.Unet(
                torch.cat([fusex, cond, irX, visX], dim=1), t.float())
            denoise_list = self.sample_training(input_img, b)
            pred_x = denoise_list[-1]

            pred_x = inverse_data_transform(pred_x)

            data_dict["denoise_list"] = denoise_list
            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["e"] = e

        else:
            denoise_LL_LL_list = self.sample_training(input_img, b)
            pred_x = denoise_LL_LL_list[-1]
            pred_x = inverse_data_transform(pred_x)
            data_dict["pred_x"] = pred_x
        return data_dict


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()

        self.content_loss = Fusionloss()

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("Load checkpoint: ", os.path.exists(load_path))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch + 1)

            for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.training.n_epochs}", leave=False):
                data_start = time.time()
                data_time = 0
                for i, (x, fileName, irImage, visImage) in enumerate(train_loader):
                    irImage = irImage.to(self.device)
                    visImage = visImage.to(self.device)
                    visImageYCrCb = RGB2YCrCb(visImage)
                    images_visY = visImageYCrCb[:, 0:1, :, :]
                    x = torch.cat([irImage, images_visY], dim=1)

                    x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x

                    data_time += time.time() - data_start
                    self.model.train()

                    self.step += 1

                    x = x.to(self.device)

                    output = self.model(x)

                    noise_loss, photo_loss = self.estimation_loss(x, output)
                    c_loss = self.clip_loss(self.args, x, output, visImageYCrCb)

                    loss = noise_loss + 0.0005 * photo_loss + 0.001 * c_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.ema_helper.update(self.model)
                    data_start = time.time()

                    if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                        utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                       'state_dict': self.model.state_dict(),
                                                       'optimizer': self.optimizer.state_dict(),
                                                       'scheduler': self.scheduler.state_dict(),
                                                       'ema_helper': self.ema_helper.state_dict(),
                                                       'params': self.args,
                                                       'config': self.config},
                                                      filename=os.path.join(self.config.data.ckpt_dir, 'our_model'))
                print("epoch:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f},c_loss:{:.4f}, "
                      "loss:{:.4f}".format(epoch + 1,
                                           self.scheduler.get_last_lr()[
                                               0],
                                           noise_loss.item(),
                                           photo_loss.item(),
                                           c_loss.item(),
                                           loss.item()))

                self.scheduler.step()

        utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                       'state_dict': self.model.state_dict(),
                                       'optimizer': self.optimizer.state_dict(),
                                       'scheduler': self.scheduler.state_dict(),
                                       'ema_helper': self.ema_helper.state_dict(),
                                       'params': self.args,
                                       'config': self.config},
                                      filename=os.path.join(self.config.data.ckpt_dir, 'our_model'))

    def estimation_loss(self, x, output):

        pred_x, noise_output, e = output["pred_x"], output["noise_output"], output["e"]

        noise_loss = self.l2_loss(noise_output, e)

        image_ir = x[:, :1, :, :].to(self.device)
        image_vis_y = x[:, 1:2, :, :].to(self.device)

        contentlossT = self.content_loss(image_vis_y, image_ir, pred_x)

        photo_loss = contentlossT

        return noise_loss, photo_loss

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(
            self.args.image_folder, self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()
        with torch.no_grad():
            print(
                f"Current Sampling Steps: {step}")
            for i, (x, y, irImage, visImage) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0,
                              img_h_32 - img_h), 'reflect')

                out = self.model(x.to(self.device))
                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]
                utils.logging.save_image(pred_x, os.path.join(
                    image_folder, str(step), f"{y[0]}.png"))

    def clip_loss(self, args, x, output, visImageYCrCb):
        pred_x = output["pred_x"]
        fusion_rgb = self.getFusionImageRGB(pred_x, visImageYCrCb)

        prompt = Prompts(args.prompt_pretrain_dir).cuda()
        prompt = torch.nn.DataParallel(prompt)

        text_encoder = TextEncoder(c_model)

        # 初始化 L_clip 损失
        L_clip = clip_loss.L_clip_from_feature(neg_weight=0.5)

        # 正向文本列表
        prompt_pos = ['an infrared and visible fused image with clear salient targets',
                      'a fused image preserving infrared target information and visible textures',
                      'a fusion result with enhanced thermal targets and rich background details',
                      'a high quality infrared visible fusion image with sharp edges',
                      'a fused image with complete structures and natural illumination',
                      'a semantic consistent fusion image with clear objects and background',
                      'a fusion image with strong target visibility and detailed scene information']
        # 负向文本列表（示例）
        prompt_neg = ['an infrared and visible fused image with missing target information',
                      'a fusion result with blurred objects and weak structural details',
                      'a fusion image with excessive noise and artifacts',
                      'a fusion image losing infrared targets and visible textures',
                      'a fused image with unclear objects and damaged structures',
                      'a fusion result with poor contrast and missing details',
                      'a fusion image with semantic inconsistency between objects and background']

        embedding_prompt = prompt.module.embedding_prompt
        embedding_prompt.requires_grad = False

        token_pos = clip.tokenize(prompt_pos).cuda()
        token_neg = clip.tokenize(prompt_neg).cuda()

        text_feat_pos = text_encoder(embedding_prompt, token_pos)
        text_feat_neg = text_encoder(embedding_prompt, token_neg)

        cliploss = L_clip(fusion_rgb, text_feat_pos, text_feat_neg)

        return cliploss

    def getFusionImageRGB(self, pred_x, visImageYCrCb):
        fusion_ycrcb = torch.cat(
            (pred_x, visImageYCrCb[:, 1:2, :,
                     :], visImageYCrCb[:, 2:, :, :]),
            dim=1,
        )
        fusion_image = YCrCb2RGB(fusion_ycrcb)
        ones = torch.ones_like(fusion_image)
        zeros = torch.zeros_like(fusion_image)
        fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
        fusion_image = torch.where(
            fusion_image < zeros, zeros, fusion_image)
        fused_image = fusion_image.detach().cpu().numpy()
        fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
        )
        fused_image = np.uint8(255.0 * fused_image)
        tensor = torch.from_numpy(fused_image)

        image = tensor.to(self.device)

        return image


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis_y, image_ir, generate_img):
        image_y = image_vis_y
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        loss_total = loss_in + 10 * loss_grad
        return loss_total


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)
