# from https://github.com/ashawkey/stable-dreamfusion/blob/main/sd.py

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd 

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad) 
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype) 

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.1', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, override_t=None):
        
        # interp to 512x512 to be fed into vae.

        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        
        if override_t == None:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        else:
            t = override_t

        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_512)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        grad = grad.clamp(-10, 10)
        #grad = torch.nan_to_num(grad)
        
        with torch.no_grad():
            grad_mag = grad.abs().sum()
        #print(f"grad shape: {grad.shape}")

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad) 
        
        #print(f"loss shape: {loss.shape}")

        return loss, grad_mag

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents
