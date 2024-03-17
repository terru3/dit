## Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import pack, rearrange

# from torch.backends.cuda import sdp_kernel, SDPBackend

from constants import *
from utils import set_seed, device

## Setup
set_seed()

## Model


def extract(a, t, x_shape):
    """
    Extract relevant alpha/beta/etc given timestep tensor t, reshape for compatibility with image x.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DDPM(nn.Module):
    def __init__(self, t_max, beta_min, beta_max, schedule="linear", s=0.008, eta=1):

        assert schedule in [
            "linear",
            "cosine",
        ], "beta variance schedule must be `linear` or `cosine`"
        if schedule == "cosine":
            assert s > 0, "cosine offset must be positive"

        super().__init__()
        ## init variance schedule, alphas, etc.
        self.t_max = t_max
        self.beta_min, self.beta_max = beta_min, beta_max
        self.eta = eta  # for DDIM-style sampling if you enable. (1 = DDPM equivalent, 0 = DDIM proper)

        ## linear schedule
        if schedule == "linear":
            self.betas = torch.linspace(beta_min, beta_max, t_max)
        ## cosine schedule
        elif schedule == "cosine":
            # https://arxiv.org/abs/2102.09672
            # β_t = 1 - ( a-bar_t / a-bar_{t-1}), where
            # a-bar_t = f(t) / f(0), where
            # f(t) = cos^2([π/2] * [((t/T) + s) /  (1+s)])
            steps = t_max + 1
            t = torch.linspace(0, t_max, steps)
            ft = torch.cos(((t / t_max) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_bar_t = ft / ft[0]
            betas = 1 - (alphas_bar_t[1:] / alphas_bar_t[:-1])
            self.betas = torch.clip(
                betas, beta_min, beta_max
            )  ### changed from 0.9999 to beta_max
            ## observation: mixing cosine schedule w/ max 0.999 (rather than 0.02 max) is incompatible with current sampling code it seems
            # (at least with ε-prediction rather than x0), yields grey blurry images when sampling

        ## to parameterize p and q
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def forward(self, x0, t):
        ## i.e. q_sample
        noise = torch.randn_like(x0)

        # recall xt  = √(a-bar_t)x0 + √(1-a-bar_t)ε
        # t will be a tensor when .forward is called, no need to create tensor version
        alpha_bar = extract(self.alphas_bar, t, x0.shape)
        mean = x0 * alpha_bar.sqrt()
        std = (1 - alpha_bar).sqrt()
        xt = mean + std * noise
        return xt, noise

    def backward(
        self,
        xt,
        model,
        labels,
        steps=None,
        method="conditional",
        scale=SAMPLE_GUIDANCE,
        neg_label=UNCOND_LABEL,
    ):

        assert method in ["conditional", "cfg"], "method must be `conditional` or `cfg`"
        if method == "cfg":
            assert (
                isinstance(scale, (int, float, np.integer)) and scale >= 0
            ), "scale must be a float greater than or equal to 0"

        if steps is None:
            steps = self.t_max

        ## i.e. p_sample in a loop
        ## for i in reversed(range(0, timesteps)
        ## send through model, convert predicted noise (eta) into posterior mean, return mean + std*noise
        ## if last timestep, just return mean

        ## fix variances simply as beta_t, like in paper, rather than learn

        # note labels is a tensor already

        B, C, H, W = xt.shape

        ## list of how the images look after each denoising step
        xt_denoised_list = []

        if method == "cfg":
            neg_labels = torch.full(
                (B,), neg_label, dtype=torch.long, device=device
            )  ## unconditional label, or custom negative prompt

        for t in reversed(range(0, steps)):

            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)

            beta = extract(self.betas, t_tensor, xt.shape)
            alpha = extract(self.alphas, t_tensor, xt.shape)
            alpha_bar = extract(self.alphas_bar, t_tensor, xt.shape)

            with torch.no_grad():
                noise_pred = model(xt, t_tensor, labels)
            if method == "cfg":
                # if input label is already unconditional, skip
                if labels[0] == UNCOND_LABEL:
                    pass
                else:
                    with torch.no_grad():
                        neg_noise_pred = model(xt, t_tensor, neg_labels)
                    noise_pred = scale * noise_pred + (1 - scale) * neg_noise_pred

            # formula to convert predicted noise into model mean:
            # μθ(xt, t) = (1/√a_t) * (xt - (β_t / √(1-a-bar_t)) * noise_pred)
            mean = (1 / alpha.sqrt()) * (
                xt - noise_pred * (beta / (1 - alpha_bar).sqrt())
            )
            if t == 0:
                xt = mean  # if last step, return noiseless image
            else:
                noise = torch.randn_like(xt)
                xt = mean + beta.sqrt() * noise

            if (
                t == steps - 1 or t % 20 == 0
            ):  ## first step / every 20 steps, save partially denoised image
                ##### TODO: change to depend on `steps`...

                # normalize to [0,1] manually
                im_min, im_max = xt.min(), xt.max()
                image = (xt - im_min) / (im_max - im_min)
                xt_denoised_list.append(image)

        return xt, xt_denoised_list


############## DDIM sampling. Above DDPM sampling does not allow for denoising fewer steps (changing `steps` param will result in just noise)
## DDIM does. If eta=1 (default), equivalent to DDPM but with step interpolation enabled,
## and if eta = 0 that's proper DDIM. For some reason eta=0 doesn't do as well and looks a little sus but eta=1 is very efficient and nice

# ## for DDIM, setup tensor of timesteps t to denoise
# interval = self.t_max // steps
# ts = torch.arange(0, self.t_max, step=interval)
# steps = 10

# ts_next = torch.cat([torch.tensor([ts[0]]), ts[:-1]])

# for i, step in enumerate(reversed(range(0, steps))):

#     t_tensor = torch.full((B,), ts[step], dtype=torch.long, device=device)
#     t_tensor_next = torch.full((B,), ts_next[step], dtype=torch.long, device=device)

#     alpha_bar = extract(self.alphas_bar, t_tensor, xt.shape)
#     alpha_bar_next = extract(self.alphas_bar, t_tensor_next, xt.shape)

#     with torch.no_grad():
#       noise_pred = model(xt, t_tensor, labels)
#     if method == 'cfg':
#       # if input label is already unconditional, skip
#       if labels[0] == UNCOND_LABEL:
#         pass
#       else:
#         with torch.no_grad():
#           neg_noise_pred = model(xt, t_tensor, neg_labels)
#         noise_pred = scale*noise_pred + (1-scale) * neg_noise_pred

#     x0_t = (xt - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()

#     c1 = self.eta * ((1 - alpha_bar / alpha_bar_next) * (1 - alpha_bar_next) / (1 - alpha_bar)).sqrt() ## σ
#     c2 = ((1 - alpha_bar_next) - c1 ** 2).sqrt()

#     xt = alpha_bar_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * noise_pred
#     # putting it all together, this is the generation formula for the next denoised iter

#     if ts[step] == steps - 1 or ts[step] % 20 == 0: ## first step / every 20 steps, save partially denoised image
#       ##### TODO: change to depend on `steps`...

#       # normalize to [0,1] manually
#       im_min, im_max = xt.min(), xt.max()
#       image = (xt - im_min)/(im_max - im_min)
#       xt_denoised_list.append(image)

# return xt, xt_denoised_list

# # for flash attention
# backend_map = {
#     SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
#     SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
#     SDPBackend.EFFICIENT_ATTENTION: {
#         "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
# }


class MLP(nn.Module):
    def __init__(self, n_embd, n_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_ff, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        # n_kv_head,
        device,
        dropout=0.1,
    ):
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.drop = nn.Dropout(p=dropout)

        # self.n_kv_head = n_kv_head
        # self.n_repeat = self.n_head // self.n_kv_head

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

        # self.key = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        # self.value = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)

        self.device = device

    def split_heads(self, x, n_head):
        B, S, D = x.size()
        # split dimension into n_head * head_dim, then transpose the sequence length w/ n_head
        # output: [B, n_head, S, head_dim]
        return x.view(B, S, n_head, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        B, _, S, head_dim = x.size()  # _ is n_head which we will merge
        # output: [B, S, n_embd]
        return x.transpose(1, 2).contiguous().view(B, S, self.n_embd)

    def scaled_dot_product(self, q, k, v, dropout):
        # q,k,v are [B, n_head, S, head_dim]
        # the key transpose sets up batch multiplication s.t. wei = [B, n_head, S, S]
        wei = q @ k.transpose(-2, -1) / np.sqrt(self.head_dim)
        # mask is [B, 1, S, S], so simply broadcasted across each head and works as expected
        wei = dropout(F.softmax(wei, dim=-1))
        out = wei @ v
        return out

    def forward(self, x):
        # x: (B, S, n_embd)
        # Step 1 and 2: Project query, key, value, then split via reshaping
        q = self.split_heads(self.query(x), self.n_head)
        k = self.split_heads(self.key(x), self.n_head)
        v = self.split_heads(self.value(x), self.n_head)
        # k = self.split_heads(self.key(x), self.n_kv_head)
        # v = self.split_heads(self.value(x), self.n_kv_head)

        # ## GQA
        # k, v = repeat_kv(k, v, self.n_repeat)
        # assert (
        #     k.shape[1] == self.n_head and v.shape[1] == self.n_head
        # ), "key and value n_head do not match query n_head"
        # # q, k, v [B, n_head, S, head_dim)

        # Step 3: Compute scaled dot-product attention
        attn = self.scaled_dot_product(q, k, v, self.drop)

        # # Step 3: Compute scaled dot-product attention
        # with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
        #     try:
        #         attn = F.scaled_dot_product_attention(
        #             q,
        #             k,
        #             v,
        #             dropout_p=self.drop.p if self.device.type == "cuda" else 0
        #         ) # ViT: not causal ofc
        #     # CPU: Both fused kernels do not support non-zero dropout. (Dec 2023)
        #     except RuntimeError:
        #         print("FlashAttention is not supported. See warnings for reasons.")

        # Step 4 and 5: Concatenate attention scores, return projected output matrix
        out = self.out(self.combine_heads(attn))  # (B, S, n_embd)
        return out


# # helper function for GQA
# def repeat_kv(k, v, n_repeat):
#     k = torch.repeat_interleave(k, repeats=n_repeat, dim=1)
#     v = torch.repeat_interleave(v, repeats=n_repeat, dim=1)
#     return k, v


class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        # n_kv_head,
        n_ff,
        device,
        norm_first,
        dropout,
    ):
        super().__init__()
        self.sa = MultiHeadAttention(
            n_embd,
            n_head,
            # n_kv_head,
            device,
            dropout,
        )
        self.ff = MLP(n_embd, n_ff, dropout=dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.norm_first = norm_first
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        # residual connection (stream)

        # pre layer norm
        if self.norm_first:
            x = x + self.drop(self.sa(self.ln1(x)))
            x = x + self.drop(self.ff(self.ln2(x)))
        else:
            x = self.ln1(x + self.drop(self.sa(x)))
            x = self.ln2(x + self.drop(self.ff(x)))

        return x


## for AdaLN
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class AdaLNBlock(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        # n_kv_head,
        n_ff,
        device,
        norm_first,
        dropout,
    ):
        super().__init__()
        self.sa = MultiHeadAttention(
            n_embd,
            n_head,
            # n_kv_head,
            device,
            dropout,
        )
        self.ff = MLP(n_embd, n_ff, dropout=dropout)

        self.ln1 = nn.LayerNorm(
            n_embd, elementwise_affine=False
        )  ######## no learnable parameters here!
        self.ln2 = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.norm_first = norm_first
        self.drop = nn.Dropout(p=dropout)

        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(n_embd, 6 * n_embd))

    def forward(self, x, c):
        ## c = (B, 1, n_embd)
        # residual connection (stream)

        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(
            c
        ).chunk(6, dim=-1)

        # pre layer norm
        if self.norm_first:
            x = x + self.drop(
                gate_sa * self.sa(modulate(self.ln1(x), shift_sa, scale_sa))
            )
            x = x + self.drop(
                gate_mlp * self.ff(modulate(self.ln2(x), shift_mlp, scale_mlp))
            )
        else:
            x = self.ln1(
                x + self.drop(gate_sa * self.sa(modulate(x, shift_sa, scale_sa)))
            )
            x = self.ln2(
                x + self.drop(gate_mlp * self.ff(modulate(x, shift_mlp, scale_mlp)))
            )
        return x


class PatchEmbedding(nn.Module):
    """
    Applies patch embeddings to an image.
    """

    def __init__(self, patch_size, n_embd, in_channels=3):
        super().__init__()

        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            in_channels, n_embd, kernel_size=patch_size, stride=patch_size
        )
        # self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # (B, C, img_size, img_size) -> (B, num_patches, n_embd)
        x = self.conv(x)  # (B, n_embd, img_size//patch_size, img_size//patch_size)
        x = rearrange(x, "b c h w -> b (h w) c")
        # equivalent to above line: x = x.flatten(2).transpose(-1, -2)
        return x


class OutputLayer(nn.Module):
    """
    The final layer of DiT. Removes timestep and label tokens, then projects (B, num_patches, n_embd) -> (B, num_patches, patch_size**2 * num_channels=C)
    """

    def __init__(self, patch_size, n_embd, out_channels=3):
        super().__init__()

        self.linear = nn.Linear(n_embd, patch_size * patch_size * out_channels)
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.norm(x)
        x = x[:, 2:, :]
        x = self.linear(x)
        return x


class AdaLNOutputLayer(nn.Module):
    """
    The final layer of DiT. Projects (B, num_patches, n_embd) -> (B, num_patches, patch_size**2 * num_channels=C)
    """

    def __init__(self, patch_size, n_embd, out_channels=3):
        super().__init__()

        self.linear = nn.Linear(n_embd, patch_size * patch_size * out_channels)
        self.norm = nn.LayerNorm(n_embd)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(n_embd, 2 * n_embd))

    def forward(self, x, c):
        ## c = (B, 1, n_embd)
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = self.norm(x)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer.

    img_size (int): Width/height of input images (assuming square)
    """

    def __init__(
        self,
        n_embd,
        n_head,
        n_ff,
        n_layer,
        n_class,
        n_channel,
        img_size,
        patch_size,
        t_max,
        beta_min,
        beta_max,
        schedule="linear",
        s=0.008,
        eta=1,
        norm_first=True,
        device=device,
        dropout=0,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(patch_size, n_embd, in_channels=n_channel)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 2 + self.num_patches, n_embd)
        )  ## original DiT paper uses 2D cos/sin embeddings
        ## 2+num_patches to pack both timestep and label embedding into x

        self.timestep_embedding = nn.Embedding(t_max, n_embd)
        self.label_embedding = nn.Embedding(
            n_class + 1, n_embd
        )  ### typically n_class. changed to n_class+1 to add CFG

        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head, n_ff, device, norm_first, dropout)
                for i in range(n_layer)
            ]
        )

        self.mlp_head = OutputLayer(patch_size, n_embd, out_channels=n_channel)

        self.drop = nn.Dropout(dropout)
        self.device = device

        self.ddpm = DDPM(t_max, beta_min, beta_max, schedule, s, eta)

    #     self.init_params()

    # ### unsure if appropriate for DiT
    # # weight initialization (Xavier uniform)
    # def init_params(self, default_initialization=False):
    #     if not default_initialization:
    #         for name, p in self.named_parameters():
    #             if p.dim() > 1:
    #                 # excludes layer norm and biases
    #                 nn.init.xavier_uniform_(p)
    #             elif 'bias' in name:
    #                 nn.init.zeros_(p)

    def forward(self, x, t, label):
        ## x = (B, num_channels, img_size, img_size)
        ## t = (B, ) of timesteps
        ## label = (B, ) of labels

        B = x.shape[0]

        x = self.patch_embedding(x)  # (B, num_patches, n_embd)
        te = self.timestep_embedding(t).unsqueeze(1)  # (B, 1, n_embd)
        lab = self.label_embedding(label).unsqueeze(1)  # (B, 1, n_embd)

        x, _ = pack(
            (te, lab, x), "B * n_embd"
        )  # expand into dim=1 to form num_patches+2
        x += self.pos_embedding
        x = self.drop(x)
        # (B, 2+num_patches, n_embd)

        for block in self.blocks:
            x = block(x)  # (B, num_patches+2, n_embd)

        # last layer: remove timestep token and project to (B, num_patches, patch_size**2 * num_channels=C)
        # unpatchify then returns img (B, C, H, W)
        x = self.mlp_head(x)
        img = rearrange(
            x,
            "B (H W) (P1 P2 C) -> B C (H P1) (W P2)",
            P1=self.patch_size,
            P2=self.patch_size,
            H=self.img_size // self.patch_size,
        )
        return img


class AdaLNDiT(nn.Module):
    """
    Diffusion Transformer with AdaLN.

    img_size (int): Width/height of input images (assuming square)
    """

    def __init__(
        self,
        n_embd,
        n_head,
        n_ff,
        n_layer,
        n_class,
        n_channel,
        img_size,
        patch_size,
        t_max,
        beta_min,
        beta_max,
        schedule="linear",
        s=0.008,
        eta=1,
        norm_first=True,
        device=device,
        dropout=0,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(patch_size, n_embd, in_channels=n_channel)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, n_embd)
        )  ## original DiT paper uses 2D cos/sin embeddings
        ######### note no more 2+patches in AdaLN-Zero because we do not concatenate the conditioning info to x

        self.timestep_embedding = nn.Embedding(t_max, n_embd)
        self.label_embedding = nn.Embedding(
            n_class + 1, n_embd
        )  ### typically n_class. changed to n_class+1 to add CFG

        self.blocks = nn.Sequential(
            *[
                AdaLNBlock(n_embd, n_head, n_ff, device, norm_first, dropout)
                for i in range(n_layer)
            ]
        )

        self.mlp_head = AdaLNOutputLayer(patch_size, n_embd, out_channels=n_channel)

        self.drop = nn.Dropout(dropout)
        self.device = device

        self.ddpm = DDPM(t_max, beta_min, beta_max, schedule, s, eta)
        self.init_params()

    def init_params(self, default_initialization=False):
        ## Xavier uniform——### unsure if  appropriate for DiT
        # if not default_initialization:
        #     for name, p in self.named_parameters():
        #         if p.dim() > 1:
        #             # excludes layer norm and biases
        #             nn.init.xavier_uniform_(p)
        #         elif 'bias' in name:
        #             nn.init.zeros_(p)

        ## key for Ada-LN Zero——initialize each block as the identity
        ## in particular, zero out the linear modulation layers and output layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)
        nn.init.constant_(self.mlp_head.adaLN[-1].weight, 0)
        nn.init.constant_(self.mlp_head.adaLN[-1].bias, 0)
        nn.init.constant_(self.mlp_head.linear.weight, 0)
        nn.init.constant_(self.mlp_head.linear.bias, 0)

    def forward(self, x, t, label):
        ## x = (B, num_channels, img_size, img_size)
        ## t = (B, ) of timesteps
        ## label = (B, ) of labels

        B = x.shape[0]

        x = self.patch_embedding(x)  # (B, num_patches, n_embd)

        x += self.pos_embedding
        x = self.drop(x)
        # (B, num_patches, n_embd)

        te = self.timestep_embedding(t).unsqueeze(1)  # (B, 1, n_embd)
        lab = self.label_embedding(label).unsqueeze(1)  # (B, 1, n_embd)
        c = te + lab  ## (B, 1, n_embd)

        # Note in AdaLN-Zero, rather than packing the t and label embeddings into x, we embed them separately and add together as 'c',
        # then send through every block and final layer

        for block in self.blocks:
            x = block(x, c)  # (B, num_patches, n_embd)

        # last layer: project to (B, num_patches, patch_size**2 * num_channels=C)
        # unpatchify then returns img (B, C, H, W)
        x = self.mlp_head(x, c)
        img = rearrange(
            x,
            "B (H W) (P1 P2 C) -> B C (H P1) (W P2)",
            P1=self.patch_size,
            P2=self.patch_size,
            H=self.img_size // self.patch_size,
        )
        return img
