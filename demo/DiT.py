import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from einops import pack, rearrange
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CLASS = 10  # MNIST Dataset
LABEL = 0  # for generating uncond sample
UNCOND_LABEL = N_CLASS
SAMPLE_GUIDANCE = 5
LABEL_DROPOUT = 0.2

## Transformer architecture
N_HEAD = 32
N_LAYER = 8
N_EMBD = 256
N_FF = N_EMBD * 4
PATCH_SIZE = 4

## Diffusion architecture
T_MAX = 400
BETA_MIN = 1e-4
BETA_MAX = 0.02
SCHEDULE = "linear"

## Data and training
DATASET = "MNIST"
IMG_CHANNELS = 1
IMG_SIZE = 28


def extract(a, t, x_shape):
    """
    Extract relevant alpha/beta/etc given timestep tensor t, reshape for compatibility with image x.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


MODEL_NAME = (
    f"dit_cfg_{N_LAYER}_LAYERs_{N_HEAD}_HEADs_{N_EMBD}_EMBD_DIM_{T_MAX}_TMAX_{DATASET}"
)
print("Model Name:", MODEL_NAME)

transform_fn = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert images to PyTorch tensors and [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1, 1]
    ]
)
# Download and load data for comparison between generations and originals
if DATASET == "CIFAR-10":
    pass
elif DATASET == "MNIST":
    val_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform_fn
    )


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
        self.eta = eta

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

        B, C, H, W = xt.shape

        xt_denoised_list = [xt]

        if method == "cfg":
            neg_labels = torch.full(
                (B,), neg_label, dtype=torch.long, device=device
            )  ## unconditional label, or custom negative prompt

        ## for DDIM, setup tensor of timesteps t to denoise
        interval = self.t_max // steps
        ts = torch.arange(0, self.t_max, step=interval)

        ts_next = torch.cat([torch.tensor([ts[0]]), ts[:-1]])

        for i, step in enumerate(reversed(range(0, steps))):

            t_tensor = torch.full((B,), ts[step], dtype=torch.long, device=device)
            t_tensor_next = torch.full(
                (B,), ts_next[step], dtype=torch.long, device=device
            )

            alpha_bar = extract(self.alphas_bar, t_tensor, xt.shape)
            alpha_bar_next = extract(self.alphas_bar, t_tensor_next, xt.shape)

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

            ################### DDIM time

            x0_t = (xt - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()

            c1 = (
                self.eta
                * (
                    (1 - alpha_bar / alpha_bar_next)
                    * (1 - alpha_bar_next)
                    / (1 - alpha_bar)
                ).sqrt()
            )  ## σ
            c2 = ((1 - alpha_bar_next) - c1 ** 2).sqrt()

            xt = (
                alpha_bar_next.sqrt() * x0_t
                + c1 * torch.randn_like(xt)
                + c2 * noise_pred
            )
            # putting it all together, this is the generation formula for the next denoised iter

            # if ts[step] == steps - 1 or ts[step] % 10 == 0: ## first step / every 20 steps, save partially denoised image
            ##### TODO: change to depend on `steps`...

            # normalize to [0,1] manually
            im_min, im_max = xt.min(), xt.max()
            image = (xt - im_min) / (im_max - im_min)
            xt_denoised_list.append(image)

        return xt, xt_denoised_list


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
        device,
        dropout=0.1,
    ):
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.drop = nn.Dropout(p=dropout)

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

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

        # Step 3: Compute scaled dot-product attention
        attn = self.scaled_dot_product(q, k, v, self.drop)

        # Step 4 and 5: Concatenate attention scores, return projected output matrix
        out = self.out(self.combine_heads(attn))  # (B, S, n_embd)
        return out


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_ff, device, norm_first, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(
            n_embd,
            n_head,
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


class DiT(nn.Module):
    """
    Diffusion Transformer

    TODO
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

        self.ddpm = DDPM(t_max, beta_min, beta_max, schedule, s)

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


def make_denoise_frames(
    model,
    label=None,
    num_rows=1,
    num_steps=20,
    scale=SAMPLE_GUIDANCE,
    neg_label=UNCOND_LABEL,
):
    neg_label = UNCOND_LABEL if neg_label == "unconditional" else neg_label
    if label is None:
        labels = torch.arange(N_CLASS, device=device).repeat(num_rows)
    else:
        labels = torch.full((N_CLASS,), label, dtype=torch.long, device=device).repeat(
            num_rows
        )
    model.eval()
    noise = torch.randn(
        N_CLASS * num_rows, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device
    )
    diffused, history = model.ddpm.backward(
        noise,
        model,
        labels,
        method="cfg",
        scale=scale,
        neg_label=neg_label,
        steps=num_steps,
    )
    frames = [vutils.make_grid(img, normalize=True, nrow=N_CLASS) for img in history]
    np_frames = [frame.permute(1, 2, 0).cpu().numpy() for frame in frames]
    images = []
    for i in range(len(np_frames)):
        fig, ax = plt.subplots()
        ax.imshow(np_frames[i])
        frame_text = f"Frame 0 (Pure Noise)" if i == 0 else f"Frame {i}"
        ax.text(
            0,
            1.23,
            frame_text,
            fontsize=10,
            va="top",
            ha="left",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        images.append(fig)
    return images


def plot_MNIST_images(num_rows, label):
    loader = DataLoader(val_data, batch_size=50, shuffle=True)

    def get_images_for_label(label, num_rows):
        images_for_label = []
        for images, labels in loader:
            images_for_label.extend(images[labels == label])
            if len(images_for_label) >= num_rows:
                break
        return images_for_label[:num_rows]

    if label is not None:
        images_to_plot = get_images_for_label(label, num_rows * N_CLASS)
    else:
        images_to_plot = []
        for digit in range(N_CLASS):
            images_to_plot.extend(get_images_for_label(digit, num_rows))
    images_to_plot = images_to_plot[: num_rows * N_CLASS]
    indices = (
        np.arange(len(images_to_plot)).reshape(N_CLASS, num_rows).T.flatten().tolist()
    )
    images_to_plot = [images_to_plot[i] for i in indices]
    image_grid = vutils.make_grid(images_to_plot, nrow=N_CLASS, normalize=True)
    fig, ax = plt.subplots()
    ax.imshow(image_grid.permute(1, 2, 0).cpu().numpy())
    ax.set_axis_off()

    return fig
