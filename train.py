## Imports
import json
import time

import numpy as np
import torch
import torchvision.utils as vutils

from constants import *
from model import DiT
from utils import set_seed, drop_label, cfg_loss, device


def load_model_optim_loss_imgs(model, optimizer, model_name):
    """
    Loads model and optimizer state as well as list of train losses and generated images from training. Intended to be used to
    resume training from checkpoints.
    """
    model.load_state_dict(
        torch.load(
            f"{path}/model/{model_name}_epoch_{LOAD_EPOCH}.pt", map_location=device
        )["model_state_dict"]
    )
    print("Model loaded")

    optimizer.load_state_dict(
        torch.load(
            f"{path}/model/{model_name}_epoch_{LOAD_EPOCH}.pt", map_location=device
        )["optimizer_state_dict"]
    )
    print("Optimizer loaded")

    try:
        with open(f"{path}/train_logs/{model_name}_train_losses.json", "r") as f3:
            train_losses = json.load(f3)

        print("Train losses loaded")
    except:
        print("Unable to find train losses, returning None")
        train_losses = None

    try:
        img_list = torch.load(f"{path}/train_logs/{model_name}_img_list.pt")
        print("Images loaded")
    except:
        print("Unable to find image list, returning None")
        img_list = None

    return train_losses, img_list


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    model_name,
    patel=False,
    train_guidance=TRAIN_GUIDANCE,
    sample_guidance=SAMPLE_GUIDANCE,
    train_loss_list=None,
    img_list=None,
):

    train_losses = train_loss_list if train_loss_list is not None else []
    img_list = img_list if img_list is not None else []
    train_times = []

    model.train()
    model.to(device)

    for epoch in range(START_EPOCH, EPOCHS):
        print(f"Epoch: {epoch+1}")
        for step, (img, label) in enumerate(train_loader):

            img, label = img.to(device), label.to(device)
            start = time.perf_counter()

            optimizer.zero_grad()

            #### diffusion forward and backward here
            B = img.shape[0]
            t = torch.randint(1, model.ddpm.t_max, (B,), device=device).long()
            xt, noise = model.ddpm.forward(img, t)
            xt, noise = xt.to(device), noise.to(device)

            if patel:  ## Patel CFG loss training

                # regular noise prediction
                preds = model(xt, t.to(device), label.to(device)).float()

                # uncond noise prediction
                uncond_label = torch.full_like(label, UNCOND_LABEL, dtype=torch.long)
                uncond_preds = model(xt, t.to(device), uncond_label.to(device)).float()

                # patel loss
                loss = cfg_loss(preds, uncond_preds, noise, train_guidance)

            else:  ## conventional CFG
                assert criterion is not None
                # label dropout
                drop_label(label, LABEL_DROPOUT)
                preds = model(xt, t.to(device), label.to(device)).float()
                loss = criterion(preds, noise)

            loss.backward()

            # Monitoring gradient norm
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            norm = torch.cat(grads).norm()

            optimizer.step()

            train_times.append(time.perf_counter() - start)

            if step % PRINT_EVERY == 0:
                print(
                    f"Step: {step} | Train Loss: {loss.item():.5f} |",
                    f"Grad Norm: {norm:.3f} | Train Batch Time: {np.mean(train_times):.3f}",
                )

                train_losses.append(loss.item())

        # save model and losses
        if epoch % SAVE_EVERY == 0:
            print(f"Saving model, epoch {epoch+1}")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"{path}/model/{model_name}_epoch_{epoch+1}.pt",
            )

            with open(f"{path}/train_logs/{model_name}_train_losses.json", "w") as f:
                json.dump(train_losses, f)

            print(f"Epoch {epoch+1}, model + data saved \n")

        # generate and save images
        if (epoch % GENERATE_EVERY == 0) or (epoch == EPOCHS - 1):
            print(f"Generating images, epoch {epoch+1}")

            print(f"Digit (CFG): {LABEL}")
            labels = torch.full((BATCH_SIZE,), LABEL, dtype=torch.long, device=device)
            generate_single(model, labels=labels, scale=sample_guidance)

            print("Every digit (CFG):")
            img_list.append(generate_all_classes(model))
            generate_all_classes_steps(model, scale=sample_guidance)

            if not patel:
                print("Unconditional:")
                generate_uncond(model)

            torch.save(img_list, f"{path}/train_logs/{model_name}_img_list.pt")
            model.train()


def generate_single(model, labels, scale=SAMPLE_GUIDANCE, neg_label=UNCOND_LABEL):
    """
    Generates a batch of images of a single class.
    """

    model.eval()
    noise = torch.randn(BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    diffused, _ = model.ddpm.backward(
        noise, model, labels, method="cfg", scale=scale, neg_label=neg_label
    )

    imgs = vutils.make_grid(diffused, normalize=True)
    plt.imshow(imgs.permute(1, 2, 0).cpu())
    plt.show()
    return imgs


def generate_all_classes(model, scale=SAMPLE_GUIDANCE, neg_label=UNCOND_LABEL):
    """
    Generates rows of all classes.
    """

    num_rows = 4

    labels = torch.arange(N_CLASS, device=device).repeat(num_rows)
    model.eval()
    noise = torch.randn(
        N_CLASS * num_rows, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device
    )
    diffused, _ = model.ddpm.backward(
        noise, model, labels, method="cfg", scale=scale, neg_label=neg_label
    )
    imgs = vutils.make_grid(
        diffused, normalize=True, nrow=N_CLASS
    )  # confusingly, nrow = number in each row
    plt.imshow(imgs.permute(1, 2, 0).cpu())
    plt.show()
    return imgs


def generate_all_classes_steps(model, scale=SAMPLE_GUIDANCE, neg_label=UNCOND_LABEL):
    """
    Generates rows of all classes.
    """

    labels = torch.arange(N_CLASS, device=device)
    model.eval()
    noise = torch.randn(N_CLASS, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    diffused, history = model.ddpm.backward(
        noise, model, labels, method="cfg", scale=scale, neg_label=neg_label
    )
    history = torch.cat(history, dim=0)

    imgs = vutils.make_grid(
        history, normalize=False, nrow=N_CLASS
    )  # confusingly, nrow = number in each row
    plt.figure(figsize=(35, 15))
    plt.imshow(imgs.permute(1, 2, 0).cpu())
    plt.show()
    return imgs


def generate_uncond(model):
    """
    Generates images of random classes unconditionally.
    NOTE: Does not use classifier-free guidance (method='cfg'), although it could with scale = 0
    """

    labels = torch.full((BATCH_SIZE,), UNCOND_LABEL, dtype=torch.long, device=device)
    model.eval()
    noise = torch.randn(BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    diffused, _ = model.ddpm.backward(noise, model, labels)
    imgs = vutils.make_grid(diffused, normalize=True)
    plt.imshow(imgs.permute(1, 2, 0).cpu())
    plt.show()
    return imgs


##### TEMP / archived interpolation methods
def interpolate(
    model,
    img,
    denoise_label,
    forward_steps=T_MAX // 2,
    backward_steps=T_MAX // 2,
    scale=SAMPLE_GUIDANCE,
    neg_label=UNCOND_LABEL,
):
    """
    Generates interpolated images between input images and images of `denoise_label`.
    """
    assert (
        isinstance(forward_steps, (int, np.integer))
        and isinstance(backward_steps, (int, np.integer))
        and 0 <= forward_steps <= T_MAX
        and 0 <= backward_steps <= T_MAX
    ), "forward and backward steps must be an integer between 0 and T_MAX inclusive"

    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(device)

    model.eval()
    B = img.shape[0]

    # forward
    t = torch.full((B,), forward_steps, dtype=torch.long, device=device)
    xt, _ = model.ddpm.forward(img, t)
    xt = xt.to(device)

    # backward
    labels = torch.full((B,), denoise_label, dtype=torch.long, device=device)
    diffused, _ = model.ddpm.backward(
        xt,
        model,
        labels,
        steps=backward_steps,
        method="cfg",
        scale=scale,
        neg_label=neg_label,
    )

    imgs = vutils.make_grid(diffused, normalize=True)
    plt.imshow(imgs.permute(1, 2, 0).cpu())
    plt.show()
    return imgs


# to-test: another type of interpolation:
# # regular noising, but call ddpm.backward to denoise the original digit half-way and some new digit during the second half
def interpolate2(
    model,
    img,
    orig_label,
    denoise_label,
    steps_first=T_MAX // 2,
    steps_second=T_MAX // 2,
    scale=SAMPLE_GUIDANCE,
    neg_label=UNCOND_LABEL,
):

    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(device)

    model.eval()
    B = img.shape[0]

    # forward
    t = torch.full((B,), T_MAX - 1, dtype=torch.long, device=device)
    xt, _ = model.ddpm.forward(img, t)
    xt = xt.to(device)

    # backward: first half—denoise regularly
    orig_labels = torch.full((B,), orig_label, dtype=torch.long, device=device)
    diffused, _ = model.ddpm.backward(
        xt,
        model,
        orig_labels,
        steps=steps_first,
        method="cfg",
        scale=scale,
        neg_label=neg_label,
    )

    # backward: second half—denoise to interpolated class
    denoise_labels = torch.full((B,), denoise_label, dtype=torch.long, device=device)
    diffused, _ = model.ddpm.backward(
        diffused,
        model,
        denoise_labels,
        steps=steps_second,
        method="cfg",
        scale=scale,
        neg_label=neg_label,
    )

    imgs = vutils.make_grid(diffused, normalize=True)
    plt.imshow(imgs.permute(1, 2, 0).cpu())
    plt.show()
    return imgs
    pass
