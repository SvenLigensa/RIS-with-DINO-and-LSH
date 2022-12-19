# I, Sven Ligensa, modified this file to fit
# the needs of this project.
# Original repository: https://github.com/facebookresearch/dino
#
# Copyright (c) 2022 Sven Ligensa
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import cv2
import random
import colorsys
from random import randrange
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window

import src.models.vision_transformer as vits


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """Generate random colors."""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def visualize_attention_heads(arg_arch,
                              arg_patch_size,
                              arg_pretrained_weights,
                              arg_checkpoint_key,
                              arg_image_size,
                              arg_output_dir,
                              arg_threshold = None,
                              arg_patch = None,
                              arg_data_directory = os.getenv('PREPROCESSED_DATA_DIR')):
    """Generates visualizations of the last multihead self-attention layer."""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Build model
    model = vits.__dict__[arg_arch](patch_size=arg_patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(arg_pretrained_weights):
        state_dict = torch.load(arg_pretrained_weights, map_location="cpu")
        if arg_checkpoint_key is not None and arg_checkpoint_key in state_dict:
            print(f"Take key {arg_checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[arg_checkpoint_key]
        # Remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # Remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(arg_pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")

    # Determine path to patch
    if arg_patch is not None:  # We are given a specific patch
        path = os.path.join(arg_data_directory, str(arg_patch)+'.npy')
    else:  # We choose a random patch
        indices = np.load(os.getenv('RANDOM_INDICES_PATH'))
        index = randrange(len(indices))
        image_index = int(indices[index])    
        path = os.path.join(arg_data_directory, str(index)+'.npy')
    print(f"Looking at image patch with index at {path}")

    # Load patch as PIL Image
    array = np.load(path)
    array = np.moveaxis(array, 0, -1)
    PIL_image = Image.fromarray(array)
    
    # Transform the image to work as input for the ViT
    transform = pth_transforms.Compose([
        pth_transforms.Resize(arg_image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(PIL_image)

    # Make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % arg_patch_size, img.shape[2] - img.shape[2] % arg_patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // arg_patch_size
    h_featmap = img.shape[-1] // arg_patch_size

    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1] # Number of head

    # Keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if arg_threshold is not None:
        # Keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - arg_threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # Interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=arg_patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=arg_patch_size, mode="nearest")[0].cpu().numpy()

    # Save attentions heatmaps
    os.makedirs(arg_output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(arg_output_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(arg_output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    if arg_threshold is not None:
        image = skimage.io.imread(os.path.join(arg_output_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(arg_output_dir, "mask_th" + str(arg_threshold) + "_head" + str(j) +".png"), blur=False)
