# I, Sven Ligensa, modified this file (originally eval_knn.py) to fit
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
import src.models.model_utility as model_utility
import src.models.vision_transformer as vits
from src.data.denmark_dataset import DenmarkDataset

import sys
import os
from dotenv import load_dotenv
import numpy as np
import torch
from torchvision import transforms as pth_transforms

load_dotenv()
sys.path.append(os.getenv('CODE_ROOT_PATH')) # Add path


def extract_feature_pipeline(arg_pretrained_weights,
                             arg_arch = 'vit_small',
                             arg_max_image_size = 400000,
                             arg_batch_size = 128,
                             arg_num_workers = 10,
                             arg_patch_size = 16,
                             arg_checkpoint_key = 'teacher',
                             arg_use_cuda = True):
    """Builds a model with the given parameters and generates feature vectors."""

    # Preparing data
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = DenmarkDataset(size=arg_max_image_size,transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=arg_batch_size,
        shuffle=False,
        num_workers=arg_num_workers,
        pin_memory=True,
        drop_last=False)
    print(f"Data loaded: there are {len(dataset)} images.")

    # Building model
    if "vit" in arg_arch:
        model = vits.__dict__[arg_arch](patch_size=arg_patch_size, num_classes=0)
        print(f"Model {arg_arch} {arg_patch_size}x{arg_patch_size} built.")
    else:
        print(f"Unknow architecture: {arg_arch}.")
        sys.exit(1)
    model.cuda()
    model_utility.load_pretrained_weights(model, arg_pretrained_weights, arg_checkpoint_key, arg_arch, arg_patch_size)
    model.eval()

    # Extract features
    print("Extracting features.")
    features = extract_features(model, data_loader, arg_batch_size, arg_use_cuda)
    return features


@torch.no_grad()
def extract_features(model, data_loader, batch_size, use_cuda=True, multiscale=False):
    """Generates feature vectors given a trained PyTorch model."""

    metric_logger = model_utility.MetricLogger(delimiter="  ")
    features = None  # We do not know the dimensionality yet
    for it, samples in enumerate(metric_logger.log_every(data_loader, 10)):
        samples = samples.cuda(non_blocking=True)
        feats = model(samples).clone()
        if features is None:  # Init storage feature matrix
            features = np.zeros([len(data_loader.dataset), feats.shape[-1]])
            print(f"Storing features into np array of shape {features.shape}")
        features[it*batch_size:(it+1)*batch_size][:] = feats.cpu()
    return features
