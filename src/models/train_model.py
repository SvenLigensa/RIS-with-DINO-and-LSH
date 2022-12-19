# I, Sven Ligensa, modified this file to fit
# the needs of this project. I most removed features
# specifically required for parallelization.
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
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms

import src.models.model_utility as model_utility
import src.models.vision_transformer as vits
from src.models.vision_transformer import DINOHead
from src.data.denmark_dataset import DenmarkDataset


def train_dino(arg_seed = 0,
               arg_global_crops_scale = (0.4, 1.),
               arg_local_crops_scale = (0.05, 0.4),
               arg_local_crops_number = 8,
               arg_max_image_size = 400000,
               arg_batch_size = 64,
               arg_num_workers = 10,
               arg_arch = "vit_small",
               arg_patch_size = 16,
               arg_drop_path_rate = 0.1,
               arg_out_dim = 65536,
               arg_use_bn_in_head = False,
               arg_norm_last_layer = True,
               arg_warmup_teacher_temp = 0.04,
               arg_teacher_temp = 0.04,
               arg_warmup_teacher_temp_epochs = 0,
               arg_epochs = 100,
               arg_optimizer = "adamw",
               arg_use_fp16 = True,
               arg_lr = 0.0005,
               arg_min_lr = 1e-6,
               arg_warmup_epochs = 10,
               arg_weight_decay = 0.04,
               arg_weight_decay_end = 0.4,
               arg_momentum_teacher = 0.996,
               arg_output_dir = os.getenv('TRAINED_MODEL'),
               arg_saveckp_freq = 20,
               arg_clip_grad = 3.0,
               arg_freeze_last_layer = 1):
    """Trains the DINO model with the given parameters."""

    model_utility.fix_random_seeds(arg_seed)  # Set all seeds
    cudnn.benchmark = True  # cudnn looks for optimal set of algorithms for particular configuration

    # ============ Preparing data ... ============
    transform = DataAugmentationDINO(
        arg_global_crops_scale,
        arg_local_crops_scale,
        arg_local_crops_number)
    dataset = DenmarkDataset(size=arg_max_image_size,transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=arg_batch_size,
        shuffle = False,
        num_workers=arg_num_workers,
        pin_memory=True,
        drop_last=True)
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ Building student and teacher networks ... ============
    # Model options: vit_tiny, vit_small, vit_base
    if arg_arch in vits.__dict__.keys():
        student = vits.__dict__[arg_arch](
            patch_size=arg_patch_size,
            drop_path_rate=arg_drop_path_rate,  # Student with drop-connections (stochastic depth)
        )
        teacher = vits.__dict__[arg_arch](
            patch_size=arg_patch_size
        )
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {arg_arch}.")
        sys.exit(1)

    # Multi-crop wrapper handles forward with inputs of different resolutions
    student = model_utility.MultiCropWrapper(student, DINOHead(
        embed_dim,
        arg_out_dim,
        use_bn=arg_use_bn_in_head,
        norm_last_layer=arg_norm_last_layer,
    ))
    teacher = model_utility.MultiCropWrapper(teacher, DINOHead(
        embed_dim,
        arg_out_dim,
        use_bn=arg_use_bn_in_head),
    )

    # Move networks to GPU
    student, teacher = student.cuda(), teacher.cuda()
    teacher_without_ddp = teacher

    # Teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.state_dict())

    # There is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Student and Teacher are built: they are both {arg_arch} network.")

    # ============ Preparing loss ... ============
    dino_loss = DINOLoss(
        arg_out_dim,
        arg_local_crops_number + 2,  # Total number of crops = 2 global crops + local_crops_number
        arg_warmup_teacher_temp,
        arg_teacher_temp,
        arg_warmup_teacher_temp_epochs,
        arg_epochs,
    ).cuda()

    # ============ Preparing optimizer ... ============
    params_groups = model_utility.get_params_groups(student)
    if arg_optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    # For mixed precision training
    fp16_scaler = None
    if arg_use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ Init schedulers ... ============
    lr_schedule = model_utility.cosine_scheduler(
        arg_lr * arg_batch_size / 256.,  # Linear scaling rule
        arg_min_lr,
        arg_epochs,
        len(data_loader), # = len(dataset)/batch_size
        warmup_epochs=arg_warmup_epochs,
    )
    wd_schedule = model_utility.cosine_scheduler(
        arg_weight_decay,
        arg_weight_decay_end,
        arg_epochs,
        len(data_loader),
    )
    # Momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = model_utility.cosine_scheduler(
        arg_momentum_teacher,
        1,
        arg_epochs,
        len(data_loader),
    )
    print(f"Loss, optimizer and schedulers ready.")

    # ============ Optionally resume training ... ============
    to_restore = {"epoch": 0}
    model_utility.restart_from_checkpoint(
        os.path.join(arg_output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training.")

    for epoch in range(start_epoch, arg_epochs):

        # ============ Training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, arg_epochs, arg_clip_grad, arg_freeze_last_layer)

        # ============ Writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        torch.save(save_dict, os.path.join(arg_output_dir, 'checkpoint.pth'))
        if arg_saveckp_freq and epoch % arg_saveckp_freq == 0:
            torch.save(save_dict, os.path.join(arg_output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        with (Path(arg_output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, arg_epochs, arg_clip_grad, arg_freeze_last_layer):
    """Trains the model for one epoch."""

    metric_logger = model_utility.MetricLogger(delimiter="  ")
    header = "Epoch: [{epoch}/{arg_epochs}]"
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # Update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # Only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        # Move images to GPU
        images = [im.cuda(non_blocking=True) for im in images]
        # Teacher and student forward passes and compute DINO loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # Only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training", force=True)
            sys.exit(1)

        # Student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if arg_clip_grad:
                param_norms = model_utility.clip_gradients(student, arg_clip_grad)
            model_utility.cancel_gradients_last_layer(epoch, student,
                                              arg_freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if arg_clip_grad:
                fp16_scaler.unscale_(optimizer)  # Unscale the gradients of optimizer's assigned params in-place
                param_norms = model_utility.clip_gradients(student, arg_clip_grad)
            model_utility.cancel_gradients_last_layer(epoch, student,
                                              arg_freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # Momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    """Defines and calculates the DINO loss."""

    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # Apply a warm up for the teacher temperature because a too
        # high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """Cross-entropy between softmax outputs of the teacher and student networks."""

        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output."""

        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    """Creates global and local crops of a given image (multi-crop strategy)."""

    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # First global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.Resampling.BICUBIC),
            flip_and_color_jitter,
            model_utility.GaussianBlur(1.0),
            normalize,
        ])
        # Second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.Resampling.BICUBIC),
            flip_and_color_jitter,
            model_utility.GaussianBlur(0.1),
            model_utility.Solarization(0.2),
            normalize,
        ])
        # Transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.Resampling.BICUBIC),
            flip_and_color_jitter,
            model_utility.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        """Applies the multi-crop strategy."""

        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
