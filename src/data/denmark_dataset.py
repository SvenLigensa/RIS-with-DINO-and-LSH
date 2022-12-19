# Copyright (c) 2022 Sven Ligensa
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
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class DenmarkDataset(Dataset):
    """Provides access to the Denmark dataset. To be used in model training."""
    
    def __init__(self,
                 directory = os.getenv('PREPROCESSED_DATA_DIR'),
                 size: int = 400000,
                 transform = None,
                 target_transform = None):
        
        self.directory = directory
        self.length = size
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        path = os.path.join(self.directory, str(idx)+'.npy')
        #print(f"Opening patch at path {path}")
        array = np.load(path)
        array = np.moveaxis(array, 0, -1)    # Previous first axis (RGB) becomes last axis (required by PIL)
        PIL_image = Image.fromarray(array)
        if self.transform: PIL_image = self.transform(PIL_image)
        return PIL_image
