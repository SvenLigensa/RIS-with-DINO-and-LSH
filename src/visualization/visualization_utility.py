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
import matplotlib.pyplot as plt


def generate_image(arr_paths, num_rows, num_cols, fig_size_factor=20, out_file = None):
    """Generates a single image from multiple numpy arrays which are interpreted as images."""
    
    assert min(num_rows, num_cols) > 0, "num_rows and num_cols has to be greater than zero."
    assert num_rows*num_cols == len(arr_paths), "num_rows*num_cols should equal the length of the array."
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_size_factor, fig_size_factor*(num_rows/num_cols)))
    
    if min(num_rows, num_cols) == 1:
        [axes[i].imshow(np.moveaxis(np.load(arr_paths[i]), 0, -1)) for i in range(max(num_rows, num_cols))]
    
    else:
        for i in range(num_rows):
            for j in range(num_cols):
                axes[i,j].imshow(np.moveaxis(np.load(arr_paths[i*num_cols+j]), 0, -1))
            
    # Do not show axes of subplots
    for ax in axes.reshape(-1):
        ax.axis('off')
    
    # Less spacing between patches
    fig.tight_layout()
    
    # Save the example patches
    if out_file is not None:
        plt.savefig(out_file)


def create_paths(root, model, category, index):
    
    nn_candidates = np.load(os.path.join(root, model, 'nn_'+category+'_candidates.npy'))
    num_candidates = nn_candidates.shape[1]
    mapped_indices = [nn_candidates[index][i] for i in range(num_candidates)]
    paths = [os.path.join(os.getenv('PREPROCESSED_DATA_DIR'), str(mapped_indices[i])+'.npy') for i in range(num_candidates)]
    return paths
