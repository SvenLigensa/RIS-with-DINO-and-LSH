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
import timeit
import falconn
import numpy as np

class LSH():
    """Class to perform LSH on a dataset."""

    def __init__(self,
                 features_path,
                 print_details = True,
                 seed = 119417657):
        """Generates the index structure."""

        # Load the array of feature vectors
        features = np.load(features_path)
        # Convert from float64 to float32
        features = features.astype(np.float32)
        # Center the dataset
        features -= np.mean(features, axis=0)
        # Normalize all lengths
        features /= np.linalg.norm(features, axis=1).reshape(-1,1)

        # Print dimensions of features
        n, d = features.shape[0], features.shape[1]
        if print_details:
            print(f"n = {n}\nd = {d}")

        # Set params for LSHIndex
        params = falconn.LSHConstructionParameters()
        # Dimension of the dataset
        params.dimension = d
        # Used LSH family
        # Alternative: Hyperplane
        params.lsh_family = falconn.LSHFamily.CrossPolytope
        # Distance function used to select NNs among filtered points
        # Alternative: NegativeInnerProduct
        params.distance_function = falconn.DistanceFunction.EuclideanSquared
        # Way of storing low-level hash tables
        # Alternative: FlatHashTable
        params.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        # Use all available threads for setup
        params.num_setup_threads = 0

        # Number of hash tables tables ("bands")
        params.l = 10
        # Number of hash bits
        # 2^20  is relatively close to 400000 => approximately twice many buckets as instances
        number_of_hash_bits = 20
        # Helper function to set `k` and `last_cp_dimension` automatically
        # so that #buckets = 2^{number_of_hash_bits}.
        # k: Number of hash functions per table ("rows")
        # last_cp_dimension: dimension of the last cross-polytope
        falconn.compute_number_of_hash_functions(number_of_hash_bits, params)
        if print_details:
            print(f"Rows (#hash functions per table) = {params.k}\n"
                  f"Bands (#hash tables) = {params.l}\n"
                  f"Dimension of last Cross-Polytope = {params.last_cp_dimension}")

        # Number of pseudo-random rotations
        params.num_rotations = 2
        # Random seed used in hash functions
        params.seed = seed ^ 833840234

        # Further optional parameter for Cross-Polytope LSH:
        # feature_hashing_dimension: intermediate hashing dimension (d')

        # Stop the time it takes to generate the index
        start = timeit.default_timer()
        # Initialize instance of LSH data structure
        lsh_table = falconn.LSHIndex(params)
        # Build the LSH datastructure for our feature vectors
        lsh_table.setup(features)
        # The query object can be used to query the LSH data structure
        self.qo = lsh_table.construct_query_object()

        # Set #probes to make for all hash tables together
        # 20 probes (multiprobe) for every table
        self.qo.set_num_probes(20 * params.l)
        stop = timeit.default_timer()
        construction_time = stop - start
        if print_details:
            print(f"Construction time: {construction_time} seconds.")

    def knn(self, instance, k):
        """Returns the indices of the k nearest neighbors of the given query instance."""
        res = self.qo.find_k_nearest_neighbors(instance, k)
        return res
