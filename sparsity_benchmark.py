
import os
import tqdm
import json
import tempfile

import numpy as np
from common_funcs import *

# For varying degrees of sparsity, track any influence on the compression for a fixed compression scheme.

if __name__ == '__main__':
  
  fname              = os.path.join(tempfile.gettempdir(), "bench_sparsity.h5")
  rng                = np.random.default_rng(seed=42)
  random_data        = rng.uniform(low=-3, high=20, size=(512, 512, 512)).astype('f')
  chunk_size         = tuple(32 for _ in range(3))
  compression_method = hdf5plugin.Blosc2(cname='lz4', clevel=9, filters=hdf5plugin.Blosc2.SHUFFLE)
  n_tiles            = [3,2,2]
  read_roi_shape     = (50,250,200)
  volume_size        = np.multiply(random_data.shape, n_tiles)
  n_rois             = 10
  
  benchmark_data = {}
  for sparsity in tqdm.tqdm(np.linspace(0,1,num=10,endpoint=True)):
    sparsified_data = sparsify_data(random_data, sparsity)
    
    write_secs = time_secs(write_file_single, fname, sparsified_data, n_tiles, chunk_size, compression_method)
    
    roi_z = rng.integers(0, volume_size[0]-read_roi_shape[0], size=n_rois)
    roi_y = rng.integers(0, volume_size[1]-read_roi_shape[1], size=n_rois)
    roi_x = rng.integers(0, volume_size[2]-read_roi_shape[2], size=n_rois)
    read_secs = np.zeros(n_rois, dtype=np.float64)
    with h5py.File(fname, 'r') as f:
      for roi_ix, roi_origin in tqdm.tqdm(enumerate(zip(roi_z, roi_y, roi_x)), leave=False):
        read_secs[roi_ix] = time_secs(read_roi_single, f, roi_origin, read_roi_shape)
    
    file_size_MiB = np.float64(os.path.getsize(fname)) / 2**20
    
    benchmark_data[sparsity] = {
      'compression_factor'    : compression_factor(file_size_MiB, volume_size),
      'write_throughput_MiBps': throughput(np.prod(volume_size), write_secs),
      'read__throughput_MiBps': throughput(np.prod(read_roi_shape), np.mean(read_secs))
    }
    os.remove(fname)
  
  if not os.path.exists('results'):
    os.mkdir('results')
  
  with open("results/h5_sparsity_benchmark.json", "w") as bench_file:
    json.dump(benchmark_data, bench_file, indent=2)
  
