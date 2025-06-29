
import os
import tqdm
import time
import json
import tempfile
import numpy as np

import h5py
import hdf5plugin

from common_funcs import *

# Consider a volume of some size
# Use chunking (to actually get compression)
# Cache size is large enough for a single chunk
# Access: (each contiguous access and random position)
#   - A Roi of size 256x256x256
#   - Exactly the chunk size
#
# Measure (1) obtained file sizes and (2) throughput depending on the (a) chunk size and (b) the compression method
  

def run(chunk_sizes, compression_method, data, sparsity, read_fixed_roi_size=(256,256,256)):
  # experiment with multiple datasets within a single file 
  # the volume is thrice as big as the read Roi size
  # reading now may involve several dataset accesses
  
  multiple        = 2
  vol_size        = np.multiply(read_fixed_roi_size, multiple)
  n_voxels        = np.prod(vol_size)
  benchmark_data  = {}
  sparsified_data = sparsify_data(data, sparsity)
  
  fname = os.path.join(tempfile.gettempdir(), "bench.h5")
  
  # Do the experiments
  
  for chunk_size in tqdm.tqdm(chunk_sizes, leave=False):
    cache_size = int(np.ceil(float(chunk_size**3 * 4.0) / 2**20)) * 2**20 # at least one chunk, round up to full MiB, in [bytes]
    cs_data = {}
    
    def _read_roi(f, z, y, x, roi_size):
      data = []
      for file_slice, tile_str in calculate_slices([z, y, x], roi_size, read_fixed_roi_size):
        data.append( f[f'data{tile_str}'][file_slice] )
      return data


    # Write the dataset
    write_time = time_secs(write_file_multi, fname, sparsified_data, [multiple]*3, tuple(chunk_size for _ in range(3)), compression_method)
    
    # Read a Roi of fixed size, contiguous 
    avg_read_time_fixed_cont, _cnt = 0, 0
    with h5py.File(fname, 'r', rdcc_nbytes=cache_size) as f:
      for z in tqdm.trange(0, multiple*2, leave=False):
        for y in tqdm.trange(0, multiple*2, leave=False):
          for x in tqdm.trange(0, multiple*2, leave=False):
            roi_origin = np.multiply([z,y,x], read_fixed_roi_size) // 2
            avg_read_time_fixed_cont += time_secs(read_roi_multi, f, roi_origin, read_fixed_roi_size, read_fixed_roi_size)
            _cnt += 1
    avg_read_time_fixed_cont /= _cnt
    
    # Read a Roi of fixed size, random location
    N = 25
    pos_z = np.random.randint(0, vol_size[0]-read_fixed_roi_size[0], N)
    pos_y = np.random.randint(0, vol_size[1]-read_fixed_roi_size[1], N)
    pos_x = np.random.randint(0, vol_size[2]-read_fixed_roi_size[2], N)
    
    avg_read_time_fixed_random = 0
    with h5py.File(fname, 'r', rdcc_nbytes=cache_size) as f:
      for (z, y, x) in tqdm.tqdm(zip(pos_z, pos_y, pos_x), leave=False):
        avg_read_time_fixed_random += time_secs(read_roi_multi, f, [z, y, x], read_fixed_roi_size, read_fixed_roi_size)
    avg_read_time_fixed_random /= N
    
    # Read the chunk size, contiguous
    average_chunk_time_cont, _cnt = 0, 0
    with h5py.File(fname, 'r', rdcc_nbytes=cache_size) as f:
      for z in tqdm.trange(0, vol_size[0], chunk_size, leave=False):
        for y in tqdm.trange(0, vol_size[1], chunk_size, leave=False):
          for x in tqdm.trange(0, vol_size[2], chunk_size, leave=False):
            average_chunk_time_cont += time_secs(read_roi_multi, f, [z,y,x], [chunk_size]*3, read_fixed_roi_size)
            _cnt += 1
    average_chunk_time_cont /= _cnt
    
    # Read the chunk size, random location
    N = 25
    pos_z = np.random.randint(0, vol_size[0]-chunk_size, N)
    pos_y = np.random.randint(0, vol_size[1]-chunk_size, N)
    pos_x = np.random.randint(0, vol_size[2]-chunk_size, N)
    
    average_chunk_time_random = 0
    with h5py.File(fname, 'r', rdcc_nbytes=cache_size) as f:
      for (z, y, x) in tqdm.tqdm(zip(pos_z, pos_y, pos_x), leave=False):
        average_chunk_time_random += time_secs(read_roi_multi, f, [z, y, x], [chunk_size]*3, read_fixed_roi_size)
    average_chunk_time_random /= N
    
    file_size_MiB = round(float(os.path.getsize(fname)) / 2**20, 1)
    
    data = {
      'file_size_MiB':           file_size_MiB,
      'compression':             compression_factor(file_size_MiB, vol_size),  # compression factor compared to a binary blob storing the same information
      'throughputs_MiBps': {
        'write':                 throughput(n_voxels,                                                             write_time),
        'read_roi_contiguous':   throughput(read_fixed_roi_size[0]*read_fixed_roi_size[1]*read_fixed_roi_size[2], avg_read_time_fixed_cont),
        'read_roi_random':       throughput(read_fixed_roi_size[0]*read_fixed_roi_size[1]*read_fixed_roi_size[2], avg_read_time_fixed_random),
        'read_chunk_contiguous': throughput(chunk_size*chunk_size*chunk_size,                                     average_chunk_time_cont),
        'read_chunk_random':     throughput(chunk_size*chunk_size*chunk_size,                                     average_chunk_time_random)
      }
    }
    benchmark_data[chunk_size] = data
    
    # Remove created file
    os.remove(fname)

  return benchmark_data
  
  
# ---------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
  
  METHODS = [
    ('zstd_l9',             hdf5plugin.Zstd(clevel=9)),
    ('zstd_l22',            hdf5plugin.Zstd(clevel=22)),
    ('lz4_512MiB',          hdf5plugin.LZ4(nbytes=512*1024*1024)),
    ('lz4_1GiB',            hdf5plugin.LZ4(nbytes=1024*1024*1024)),
    ('blosc2_blosclz_l5',   hdf5plugin.Blosc2(cname='blosclz', clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE)),
    ('blosc2_blosclz_l9',   hdf5plugin.Blosc2(cname='blosclz', clevel=9, filters=hdf5plugin.Blosc2.SHUFFLE)),
    ('blosc2_lz4_l5',       hdf5plugin.Blosc2(cname='lz4',     clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE)),
    ('blosc2_lz4_l9',       hdf5plugin.Blosc2(cname='lz4',     clevel=9, filters=hdf5plugin.Blosc2.SHUFFLE)),
    ('blosc2_lz4hc_l5',     hdf5plugin.Blosc2(cname='lz4hc',   clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE)),
    ('blosc2_lz4hc_l9',     hdf5plugin.Blosc2(cname='lz4hc',   clevel=9, filters=hdf5plugin.Blosc2.SHUFFLE)),
  ]

  chunk_sizes = [ 16, 32, 64, 128 ]
  
  # The data to be written and read from the files. Same input for everything.
  # Depending on the sparsity, random voxels are set to zero
  random_data = np.random.default_rng(seed=42).uniform(low=-3, high=20, size=tuple(256 for _ in range(3))).astype('f')
  
  benchmark_data = {}
  for method_name, method in tqdm.tqdm(METHODS):
    sparse_bench_data = {}
    
    for sparsity in tqdm.tqdm(np.linspace(0.0, 0.99, 4, endpoint=True), leave=False):
      sparse_bench_data[sparsity] = run(chunk_sizes, method, data=random_data, sparsity=sparsity)
    
    benchmark_data[method_name] = sparse_bench_data
  
  if not os.path.exists('results'):
    os.mkdir('results')
  
  with open("results/h5_compression_benchmark_sparsity_multiple_datasets.json", "w") as bench_file:
    json.dump(benchmark_data, bench_file, indent=2)
