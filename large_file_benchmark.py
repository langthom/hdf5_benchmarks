
import os
import tqdm
import json
import tempfile
import itertools
import h5py
import hdf5plugin

from common_funcs import *

# For varying file sizes of similar content, check how the different compressions perform.
# Expectation for the used ones: Approximately same compressions and access speeds irregardless of size.


if __name__ == '__main__':
  
  # Where to write a file.
  fname = os.path.join(tempfile.gettempdir(), "bench_large.h5")
  
  # The data to be written and read from the files. Same input for everything.
  # Depending on the sparsity, random voxels are set to zero
  rng             = np.random.default_rng(seed=42)
  random_data     = rng.uniform(low=-3, high=20, size=(512, 512, 512)).astype('f')
  sparsified_data = sparsify_data(random_data, sparsity=0.66)
  
  if False:
    read_roi_shape  = (256, 512, 512)
    for n_tiles in [2]:
      upper = np.subtract(np.multiply(random_data.shape, n_tiles), read_roi_shape)
      for cs in [32,64,128,256]:
        for cn, c in [ ("lz4_1gb",hdf5plugin.LZ4(nbytes=0)), ("blosc2_shuffle",hdf5plugin.Blosc2(cname='lz4', clevel=9, filters=hdf5plugin.Blosc2.SHUFFLE)), ("blosc2_bitshuffle",hdf5plugin.Blosc2(cname='lz4', clevel=9, filters=hdf5plugin.Blosc2.BITSHUFFLE)) ]:
          
          t = time_secs(write_file, fname, sparsified_data, [n_tiles]*3, chunk_sizes=(cs, cs, cs), compression_method=c)
          
          N  = 256
          ts = np.zeros(N)
          z  = rng.integers(low=0, high=upper[0], size=N)
          y  = rng.integers(low=0, high=upper[1], size=N)
          x  = rng.integers(low=0, high=upper[2], size=N)
          
          cache_size = int(np.ceil(float(cs**3 * 4.0) / 2**20)) * 2**20 # at least one chunk, round up to full MiB, in [bytes]
          with h5py.File(fname, 'r', rdcc_nbytes=cache_size) as f:
            for roi_ix, roi_origin in enumerate(zip(z, y, x)):
              ts[roi_ix] = time_secs(read_roi, f, roi_origin, read_roi_shape)
          
          s = np.float64(os.path.getsize(fname)) / 2**30
          print(f"#tiles = {n_tiles} || compr: {cn:18s} + chunk size {cs:3d} | size: {round(s,2):.2f} [GiB] | write time = {round(t,1)} [s]; roi read time = {round(np.mean(ts), 1)} [s]")
          
          os.remove(fname)
    exit(1)
  
  
  chunk_size = tuple(64 for _ in range(3))
  cache_size = int(np.ceil(float(chunk_size[0]**3 * 4.0) / 2**20)) * 2**20 # at least one chunk, round up to full MiB, in [bytes]
  
  COMPRESSION_METHODS = [
    ('Blosc2_LZ4HC_L5', hdf5plugin.Blosc2(cname='lz4hc', clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE)),
    ('LZ4_1GiB',        hdf5plugin.LZ4(nbytes=0)),
  ]
  
  read_roi_shape = (512,768,768)
  N_rois_to_read = 5
  
  benchmark_data = {}
  T = [4,3,2]
  for z_tiles in tqdm.tqdm(T):
    for y_tiles in tqdm.tqdm(T, leave=False):
      for x_tiles in tqdm.tqdm(T, leave=False):
        n_tiles  = [z_tiles,y_tiles,x_tiles]
        vol_size = np.multiply(sparsified_data.shape, n_tiles)
        upper    = np.subtract(vol_size, read_roi_shape)
        
        read_secs = np.zeros(N_rois_to_read, dtype=np.float64)
        rz = rng.integers(low=0, high=upper[0], size=N_rois_to_read, endpoint=False)
        ry = rng.integers(low=0, high=upper[1], size=N_rois_to_read, endpoint=False)
        rx = rng.integers(low=0, high=upper[2], size=N_rois_to_read, endpoint=False)
        
        for compr_method_name, compression_method in tqdm.tqdm(COMPRESSION_METHODS, leave=False):
          
          for dset_type in tqdm.tqdm(['single','multi'], leave=False):
            __write_fun = write_file_single if dset_type == 'single' else write_file_multi
            __read__fun = read_roi_single   if dset_type == 'single' else lambda *args: read_roi_multi(*args, tile_sizes=sparsified_data.shape)
            
            write_secs = time_secs(__write_fun, fname, sparsified_data, n_tiles, chunk_size, compression_method=compression_method)
            
            with h5py.File(fname, 'r', rdcc_nbytes=cache_size) as f:
              for roi_ix, roi_origin in tqdm.tqdm(enumerate(zip(rz, ry, rx)), leave=False):
                read_secs[roi_ix] = time_secs(__read__fun, f, roi_origin, read_roi_shape)
            
            compr_file_size = np.float64(os.path.getsize(fname)) / 2**20
            
            bd = {}
            bd['dataset_type']       = dset_type
            bd['compression_method'] = compr_method_name
            bd['compression']        = compression_factor(compr_file_size, vol_size)
            bd['write_MiBps']        = throughput(np.prod(vol_size),       write_secs)
            bd['read__MiBps']        = throughput(np.prod(read_roi_shape), np.mean(read_secs))
            benchmark_data[f'{dset_type}_{compr_method_name}_{z_tiles}{y_tiles}{x_tiles}'] = bd
          
            os.remove(fname)
  

  if not os.path.exists('results'):
    os.mkdir('results')
  
  with open("results/h5_large_file_benchmark.json", "w") as bench_file:
    json.dump(benchmark_data, bench_file, indent=2)
  
  
  # Estimate write/read times for some sizes
  for dset_type in ['single','multi']:
    for compr_method_name, _ in COMPRESSION_METHODS:
      avg_write_throughput = np.mean([ benchmark_data[k]['write_MiBps'] for k in benchmark_data if benchmark_data[k]['dataset_type'] == dset_type and benchmark_data[k]['compression_method'] == compr_method_name ])
      avg_read_throughput  = np.mean([ benchmark_data[k]['read__MiBps'] for k in benchmark_data if benchmark_data[k]['dataset_type'] == dset_type and benchmark_data[k]['compression_method'] == compr_method_name ])
      reads  = [ size / avg_read_throughput  for size in [1., 1024., 1024.*1024.] ]
      writes = [ size / avg_write_throughput for size in [1., 1024., 1024.*1024.] ]
      
      print(f"Dataset layout '{dset_type:6s}' + compression '{compr_method_name:15s}'")
      print(f"  1 [MiB]  |  read = {reads[0]:.1E} [s]  vs.  write = {writes[0]:.1E} [s]")
      print(f"  1 [GiB]  |  read = {reads[1]:.1E} [s]  vs.  write = {writes[1]:.1E} [s]")
      print(f"  1 [TiB]  |  read = {reads[2]:.1E} [s]  vs.  write = {writes[2]:.1E} [s]")
      print()
  
