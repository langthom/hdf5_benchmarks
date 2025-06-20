
import time
import itertools
import h5py
import hdf5plugin
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------------

def sparsify_data(data, sparsity):
  N = int(data.size * sparsity)
  d = np.copy(data)
  d.ravel()[:N] = 0
  return d


def throughput(size, secs): # float data; [MiB/s]
  return np.around(size * 4 / 2**20 / secs, decimals=1)


def time_secs(func, *args, **kwargs):
  begin = time.perf_counter_ns()
  func(*args, **kwargs)
  end = time.perf_counter_ns()
  return np.float64(end - begin) / 1e9


def compression_factor(compressed_size_MiB, vol_size):
  vol_rek_size = np.prod(vol_size) * 4 + 2048 # float data, in [bytes]
  vol_rek_size_MiB = np.float64(vol_rek_size) / 2**20
  return np.around(vol_rek_size_MiB / compressed_size_MiB, decimals=1)

# ---------------------------------------------------------------------------------------------------------------------------

def write_file_single(fname, tile_data, n_tiles, chunk_sizes, compression_method):
  with h5py.File(fname, 'w') as f:
    dset = f.create_dataset('data', np.multiply(tile_data.shape, n_tiles), dtype='f', chunks=chunk_sizes, shuffle=True, compression=compression_method)
    
    for z in range(n_tiles[0]):
      for y in range(n_tiles[1]):
        for x in range(n_tiles[2]):
          beg = np.multiply([z, y, x], tile_data.shape)
          end = np.add(beg, tile_data.shape)
          dset[tuple(slice(beg[i], end[i], None) for i in range(3))] = tile_data
  

def read_roi_single(f, roi_origin, roi_dimension):
  return f['data'][tuple(slice(roi_origin[i], roi_origin[i]+roi_dimension[i], None) for i in range(3))]

# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

def write_file_multi(fname, tile_data, n_tiles, chunk_sizes=(32, 32, 32), compression_method=hdf5plugin.LZ4(nbytes=512*1024*1024)):
  with h5py.File(fname, 'w') as f:
    for z in range(n_tiles[0]):
      for y in range(n_tiles[1]):
        for x in range(n_tiles[2]):
          dset = f.create_dataset(f'data{z}{y}{x}', tile_data.shape, dtype='f', chunks=chunk_sizes, shuffle=True, compression=compression_method)
          dset[:] = tile_data


def calculate_slices(roi_origin, roi_dim, tile_sizes):
  roi_tile_beg = np.floor_divide(roi_origin,                    tile_sizes).astype(int)
  roi_tile_end = np.floor_divide(np.add(roi_origin, roi_dim)-1, tile_sizes).astype(int)
  num_tiles    = roi_tile_end - roi_tile_beg + 1
  
  sb = np.mod(roi_origin,                                               tile_sizes).astype(int)
  se = np.mod(np.add(roi_origin, roi_dim) - (num_tiles-1) * tile_sizes, tile_sizes).astype(int) + 1
  
  tile_strs = []
  slices = []
  
  for axis in range(len(roi_origin)):
    slices_per_axis = [slice(sb[axis], se[axis] if num_tiles[axis] == 1 else None, None)]
    tile_strs_axis  = [roi_tile_beg[axis]]
    
    if num_tiles[axis] > 1:
      for t in range(roi_tile_beg[axis]+1, roi_tile_end[axis]):
        slices_per_axis.append(slice(None))
        tile_strs_axis.append(t)
      slices_per_axis.append(slice(0, se[axis], None))
      tile_strs_axis.append(roi_tile_end[axis]-1)
    
    slices.append(slices_per_axis)
    tile_strs.append(tile_strs_axis)
  
  all_slices = itertools.product(*slices)
  all_tiles  = itertools.product(*tile_strs)
  return zip(all_slices, [ f"data{z}{y}{x}" for z,y,x in all_tiles ])


def read_roi_multi(f, roi_origin, roi_dimension, tile_sizes):
  data = []
  for file_slice, tile_str in calculate_slices(roi_origin, roi_dimension, tile_sizes):
    data.append(f[tile_str][file_slice])
  return data

