
import json
import re
import numpy as np
import matplotlib.pyplot as plt

"""
{
  "single_Blosc2_LZ4HC_L5_444": {
    "dataset_type": "single",
    "compression_method": "Blosc2_LZ4HC_L5",
    "compression": 1.4,
    "write_MiBps": 30.5,
    "read__MiBps": 454.2
  },
  ...
"""

def large_file_plots(bench_data_fname):
  with open(bench_data_fname, 'r') as f:
    bench_data = json.load(f)
  
  # 1. throughputs over size 
  size_pattern = re.compile(r'[a-zA-Z0-9_]+_(\d)(\d)(\d)')
  def __get_size(key):
    m = size_pattern.match(key)
    t = list(map(int, m.group(1, 2, 3)))
    return np.prod(np.multiply(t, 512)) * 4 / 2**30 # [GiB]
  
  def __filter(method, dset_type, access_type):
    throughputs = {}
    for key in bench_data:
      if bench_data[key]['dataset_type'] == dset_type and bench_data[key]['compression_method'] == method:
        throughputs[__get_size(key)] = bench_data[key][access_type]
    return [ throughputs[k] for k in sorted(throughputs.keys()) ]
  
  sizes            = sorted(np.unique([__get_size(k) for k in bench_data]))
  lz4_read_single  = __filter('LZ4_1GiB', 'single', 'read__MiBps')
  lz4_read_multi   = __filter('LZ4_1GiB', 'multi',  'read__MiBps')
  lz4_write_single = __filter('LZ4_1GiB', 'single', 'write_MiBps')
  lz4_write_multi  = __filter('LZ4_1GiB', 'multi',  'write_MiBps')
  
  
  plt.figure(figsize=(15,10))
  for method in ['LZ4_1GiB', 'Blosc2_LZ4HC_L5']:
    for dset_type in ['single', 'multi']:
      for access_type in ['read__MiBps', 'write_MiBps']:
        plt.plot(sizes, __filter(method, dset_type, access_type), label=f'{'      lz4' if method == 'LZ4_1GiB' else 'Blosc2'} ({dset_type:>6s}; {'read' if access_type == 'read__MiBps' else 'write'})')
  plt.xlabel("Dataset size [GiB]")
  plt.ylabel("Throughput [MiB/s]")
  plt.legend()
  plt.tight_layout()
  plt.savefig("results/large_files__lz4_vs_blosc.png", dpi=600, bbox_inches=0)


if __name__ == '__main__':
  import sys
  large_file_plots(sys.argv[1])
