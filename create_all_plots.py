
import os
from benchmark_plots  import plot_results
from large_file_plots import large_file_plots
from sparsity_plots   import sparsity_plots

if __name__ == '__main__':
  if not os.path.exists('results'):
    print('Result directory does not exist, please run a benchmark first')
    exit(1)
  
  plot_results('results/h5_compression_benchmark_sparsity.json', '_single_dataset')
  plot_results('results/h5_compression_benchmark_sparsity_multiple_datasets.json', '_multiple_datasets')
  large_file_plots('results/h5_large_file_benchmark.json')
  sparsity_plots('results/h5_sparsity_benchmark.json')
  
