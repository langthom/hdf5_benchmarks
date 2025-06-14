
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""
json layout:

  compression method
    |-> sparsity
          |-> chunk size
                |-> file size in MiB
                |-> compression factor over a REK
                |-> throughputs in MiB/s
                      |-> write full file 
                      |-> read Roi of fixed size contiguously
                      |-> read Roi of fixed size random position
                      |-> read full chunk contiguously
                      |-> read full chunk random position

"""

def throughput_plots(bench_data):
  # compression methods -> by color
  # chunk sizes         -> by marker 
  # sparsities          -> by marker size
  
  methods     = list(bench_data)
  sparsities  = list(bench_data[methods[0]])
  chunk_sizes = list(bench_data[methods[0]][sparsities[0]])
  
  method_colors = matplotlib.colormaps['tab10'](np.linspace(0, 1, len(methods)))
  chunk_markers = [ matplotlib.lines.Line2D.filled_markers[i+1] for i in range(len(chunk_sizes)) ]
  
  plt.figure(figsize=(30,12))
  
  for method_ix, method in enumerate(methods):
    for chunk_ix, chunk_size in enumerate(chunk_sizes):
      _sizes = [ (float(sparsity) + 0.5) * 100 for sparsity in sparsities ]
      X      = [ bench_data[method][sparsity][chunk_size]['throughputs_MiBps']['write']           for sparsity in sparsities ]
      Y      = [ bench_data[method][sparsity][chunk_size]['throughputs_MiBps']['read_roi_random'] for sparsity in sparsities ]
      plt.scatter(X, Y, marker=chunk_markers[chunk_ix], s=_sizes, color=method_colors[method_ix], label=f"{method} @ {chunk_size}")
  
  plt.title("Read/Write throughputs in [MiB/s]\n(marker size indicates sparsity)")
  plt.ylabel("Write throughput [MiB/s]")
  plt.xlabel("Read throughput [MiB/s]")
  plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=len(methods))
  plt.tight_layout()
  plt.savefig('results/throughput_plot.png', dpi=600, bbox_inches=0)



def compression_plots(bench_data):
  # One plot per data sparsity
  #   X: compression method 
  #   Y: compression factor 
  #   per method (X): 1 bar per chunk method
  
  methods = list(bench_data)
  sparsities  = list(bench_data[methods[0]])
  chunk_sizes = list(bench_data[methods[0]][sparsities[0]])
  
  X = np.arange(len(methods)) * 3
  bar_width = 0.5
  
  for sparsity in sparsities:
    fig, ax = plt.subplots(layout='constrained', figsize=(20,6))
    
    for chunk_size_ix, chunk_size in enumerate(chunk_sizes):
      off = bar_width * chunk_size_ix
      
      Y = [ bench_data[method][sparsity][chunk_size]['compression'] for method in methods ]
      rects = ax.bar(X + off, Y, bar_width, label=chunk_size)
      
      # annotate the bars with the Roi read throughput
      read_throughputs = [bench_data[method][sparsity][chunk_size]['throughputs_MiBps']['read_roi_random'] for method in methods]
      ax.bar_label(rects, labels=read_throughputs, padding=3, rotation='horizontal', fontsize='x-small')
      

    ax.set_title("Compression factors")
    ax.set_ylabel("Compression factor")
    ax.set_xticks(X + (len(chunk_sizes)-1) * bar_width/2, methods, rotation=22)
    ax.legend(loc='upper right')
    plt.savefig(f'results/compression_{sparsity}.png', dpi=600, bbox_inches=0)


def plot_results(bench_json):
  with open(bench_json, "r") as f:
    bench_data = json.load(f)
  
  throughput_plots(bench_data)
  compression_plots(bench_data)


if __name__ == '__main__':
  import sys
  if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <benchmark-results.json>")
    exit(1)
  
  if not os.path.exists('results'):
    os.mkdir('results')
  
  plot_results(sys.argv[1])
  