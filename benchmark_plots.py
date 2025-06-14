
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
  
  plt.figure(figsize=(20,15))
  
  for method_ix, method in enumerate(methods):
    for chunk_ix, chunk_size in enumerate(chunk_sizes):
      _sizes = [ (float(sparsity) + 0.5) * 100 for sparsity in sparsities ]
      X      = [ bench_data[method][sparsity][chunk_size]['throughputs_MiBps']['read_roi_random'] for sparsity in sparsities ]
      Y      = [ bench_data[method][sparsity][chunk_size]['throughputs_MiBps']['write']           for sparsity in sparsities ]
      plt.scatter(X, Y, marker=chunk_markers[chunk_ix], s=_sizes, color=method_colors[method_ix], label=f"{method} @ {chunk_size}")
  
  plt.gca().set_aspect('equal')
  plt.xlim(0, plt.xlim()[1])
  plt.ylim(0, plt.ylim()[1])
  plt.title("Read/Write throughputs in [MiB/s]\n(marker size indicates sparsity)")
  plt.xlabel("Read throughput [MiB/s]")
  plt.ylabel("Write throughput [MiB/s]")
  plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=len(methods)//2)
  plt.tight_layout()
  plt.savefig('results/throughput_plot.png', dpi=600, bbox_inches=0)



def compression_plots(bench_data):
  # One plot per data sparsity
  #   X: compression method 
  #   Y: compression factor 
  #   per method (X): 1 bar per chunk method
  
  methods     = list(bench_data)
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
      read_throughputs = [ bench_data[method][sparsity][chunk_size]['throughputs_MiBps']['read_roi_random'] for method in methods ]
      ax.bar_label(rects, labels=read_throughputs, padding=3, rotation='horizontal', fontsize='x-small')
      

    ax.set_title("Compression factors and read throughput (random position; [MiB/s])")
    ax.set_ylabel("Compression factor")
    ax.set_xticks(X + (len(chunk_sizes)-1) * bar_width/2, methods, rotation=22)
    ax.legend(loc='upper right')
    plt.savefig(f'results/compression_{sparsity}.png', dpi=600, bbox_inches=0)


  # Heatmap plots (one for each sparsity)
  #   o compression method vs. chunk sizes
  #   o each cell is diagonally split 
  #       - half 1: write throughput  (color coding)
  #       - half 2: random Roi read throughput (color coding)
  #       - annotation text: compression factor
  #
  # See: https://stackoverflow.com/a/63531813/10696884 
  for sparsity in sparsities:
    fig, ax = plt.subplots(figsize=(20,10))
    
    M = len(methods)
    N = len(chunk_sizes)
    
    X = np.arange(M + 1)
    Y = np.arange(N + 1)
    Xs, Ys = np.meshgrid(X, Y)
    
    # create a triangulation, i.e., a representation of all triangles, for write and read throughputs 
    triangles_write     = [(i + j*(M+1), i+1 + j*(M+1), i + (j+1)*(M+1)) for j in range(N) for i in range(M)]
    triangulation_write = matplotlib.tri.Triangulation(Xs.ravel()-0.5, Ys.ravel()-0.5, triangles_write)
    
    triangles_read     = [(i+1 + j*(M+1), i+1 + (j+1)*(M+1), i + (j+1)*(M+1)) for j in range(N) for i in range(M)]
    triangulation_read = matplotlib.tri.Triangulation(Xs.ravel()-0.5, Ys.ravel()-0.5, triangles_read)
    
    # create (for each write and read) a triangular plot 
    write_throughput_values = [ bench_data[method][sparsity][chunk_size]['throughputs_MiBps']['write']           for chunk_size in chunk_sizes for method in methods ]
    read_throughput_values  = [ bench_data[method][sparsity][chunk_size]['throughputs_MiBps']['read_roi_random'] for chunk_size in chunk_sizes for method in methods ]
    
    img_write = plt.tripcolor(triangulation_write, write_throughput_values, cmap='winter')
    img_read  = plt.tripcolor(triangulation_read,  read_throughput_values,  cmap='summer')
    ax.set_aspect('equal')
    
    # color bars
    cbar_ax_write = ax.inset_axes((1.03, 0, 0.02, 1.0))
    plt.colorbar(img_write, cax=cbar_ax_write)
    cbar_ax_write.set(title='write')
    cbar_ax_write.yaxis.set_ticks_position('left')
    
    cbar_ax_read = ax.inset_axes((1.06, 0, 0.02, 1.0))
    plt.colorbar(img_read, cax=cbar_ax_read)
    cbar_ax_read.set(title='read')
    
    # text annotations with the compression factor
    for method_ix, method in enumerate(methods):
      for chunk_size_ix, chunk_size in enumerate(chunk_sizes):
        ax.text(method_ix, chunk_size_ix, bench_data[method][sparsity][chunk_size]['compression'], ha='center', va='center', color='black', fontsize='x-large')
    
    # finish plot 
    plt.xlim(X[0]-0.5, X[-1]-0.5)
    plt.ylim(Y[0]-0.5, Y[-1]-0.5)
    plt.xticks(X[:-1], methods)
    plt.yticks(Y[:-1], chunk_sizes)
    plt.xlabel("Compression method")
    plt.ylabel("Chunk size (side length)")
    plt.title(f"Write/read throughputs (color-coded; [MiB/s]) and compression factors (annotations) for sparsity {sparsity}")
    plt.tight_layout()
    plt.savefig(f'results/throughputs_and_compressions_{sparsity}.png', dpi=600, bbox_inches=0)



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
  