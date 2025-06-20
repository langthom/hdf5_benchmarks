
import json
import matplotlib.pyplot as plt

def sparsity_plots(bench_data_fname):
  with open(bench_data_fname, 'r') as f:
    bench_data = json.load(f)
  
  sparsities          = list(map(float, bench_data.keys()))
  write_throughputs   = [ bench_data[k]['write_throughput_MiBps'] for k in bench_data ]
  read__throughputs   = [ bench_data[k]['read__throughput_MiBps'] for k in bench_data ]
  compression_factors = [ bench_data[k]['compression_factor'] for k in bench_data ]
  
  # Plot of throughputs over the data sparsity
  plt.figure(figsize=(8,8))
  plt.plot(sparsities, write_throughputs, label='write')
  plt.plot(sparsities, read__throughputs, label='read')
  for i in range(len(sparsities)):
    plt.text(sparsities[i]-0.02, max(write_throughputs[i],read__throughputs[i]) + 10, f'{compression_factors[i]:.1f}')
  plt.xticks(sparsities, [ f'{int(round(sparsity*100.)):d}' for sparsity in sparsities ])
  plt.xlabel('Sparsity [%]')
  plt.ylabel('Throughput [MiB/s]')
  plt.title('Throughputs depending on sparsity\n(annotations give compression factors)')
  plt.legend()
  plt.tight_layout()
  plt.savefig('results/sparsity_throughputs.png', dpi=600, bbox_inches=0)
  plt.clf()
  
  

if __name__ == '__main__':
  import sys
  sparsity_plots(sys.argv[1])
  