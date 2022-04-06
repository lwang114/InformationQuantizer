import numpy as np
import argparse
import os

def average_performance(in_dirs, out_path):
  n = len(in_dirs)
  means = np.zeros((4, n))
  for i, in_dir in enumerate(in_dirs):     
    with open(os.path.join(in_dir, 'nmi'), 'r') as f:
      nmi = float(f.read().strip().split('\n')[-1].split()[-1])
      means[0][i] += nmi

    with open(os.path.join(in_dir, 'phone_boundaries'), 'r') as f:
      bf1 = float(f.read().strip().split('\n')[-1].split()[-1])
      means[1][i] += bf1

    with open(os.path.join(in_dir, 'token_f1'), 'r') as f:
      tf1 = float(f.read().strip().split('\n')[-1].split()[-1])
      means[2][i] += tf1

    with open(os.path.join(in_dir, 'eq_per'), 'r') as f:
      eq_per = float(f.read().strip().split('\n')[-1].split()[-1])
      means[3][i] += eq_per

  mean = means.mean(-1)
  std = np.std(means, axis=1)
  info = f'Average NMI: {mean[0]:.3f}+/-{std[0]:.3f}\n' \
         f'Average boundary F1: {mean[1]:.3f}+/-{std[1]:.3f}\n' \
         f'Average token F1: {mean[2]:.3f}+/-{std[2]:.3f}\n' \
         f'Average equivalent PER: {mean[3]:.3f}+/-{std[3]:.3f}'
  with open(out_path, 'w') as f:
    f.write(info)
  print(info)
  return mean, std

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--in_dirs')
  parser.add_argument('--out_path')
  args = parser.parse_args()
  in_dirs = args.in_dirs.split(',')
  means, stds = average_performance(in_dirs, args.out_path)

