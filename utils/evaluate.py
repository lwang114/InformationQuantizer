import argparse
import pandas as pd
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
import editdistance

EPS = 1e-40
SIL = 'SIL'
def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the discovered phone units.'
    )
    parser.add_argument(
        'TASK', type=int
    )
    parser.add_argument(
        '--config', type=str
    )
    return parser.parse_known_args()[0]

def compute_accuracy(reference, test):
  if len(reference) != len(test):
    raise ValueError("Lists must have the same length.")
  return sum(x == y for x, y in zip(reference, test)) / len(test)

def compute_edit_distance(predictions, targets, preprocessor=None):
  tokens_dist = 0
  n_tokens = 0
  for p, t in zip(predictions, targets):
    if preprocessor is not None:
      p, t = preprocessor.tokens_to_text(p), preprocessor.to_text(t)
    p, t = list(filter(None, p)), list(filter(None, t))
    tokens_dist += editdistance.eval(p, t)
    n_tokens += len(t)
  return tokens_dist, n_tokens 

def compute_token_f1(pred_path, gold_paths, out_path):
  """
  Compute token F1 for predictions in zerospeech 2021 format
  Args:
      pred_path : str, path to the prediction file in the format
          {sentence_id} {cluster ids separated by commas}
      gold_path : str, path to the gold phoneme transcripts in the format
          line 0 : whatever (not read)
          line > 0: {sentence_id} {onset} {offset} {phone} {prev-phone} {next-phone} {speaker}
          onset : begining of the triplet (in s)
          offset : end of the triplet (in s)
      out_path : str
  """
  def _extract_gold_units(gold_file_path):
    with open(gold_file_path, 'r') as f:
      line0 = True
      for line in f:
        if line0:
          line0 = False
          continue
        sent_id, begin, end, phn, _, _, _ = line.rstrip('\n').split()
        begin = int(round(float(begin)*100, 3))
        end = int(round(float(end)*100, 3))
        if not phn.lower() == SIL.lower():
          gold_tokens.add(phn)
          if not sent_id in gold_units:
            gold_units[sent_id] = {(begin, end): phn}
          else:
            gold_units[sent_id][(begin, end)] = phn

  if isinstance(gold_paths, str):
    gold_paths = [gold_paths]

  gold_units = dict()
  gold_tokens = set()
  for gold_path in gold_paths:
    if os.path.isdir(gold_path):
      for gold_root, gold_dirs, gold_files in os.walk(gold_path):
        if len(gold_dirs):
          continue
        else:
          nonoverlap_candidates = [gold_file for gold_file in gold_files if gold_file.split('_')[-1] == 'nonoverlap.item']
          candidates = [gold_file for gold_file in gold_files if gold_file.endswith('.item')]
          
          if len(nonoverlap_candidates):
            gold_file = nonoverlap_candidates[0]
          elif len(candidates):
            gold_file = candidates[0]
          else:
            continue
          gold_file_path = os.path.join(gold_root, gold_file)
          _extract_gold_units(gold_file_path)
    else:
      _extract_gold_units(gold_path)
    
  pred_units = dict()
  pred_tokens = set()
  with open(pred_path, 'r') as f:
    for line in f:
      parts = line.rstrip('\n').split()
      sent_id = parts[0]
      if not sent_id in gold_units:
        continue
      pred_unit = parts[1].split(',')
      pred_tokens.update(pred_unit)
      pred_units[sent_id] = dict()
      gold_unit = sorted(gold_units[sent_id])
      for i, interval in enumerate(gold_unit):
        if i == 0:
          begin = interval[0]
        else:
          begin = max(gold_unit[i-1][1], interval[0])
        
        if i == len(gold_unit) - 1:
          end = interval[1]
        else:
          end = min(gold_unit[i+1][0], interval[1])
        pred_units[sent_id][interval] = pred_unit[begin:end] 

  n_gold_tokens = len(gold_tokens)
  n_pred_tokens = len(pred_tokens)

  pred_stoi = {p:i for i, p in enumerate(sorted(pred_tokens, key=lambda x:int(x)))}
  gold_stoi = {g:i for i, g in enumerate(sorted(gold_tokens))}
  confusion = np.zeros((n_gold_tokens, n_pred_tokens))
  for sent_id in pred_units:
    for interval in pred_units[sent_id]:
      gold_unit = gold_units[sent_id][interval]
      g_idx = gold_stoi[gold_unit]
      for pred_unit in pred_units[sent_id][interval]:
        p_idx = pred_stoi[pred_unit]
        confusion[g_idx, p_idx] += 1
        
  n = confusion.sum()
  token_recall = confusion.max(1).sum() / n
  token_precision = confusion.max(0).sum() / n
  token_f1 = 2 * token_recall * token_precision /\
               (token_recall + token_precision)\
               if (token_recall + token_precision) > 0 else 0 
  print(f'Token precision: {token_precision}\t'
        f'Token recall: {token_recall}\t'
        f'Token F1: {token_f1}')

  fig, ax = plt.subplots(figsize=(8, 8))
  
  confusion_norm = confusion / np.maximum(confusion.sum(1, keepdims=True), 1.)
  new_row_order = sorted(list(range(n_gold_tokens)), key=lambda x:confusion_norm[x].max(), reverse=True)
  confusion_norm = confusion_norm[new_row_order]

  new_col_order = []
  pred_idxs = list(range(n_pred_tokens))
  for i in range(n_gold_tokens):
    if i >= n_pred_tokens: # Unable to assign when the number of gold tokens exceed the pred tokens
      break
    max_s = 0
    max_j = -1
    for j, s in enumerate(confusion_norm[i]):
      if (s >= max_s) and not j in new_col_order: # If cluster j is not used and has higher prob, update the assigned cluster
        max_j = j
        max_s = s
    new_col_order.append(max_j)
  
  for i in range(n_pred_tokens): # Append the rest of the unassigned clusters if any
    if not i in new_col_order:
      new_col_order.append(i)

  plt.pcolor(confusion_norm[:, new_col_order], cmap=plt.cm.Blues)
  ax.set_xticks(np.arange(len(pred_tokens))+0.5)
  ax.set_yticks(np.arange(len(gold_tokens))+0.5)
  pred_names = sorted(pred_stoi, key=lambda x:pred_stoi[x])
  ax.set_xticklabels([pred_names[i] for i in new_col_order], rotation='vertical')
  gold_names = sorted(gold_stoi, key=lambda x:gold_stoi[x])
  ax.set_yticklabels([gold_names[i] for i in new_row_order])
  ax.invert_yaxis()
  plt.colorbar()
  plt.savefig(out_path)
  plt.show()
  plt.close()
  return token_f1, token_precision, token_recall

def compute_token_f1_beer(pred_path, gold_path, out_path, debug=False):
  """
  Compute token F1 for predictions in beer format
  Args:
      pred_path : str, path to the prediction file in the format
          {sentence_id} {cluster ids separated by spaces}
      gold_path : str, path to the gold phoneme transcripts in the format
          {sentence_id} {phonemes separated by spaces}
      out_path : str
  """
  gold_units = dict()
  gold_tokens = set()
  with open(gold_path, 'r') as f:
    for line in f:
      if debug and len(gold_units) > 1:
        break
      parts = line.rstrip('\n').split()
      sent_id = parts[0]
      gold_units[sent_id] = dict()
      begin = 0
      phns = parts[1:]
      for phn_idx, phn in enumerate(phns):
        if phn != phns[max(phn_idx-1, 0)]:
          if phns[max(phn_idx-1, 0)] != SIL.lower():
            gold_units[sent_id][(begin, phn_idx)] = phns[phn_idx-1]
            gold_tokens.add(phns[phn_idx-1])
          begin = phn_idx 
      if phn != SIL.lower():
        gold_units[sent_id][(begin, phn_idx)] = phn
        gold_tokens.add(phn)

  pred_units = dict()
  pred_tokens = set()
  with open(pred_path, 'r') as f:
    for line in f:
      parts = line.rstrip('\n').split()
      sent_id = parts[0]
      if not sent_id in gold_units:
        continue
      pred_unit = parts[1:]
      pred_tokens.update(pred_unit)
      pred_units[sent_id] = dict()
      gold_unit = sorted(gold_units[sent_id])
      for i, interval in enumerate(gold_unit):
        if i == 0:
          begin = interval[0]
        else:
          begin = max(gold_unit[i-1][1], interval[0])
        
        if i == len(gold_unit) - 1:
          end = interval[1]
        else:
          end = min(gold_unit[i+1][0], interval[1])
        pred_units[sent_id][interval] = pred_unit[begin:end] 

  n_gold_tokens = len(gold_tokens)
  n_pred_tokens = len(pred_tokens)

  pred_stoi = {p:i for i, p in enumerate(sorted(pred_tokens, key=lambda x:int(x)))}
  gold_stoi = {g:i for i, g in enumerate(sorted(gold_tokens))}
  confusion = np.zeros((n_gold_tokens, n_pred_tokens))
  for sent_id in pred_units:
    for interval in pred_units[sent_id]:
      gold_unit = gold_units[sent_id][interval]
      g_idx = gold_stoi[gold_unit]
      for pred_unit in pred_units[sent_id][interval]:
        p_idx = pred_stoi[pred_unit]
        confusion[g_idx, p_idx] += 1
        
  n = confusion.sum()
  token_recall = confusion.max(1).sum() / n
  token_precision = confusion.max(0).sum() / n
  token_f1 = 2 * token_recall * token_precision /\
               (token_recall + token_precision)\
               if (token_recall + token_precision) > 0 else 0 
  
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  
  with open(os.path.join(out_path, 'token_f1'), 'w') as f:
    info = f'# units: {len(pred_stoi)}\n' \
           f'recall (%), precision (%), f1 (%)\n' \
           f'{token_recall*100:.2f}, {token_precision*100:.2f}, {token_f1*100:.2f}'
    f.write(info)
    print(info)
  return token_precision, token_recall, token_f1

def compute_boundary_f1(preds, golds, tol=0.02):
  n_p = 0
  n_g = 0
  correct_p = 0
  correct_g = 0
  for p, g in zip(preds, golds):
    n_p += len(p)
    n_g += len(g)

    p = np.asarray([round(p_t, 2) for p_t in p])
    g = np.asarray([round(g_t, 2) for g_t in g])
    if len(p):
      for g_t in g:
        min_dist = np.abs(g_t - p).min()
        correct_g += (min_dist <= tol)
    
    if len(g):
      for p_t in p:
        min_dist = np.abs(p_t - g).min()
        correct_p += (min_dist <= tol)
 
  precision = float(correct_p) / n_p if n_p > 0 else 0
  recall = float(correct_g) / n_g
  f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) != 0 else 0
  return precision, recall, f1

def main(argv):
  args = parse_args()
  
  if args.TASK == 0:
    if args.config is not None:
        config = json.load(open(args.config)) 
        gold_path = config['data_path']
        pred_path = os.path.join(config['ckpt_dir'], 'quantized_outputs.txt')
        out_path = os.path.join(config['ckpt_dir'], 'confusion.png')
    else:
        gold_path = argv[1]
        pred_path = argv[2]
        out_path = argv[3]
    compute_token_f1(pred_path, gold_path, out_path)
  elif args.TASK == 1:
    gold_path = argv[1]
    pred_path = argv[2]
    out_path = argv[3]
    compute_token_f1_beer(pred_path, gold_path, out_path)
  elif args.TASK == 2:
    gold_path = 'unit_tests/'
    if not os.path.exists(gold_path):
        os.makedirs(gold_path)
    gold_file = os.path.join(gold_path, 'test_token_f1_gold.item')
    pred_file = os.path.join(gold_path, 'test_token_f1_pred.txt')
    gold_f = open(gold_file, 'w')
    pred_f = open(pred_file, 'w')

    gold_f.write('header\n')
    gold_f.write('file_01 0.0 0.01 1 # # 0\n')
    gold_f.write('file_01 0.01 0.02 2 # # 0\n')
    gold_f.write('file_01 0.02 0.03 3 # # 0\n')
    gold_f.write('file_01 0.03 0.04 2 # # 0\n')
    pred_f.write('file_01 3,1,2,1\n')
    gold_f.close()
    pred_f.close()
    
    out_file = os.path.join(gold_path, 'confusion')
    compute_token_f1(pred_file,
                     gold_path,
                     out_file)
    

if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
