import argparse


def main(item_files,
         beer_file,
         frame_rate=0.01,
         pred_file=None):
  used_utt_ids = None
  if pred_file:
      used_utt_ids = []
      with open(pred_file, 'r') as f:
          for line in f:
              if not line.strip().split()[0] in used_utt_ids:
                  used_utt_ids.append(line.strip().split()[0])

  if isinstance(item_files, str):
    item_files = [item_files]
 
  with open(beer_file, 'w') as out_f:
    n_files = 0
    for item_file in item_files:
      with open(item_file, 'r') as in_f:
        cur_utt_str = ['']
        begin = 0
        for idx, line in enumerate(in_f):
          if idx == 0:
              continue
          tokens = line.split()
          
          if used_utt_ids is not None:
              if not tokens[0] in used_utt_ids:
                  continue

          if tokens[0] != cur_utt_str[0]:
            if cur_utt_str[0]:
              out_f.write(' '.join(cur_utt_str)+'\n')
              n_files += 1
            cur_utt_str = [tokens[0]]
            begin = 0

          cur_begin = int(round(float(tokens[1]) / frame_rate, 3))
          cur_end = int(round(float(tokens[2]) / frame_rate, 3))
          for _ in range(begin, cur_begin):
            cur_utt_str.append(SIL)
          begin = cur_end

          dur = cur_end - cur_begin 
          cur_utt_str.extend([tokens[3]]*dur)
        out_f.write(' '.join(cur_utt_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--item_files')
    parser.add_argument('--gold_beer_file')
    parser.add_argument('--frame_rate', type=int, default=0.01)
    parser.add_argument('--pred_file')
    args = parser.parse_args()
    item_files = args.item_files.split(',')
    main(item_files, args.gold_beer_file, frame_rate=args.frame_rate, pred_file=args.pred_file)
