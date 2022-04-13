import os
import json
from pyhocon import ConfigFactory
import argparse
from copy import deepcopy

PHN = 'PHN'
def main(segment_path, 
         sent_info_path,
         out_prefix,
         debug=False,
         key='predicted_segments'):
  boundary_dict = dict() 
  for segment_file in os.listdir(segment_path):
    audio_id = segment_file.split(".")[0]
    with open(os.path.join(segment_path, segment_file), "r") as f:
      boundary_str = f.read().strip().split("\n")[1]
      boundary_dict[audio_id] = [float(b) for b in boundary_str.split(':')[-1].lstrip().split()]

  with open(sent_info_path, "r") as in_f,\
       open(out_prefix+'_with_predicted_segments.json', "w") as out_f:
    n_line = 0
    for line in in_f:
      if debug and n_line > 3:
        break
      sent_dict = json.loads(line.rstrip("\n"))
      if "audio_id" in sent_dict:
        audio_id = sent_dict["audio_id"]
      else:
        audio_id = sent_dict["utterance_id"]
    
      if "word_id" in sent_dict:
        audio_id = f'{audio_id}_{sent_dict["word_id"]}'
      
      if not audio_id in boundary_dict:
        print(f'{audio_id} not found')
        continue
      
      n_line += 1
      pred_boundary = boundary_dict[audio_id]
      pred_boundary = sorted(set(pred_boundary))
      pred_segments = [{"text": PHN, 
                        "begin": begin,
                        "end": end} for begin, end in zip(pred_boundary[:-1], pred_boundary[1:])]
      #print("Gold segments: ", gold_segments)
      #print("Pred segments: ", pred_segments)
      #print("\n")
      sent_dict[key] = deepcopy(pred_segments)
      out_f.write(json.dumps(sent_dict)+"\n")
  print('Number of audios: ', n_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf")
    args = parser.parse_args()
    config = ConfigFactory.parse_file(args.conf)
    main(config.out_path,
         config.metadata_path,
         config.metadata_path.rstrip('.json'),
         debug=config.debug)
