import argparse
import dill
from argparse import Namespace
import tqdm
import torch
import torchaudio
import os
import json
import numpy as np
from utils import (detect_peaks, max_min_norm, replicate_first_k_frames, PrecisionRecallMetric)
from next_frame_classifier import NextFrameClassifier
from torch.utils.data import DataLoader
from speech_set import SpeechSet
from dataloader import *

SIL = 'SIL'
def main(data_loader, ckpt, out_path, debug=False, vad=False):
    if not os.path.exists(out_path):
      os.makedirs(out_path)
    print('Output path: ', out_path)
    batch_size = data_loader.batch_size
    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    hp = Namespace(**dict(ckpt["hparams"]))
    # load weights and peak detection params
    model = NextFrameClassifier(hp)
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k,v in weights.items()}
    model.load_state_dict(weights)
    peak_detection_params = dill.loads(ckpt['peak_detection_params'])['cpc_1']
    pr = PrecisionRecallMetric() 
    
    progress = tqdm(ncols=80, total=len(data_loader))
    for b_idx, batch in enumerate(data_loader):
      if debug and b_idx > 2:
        break
      audio = batch[0]
      segments = batch[1]
      phonemes = batch[2]
      length = batch[3]
      fnames = batch[4]
      preds = model(audio)

      # run inference
      preds = preds[1][0]  # get scores of positive pairs
      preds = replicate_first_k_frames(preds, k=1, dim=1)  # padding
      preds = 1 - max_min_norm(preds)  # normalize scores (good for visualizations)

      # VAD masking
      if vad:
        vad_mask = batch[5]
        preds = preds * vad_mask

      pr.update(segments, preds, length)
      peaks = detect_peaks(x=preds,
                           lengths=length,
                           prominence=peak_detection_params["prominence"],
                           width=peak_detection_params["width"],
                           distance=peak_detection_params["distance"])  # run peak detection on scores

      sr = 16000
      for idx, peak in enumerate(peaks):
        audio_len = len(torchaudio.load(fnames[idx])[0][0])
        spectral_len = spectral_size(audio_len)
        len_ratio = audio_len / spectral_len
        phoneme_segment = [phonemes[idx][0][0]] + [phn[1] for phn in phonemes[idx]]

        peak_in_sec = np.round(peak * len_ratio / sr, 4) #(peak - spectral_size(0)) * 160 / sr
        gold_in_sec = np.round(np.asarray(phoneme_segment) * len_ratio / sr, 4)
        audio_id = os.path.basename(fnames[idx]).split('.')[0]
        with open(os.path.join(out_path, audio_id+'.txt'), 'w') as f:
          peak_str = ' '.join([str(p) for p in peak])
          peak_in_sec_str = ' '.join([str(p) for p in peak_in_sec])
          gold_str =  ' '.join([str(g) for g in segments[idx]])
          gold_in_sec_str = ' '.join([str(g) for g in gold_in_sec])
          f.write(f'Predicted: {peak_str}\n')
          f.write(f'Predicted in seconds: {peak_in_sec_str}\n')
          f.write(f'Gold: {gold_str}\n')
          f.write(f'Gold in seconds: {gold_in_sec_str}')
      progress.update(1)
    progress.close()        
    scores, best_params = pr.get_stats()
    info = f'Boundary Precision -- {scores[0]*100:.2f}\tRecall -- {scores[1]*100:.2f}\tF1 -- {scores[2]*100:.2f}\tRval -- {scores[3]*100:.2f}'
    print(info)
    with open(os.path.join(out_path, '../results.txt'), 'a') as f:
      f.write(info+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unsupervised segmentation inference script')
    parser.add_argument('--config', help='path to config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    ckpt = config['ckpt']
    out_path = config['out_path']
    for split in ['test']:
      dataset = SpeechSet(config['data_path'], 
                          split,
                          config['splits'], 
                          config.get('metadata_path', None),
                          debug=False) 
      #dataset = WavPhnDataset(os.path.join(config['data_path'], split), vad=config['vad']) 
      data_loader = DataLoader(dataset,
                               batch_size=config['batch_size'],
                               collate_fn=collate_fn_padd,
                               num_workers=8)
      main(data_loader, ckpt, out_path, debug=config['debug'], vad=config['vad'])
