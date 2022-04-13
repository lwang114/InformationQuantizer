import torch
from torch.utils.data import Dataset
import torchaudio
import os
import math
import json
from copy import deepcopy

SIL = 'SIL'
def collate_fn_padd(batch):
    """collate_fn_padd
    Padds batch of variable length

    :param batch:
    """
    spects = [t[0] for t in batch]
    segments = [t[1] for t in batch]
    phonemes = [t[2] for t in batch]
    lengths = [t[3] for t in batch]
    fnames = [t[4] for t in batch]
  
    padded_spects = torch.nn.utils.rnn.pad_sequence(spects, batch_first=True)
    lengths = torch.LongTensor(lengths)

    return padded_spects, segments, phonemes, lengths, fnames

def spectral_size(wav_len):
    layers = [(10,5,0), (8,4,0), (4,2,0), (4,2,0), (4,2,0)]
    for kernel, stride, padding in layers:
        wav_len = math.floor((wav_len + 2*padding - 1*(kernel-1) - 1)/stride + 1)
    return wav_len

class SpeechSet(Dataset):
    def __init__(self, data_path, 
                 split, 
                 splits,
                 metadata_path, 
                 debug=False):
        self.data_path = data_path
        self.splits = splits
        data = []
        if isinstance(metadata_path, str):
          metadata_path = [metadata_path for _ in self.splits[split]]

        for sp, path in zip(self.splits[split], metadata_path):
          examples = load_data_split(data_path, sp, path, debug=debug)
          data.extend(examples)
          print(f'Number of {sp} audio files = {len(examples)}')
        self.data = data

    def __getitem__(self, idx):
        wav_path, segments, phonemes = self.data[idx]
        audio, sr = torchaudio.load(wav_path)
        audio = audio[0]
        audio_len = len(audio)
        spectral_len = spectral_size(audio_len)
        len_ratio = audio_len / spectral_len

        segments = [int(t*sr / len_ratio) for t in segments]
        phonemes = [[begin*sr / len_ratio,
                     end*sr / len_ratio] for begin, end in phonemes]
        return audio, segments, phonemes, spectral_len, wav_path

    def __len__(self):
        return len(self.data)


class TrainTestSpeechSet(SpeechSet):
    def __init__(self, data_path,
                 split,
                 splits,
                 metadata_path,
                 debug=False):
        super(TrainTestSpeechSet, self).__init__(data_path, split, splits, metadata_path, debug)
        
    @staticmethod
    def get_datasets(data_path, 
                     splits, 
                     metadata_paths, 
                     val_ratio=0.1,
                     debug=False):
        train_dataset = TrainTestSpeechSet(data_path,
                                           'train', 
                                           splits, 
                                           metadata_paths[0],
                                           debug=debug)
        test_dataset = TrainTestSpeechSet(data_path,
                                          'test',
                                          splits,
                                          metadata_paths[1],
                                          debug=debug)
        train_len = len(train_dataset)
        train_split = int(train_len * (1 - val_ratio))
        val_split = train_len - train_split
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_split, val_split])

        return train_dataset, val_dataset, test_dataset


def load_data_split(data_path, sp, metadata_path, debug=False):
    examples = []
    segments = dict()
    phonemes = dict()
    n_lines = 0
    with open(metadata_path, 'r') as f:
      for line in f:
        if debug and n_lines > 20:
          break
        n_lines += 1
        sent_dict = json.loads(line.rstrip('\n'))
        if 'audio_id' in sent_dict:
          audio_id = sent_dict['audio_id']
        else:
          audio_id = sent_dict['utterance_id']

        if 'word_id' in sent_dict:
          word_id = sent_dict['word_id']
          audio_id = f'{audio_id}_{word_id}'
          phns = sent_dict['phonemes']
          begin = phns[0]['begin']
          for phn_idx in range(len(phns)):
            phns[phn_idx]['begin'] = phns[phn_idx]['begin'] - begin
            phns[phn_idx]['end'] = phns[phn_idx]['end'] - begin
        else:
          phns = [phn for word in sent_dict['words'] for phn in word['phonemes']]
        
        segments[audio_id] = []
        phonemes[audio_id] = []

        for phn_idx, phn in enumerate(phns):
          if phn_idx == 0 and phn['begin'] != 0:
            segments[audio_id].append(phn['begin'])
          segments[audio_id].append(phn['end'])
          # XXX if phn['text'] != SIL:
          phonemes[audio_id].append([phn['begin'], phn['end']])

    for wav_file in os.listdir(os.path.join(data_path, sp)):
      if wav_file.split('.')[-1] != 'wav':
        continue
      audio_id = wav_file.split('.')[0]
      if not audio_id in segments:
        print(f'{audio_id} not found')
        continue
      segs = deepcopy(segments[audio_id])
      phns = deepcopy(phonemes[audio_id])
      examples.append([os.path.join(data_path, sp, wav_file), segs, phns])
    return examples
