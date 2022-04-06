import torch
import torchaudio
import torchvision
from torchvision import transforms
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.structured_prediction
import numpy as np
import re
import os
import json
from tqdm import tqdm
from itertools import combinations
from copy import deepcopy
from PIL import Image
from scipy import signal
from kaldiio import ReadHelper
# dep_parser = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
# dep_parser._model = dep_parser._model.cuda()
# lemmatizer = WordNetLemmatizer()
UNK = "###UNK###"
NULL = "###NULL###"
BLANK = "###BLANK###"
SIL = "SIL" 
IGNORED_TOKENS = ["GARBAGE", "+BREATH+", "+LAUGH+", "+NOISE+"]  

lemmatizer = WordNetLemmatizer() 

def log_normalize(x):
    x.add_(1e-6).log_()
    mean = x.mean()
    std = x.std()
    return x.sub_(mean).div_(std + 1e-6)

def fix_embedding_length(emb, L, padding=0):
  size = emb.size()[1:]
  if emb.size(0) < L:
    if padding == 0:
      pad = torch.zeros((L-emb.size(0),)+size, dtype=emb.dtype)
    else:
      pad = padding*torch.ones((L-emb.size(0),)+size, dtype=emb.dtype) 
    emb = torch.cat([emb, pad], dim=0)
  else:
    emb = emb[:L]
  return emb

def collate_fn_spoken_word(batch):
  audios = [t[0] for t in batch]
  phone_labels = [t[1] for t in batch]
  labels = [t[2] for t in batch]
  input_masks = [t[3] for t in batch]
  phone_masks = [t[4] for t in batch]
  word_masks = [t[5] for t in batch] 
  indices = [t[6] for t in batch]
 
  if isinstance(audios[0], list): 
    audios = [
        torch.nn.utils.rnn.pad_sequence(audio)\
        for audio in audios
    ]
    input_masks = [
        torch.nn.utils.rnn.pad_sequence(input_mask)
        for input_mask in input_masks
    ]
    audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True) # (bsz, n_seg, n_pos, d)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True) # (bsz, n_seg, n_pos)
    audios = audios.permute(0, 2, 1, 3)
    input_masks = input_masks.permute(0, 2, 1)
  else:
    audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True)
  phone_labels = torch.nn.utils.rnn.pad_sequence(phone_labels, batch_first=True)
  labels = torch.stack(labels)
  phone_masks = torch.nn.utils.rnn.pad_sequence(phone_masks, batch_first=True)
  return audios, phone_labels, labels, input_masks, phone_masks, indices

def embed(feat, method='average'):
  if method == 'average':
    return feat.mean(0)
  elif method == 'resample':
    new_feat = signal.resample(feat.detach().numpy(), 4)
    return torch.FloatTensor(new_feat.flatten())

class SpokenWordDataset(torch.utils.data.Dataset):
  def __init__(
      self, data_path, 
      preprocessor, split,
      splits = {
        "train": ["train-clean-100", "train-clean-360"],
        "validation": ["dev-clean"],
        "test": ["dev-clean"],     
      },
      augment=False,
      use_segment=False,
      audio_feature="cpc",
      phone_label="predicted",
      ds_method="average",
      sample_rate=16000,
      min_class_size=50,
      n_positives=0,
      debug=False
  ):
    self.preprocessor = preprocessor
    if debug:
      splits['train'] = [splits['train'][0]]
    self.splits = splits[split]

    self.data_path = data_path
    self.use_segment = use_segment
    self.ds_method = ds_method
    self.sample_rate = sample_rate
    self.n_positives = n_positives
    self.debug = debug
    
    data = []
    for sp in self.splits:
      # Load data paths to audio and visual features
      examples = load_data_split(preprocessor.dataset_name,
                                 data_path, sp,
                                 min_class_size=min_class_size,
                                 audio_feature=audio_feature,
                                 phone_label=phone_label,
                                 debug=debug)      
      data.extend(examples)
      print("Number of {} audio files = {}".format(split, len(examples)))

    # Set up transforms
    self.audio_transforms = [
        torchaudio.transforms.MelSpectrogram(
          sample_rate=sample_rate, win_length=sample_rate * 25 // 1000,
          n_mels=preprocessor.num_features,
          hop_length=sample_rate * 10 // 1000,
        ),
        torchvision.transforms.Lambda(log_normalize),
    ]
  
    if augment:
        augmentation = [
                torchaudio.transforms.FrequencyMasking(27, iid_masks=True),
                torchaudio.transforms.FrequencyMasking(27, iid_masks=True),
                torchaudio.transforms.TimeMasking(100, iid_masks=True),
                torchaudio.transforms.TimeMasking(100, iid_masks=True),
            ]
        self.audio_transforms.extend(augmentation)
    self.audio_transforms = torchvision.transforms.Compose(self.audio_transforms)
    audio = [example["audio"] for example in data]
    text = [example["text"] for example in data]
    phonemes = [example["phonemes"] for example in data]
    true_phonemes = [example["true_phonemes"] for example in data]
    self.dataset = [list(item) for item in zip(audio, text, phonemes, true_phonemes)]
    self.class_to_indices = defaultdict(list)
    for idx, item in enumerate(self.dataset):
      self.class_to_indices[item[1]].append(idx)

    self.audio_feature_type = audio_feature

  def load_audio(self, audio_file):
    if self.audio_feature_type in ["mfcc", "bnf+mfcc"]:
      audio, _ = torchaudio.load(audio_file)
      inputs = self.audio_transforms(audio)
      inputs = inputs.squeeze(0).t()
    elif self.audio_feature_type in ["cpc", "cpc_big"]:
      if audio_file.split('.')[-1] == "txt":
        audio = np.loadtxt(audio_file)
      else:
        with ReadHelper(f"ark: gunzip -c {audio_file} |") as ark_f:
          for k, audio in ark_f:
            continue
      inputs = torch.FloatTensor(audio)
    elif self.audio_feature_type in ["bnf", "bnf+cpc"]:
      if audio_file.split('.')[-1] == "txt":
        audio = np.loadtxt(audio_file)
      else:
        with ReadHelper(f"ark: gunzip -c {audio_file} |") as ark_f:
          for k, audio in ark_f:
            continue
      if self.audio_feature_type == "bnf+cpc":
        cpc_feat = np.loadtxt(audio_file.replace("bnf", "cpc"))
        feat_len = min(audio.shape[0], cpc_feat.shape[0])
        audio = np.concatenate([audio[:feat_len], cpc_feat[:feat_len]], axis=-1)
      inputs = torch.FloatTensor(audio)
    elif self.audio_feature_type in ['vq-wav2vec', 'wav2vec', 'wav2vec2']:
      audio, _ = torchaudio.load(audio_file)
      inputs = audio.squeeze(0)
    else: Exception(f"Audio feature type {self.audio_feature_type} not supported")

    input_mask = torch.ones(inputs.size(0))
    return inputs, input_mask

  def segment(self, feat, segments, 
              method="average"):
    """ 
      Args:
        feat : (num. of frames, feature dim.)
        segments : a list of dicts of phoneme boundaries
      
      Returns:
        sfeat : (max num. of segments, feature dim.)
        mask : (max num. of segments,)
    """
    feat = feat
    sfeats = []

    word_begin = segments[0]["begin"]
    for i, segment in enumerate(segments):
      if segment["text"] == SIL:
        continue
      phn = segment["text"]
      begin = int(round((segment["begin"]-word_begin)*100, 3))
      end = int(round((segment["end"]-word_begin)*100, 3))
      dur = max(end - begin, 1)
      if method == "sample":
        end = min(max(begin+1, end), feat.size(0)) 
        t = torch.randint(begin, end, (1,)).squeeze(0)
        segment_feat = feat[t]
      else:
        if begin >= feat.size(0):
          print(f'Warning: ({phn}, {begin}, {end}) begin idx {begin} >= feature size {feat.size(0)}')
          segment_feat = feat[-1]
        elif begin != end:
          segment_feat = embed(feat[begin:end], method=method)
        else:
          segment_feat = embed(feat[begin:end+1], method=method)
      if torch.any(torch.isnan(segment_feat)):
        print(f'Bad segment feature for feature of size {feat.size()}, begin {begin}, end {end}')
      sfeats.append(segment_feat)
    sfeat = torch.stack(sfeats)
    if method == "no-op":
      mask = torch.zeros(len(feat), len(sfeats))
    else:
      mask = torch.ones(len(sfeats))

    return sfeat, mask
    
  def unsegment(self, sfeat, segments):
    """
      Args:
        sfeat : (num. of segments, feature dim.)
        segments : a list of dicts of phoneme boundaries
      
      Returns:
        feat : (num. of frames, feature dim.) 
    """
    if sfeat.ndim == 1:
      sfeat = sfeat.unsqueeze(-1)
    word_begin = segments[0]['begin']
    dur = segments[-1]["end"] - segments[0]["begin"]
    nframes = int(round(dur * 100, 3))
    feat = torch.zeros((nframes, *sfeat.size()[1:]))
    for i, segment in enumerate(segments):
      if segment["text"] == SIL:
        continue
      begin = int(round((segment["begin"]-word_begin)*100, 3))
      end = int(round((segment["end"]-word_begin)*100, 3))
      if i >= sfeat.size(0):
        break
      if begin != end:
        feat[begin:end] = sfeat[i]
      else:
        feat[begin:end+1] = sfeat[i]
    return feat.squeeze(-1)

  def update_segment(self, idx, new_segments):
    self.dataset[idx][2] = None
    self.dataset[idx][2] = [{k:v for k,v in s.items()} for s in new_segments]

  def __getitem__(self, idx):
    audio_file, label, phoneme_dicts, _ = self.dataset[idx]
    audio_inputs, input_mask = self.load_audio(audio_file)
    if self.use_segment:
      audio_inputs, input_mask = self.segment(audio_inputs, 
                                              phoneme_dicts,
                                              method=self.ds_method)
    if self.n_positives > 0: # Draw positive examples
      audio_inputs = [audio_inputs]
      input_mask = [input_mask]
      pos_idxs = self.sample_positives(idx, label, self.n_positives)
      for pos_idx in pos_idxs:
        outputs = self.load_audio(self.dataset[pos_idx][0])
        audio_inputs.append(outputs[0].clone())
        input_mask.append(outputs[1].clone())

    phonemes = [phn_dict["text"] for phn_dict in phoneme_dicts]
    
    word_labels = self.preprocessor.to_word_index([label])
    phone_labels = self.preprocessor.to_index(phonemes)
    if self.use_segment:
      word_mask = torch.zeros(1, len(phoneme_dicts), len(phoneme_dicts))
    else:
      word_mask = torch.zeros(1, len(audio_inputs), len(audio_inputs))

    for t in range(len(phoneme_dicts)):
      word_mask[0, t, t] = 1. 
    phone_mask = torch.ones(len(phonemes))

    return audio_inputs,\
           phone_labels,\
           word_labels,\
           input_mask,\
           phone_mask,\
           word_mask,\
           idx

  def sample_positives(self, idx, label, n):
    random_idxs = np.random.permutation(self.class_to_indices[label])
    pos_idxs = []
    for i in random_idxs:
      if i == idx:
        continue
      pos_idxs.append(i)
      if len(pos_idxs) == n:
        break
    if not len(pos_idxs):
      print(f'No positive examples found for class {self.dataset[idx][1]}')
      return [idx]*n
        
    if len(pos_idxs) < n:
      return pos_idxs+[pos_idxs[-1]]*(n-len(pos_idxs))
    return pos_idxs

  def __len__(self):
    return len(self.dataset)


class SpokenWordPreprocessor:
  def __init__(
    self,
    dataset_name,
    data_path,
    num_features,
    splits = {
        "train": ["train-clean-100", "train-clean-360"],
        "validation": ["dev-clean"],
        "test": ["dev-clean"]
    },
    tokens_path=None,
    lexicon_path=None,
    use_words=False,
    prepend_wordsep=False,
    audio_feature="mfcc",
    phone_label="predicted",
    sample_rate=16000,
    min_class_size=50,
    ignore_index=-100,
    use_blank=True,
    debug=False,      
  ):
    self.dataset_name = dataset_name
    self.data_path = data_path
    self.num_features = num_features
    self.ignore_index = ignore_index
    self.min_class_size = min_class_size
    self.use_blank = use_blank
    self.wordsep = " "
    self._prepend_wordsep = prepend_wordsep
    if debug:
      splits['train'] = [splits['train'][0]]

    metadata_file = os.path.join(data_path, f"{dataset_name}.json")    
    data = []
    for split_type, spl in splits.items(): 
      if split_type == 'test_oos':
        continue
      for sp in spl:
        data.extend(load_data_split(dataset_name,
                                    data_path, sp,
                                    audio_feature=audio_feature,
                                    phone_label=phone_label,
                                    min_class_size=self.min_class_size,
                                    debug=debug)) 
    visual_words = set()
    tokens = set()
    for ex in data:
      visual_words.add(ex["text"])
      for phn in ex["phonemes"]:
        if phone_label == "groundtruth" and not "phoneme" in phn["text"]:
          phn["text"] = re.sub(r"[0-9]", "", phn["text"])
        tokens.add(phn["text"])

    self.tokens = sorted(tokens)
    self.visual_words = sorted(visual_words)
    if self.use_blank:
      self.tokens = [BLANK]+self.tokens
      self.visual_words = [BLANK]+self.visual_words
      
    self.tokens_to_index = {t:i for i, t in enumerate(self.tokens)}
    self.words_to_index = {t:i for i, t in enumerate(self.visual_words)}

    print(f"Preprocessor: number of phone classes: {self.num_tokens}")
    print(f"Preprocessor: number of visual word classes: {self.num_visual_words}")
    
  @property
  def num_tokens(self):
    return len(self.tokens)

  @property
  def num_visual_words(self):
    return len(self.visual_words)

  def to_index(self, sent):
    tok_to_idx = self.tokens_to_index
    return torch.LongTensor([tok_to_idx.get(t, 0) for t in sent])

  def to_word_index(self, sent):
    tok_to_idx = self.words_to_index
    return torch.LongTensor([tok_to_idx.get(t, 0) for t in sent])

  def to_text(self, indices):
    text = []
    for t, i in enumerate(indices):
      if (i == 0) and (t != 0):
        prev_token = text[t-1]
        text.append(prev_token)
      else:
        text.append(self.tokens[i])
    return text

  def to_word_text(self, indices):
    return [self.visual_words[i] for i in indices]

  def tokens_to_word_text(self, indices):
    T = len(indices)
    path = [self.visual_words[i] for i in indices]
    sent = []
    for i in range(T):
      if path[i] == BLANK:
        continue
      elif (i != 0) and (path[i] == path[i-1]):
        continue
      else:
        sent.append(path[i])
    return sent

  def tokens_to_text(self, indices): 
    T = len(indices)
    path = self.to_text(indices)
    sent = []
    for i in range(T):
      if path[i] == BLANK:
        continue
      elif (i != 0) and (path[i] == path[i-1]):
        continue 
      else:
        sent.append(path[i])
    return sent


def load_data_split(dataset_name,
                    data_path, split,
                    audio_feature="mfcc",
                    phone_label="predicted",
                    min_class_size=50,
                    max_keep_size=1000,
                    debug=False):
  """
  Returns:
      examples : a list of mappings of
          { "audio" : filename of audio,
            "text" : a list of tokenized words for the class name,
          }
  """
  examples = []
  if os.path.exists(os.path.join(data_path, f'{split}.json')):
    word_files = [f'{split}.json']
  else:
    word_files = [word_file for word_file in os.listdir(data_path) if word_file.split('.')[-1] == 'json']

  for word_file in word_files:
    word_f = open(os.path.join(data_path, word_file), "r")

    label_counts = dict()
    for line in word_f:
      if debug and len(examples) >= 100:
        break
      word = json.loads(line.rstrip("\n"))
      label = lemmatizer.lemmatize(word["label"].lower())
      if not label in label_counts:
        label_counts[label] = 1
      else:
        label_counts[label] += 1
      if label_counts[label] > max_keep_size:
        continue

      if word["split"] != split:
        continue
      
      audio_id = word["audio_id"]
      audio_path = None
      word_id = word['word_id']
      if audio_feature in ["mfcc", "vq-wav2vec", "wav2vec2", "wav2vec"]:
        audio_path = os.path.join(data_path, split, f"{audio_id}_{word_id}.wav")
        if not os.path.exists(audio_path):
          word_id = int(word_id)
          audio_file = f"{audio_id}_{word_id:04d}.wav"
          audio_path = os.path.join(data_path, split, audio_file)
      elif audio_feature in ["cpc", "cpc_big"]:
        audio_path = os.path.join(data_path, f"../{dataset_name}_{audio_feature}_txt/{audio_id}_{word_id}.txt")
        if not os.path.exists(audio_path):
          audio_path = os.path.join(data_path, f"../{dataset_name}_{audio_feature}/{audio_id}_{word_id}.ark.gz")
        if not os.path.exists(audio_path):
          word_id = int(word_id)
          audio_file = f"{audio_id}_{word_id:04d}.txt"
          audio_path = os.path.join(data_path, f"../{dataset_name}_{audio_feature}_txt", audio_file)
      elif audio_feature in ["bnf", "bnf+cpc"]:
        audio_file = f"{audio_id}_{word_id}.txt"
        audio_path = os.path.join(data_path, f"../{dataset_name}_bnf_txt", audio_file)
      else: Exception(f"Audio feature type {audio_feature} not supported")

      true_phonemes = word["phonemes"]
      if "children" in true_phonemes:
        true_phonemes = [phn for phn in true_phonemes["children"] if phn["text"] != SIL]
        if len(true_phonemes) == 0:
          continue
                 
      for phn_idx in range(len(true_phonemes)):
        if not "phoneme" in true_phonemes[phn_idx]["text"]:
          true_phonemes[phn_idx]["text"] = re.sub(r"[0-9]", "", true_phonemes[phn_idx]["text"]) 

      noisy = False
      for phn in true_phonemes:
        if phn["text"] in IGNORED_TOKENS or (phn["text"][0] == "+"):
          noisy = True
          break
      if noisy:
        continue

      dur = round(true_phonemes[-1]['end'] - true_phonemes[0]['begin'], 3)
      phonemes = None
      if phone_label == "groundtruth":
          phonemes = deepcopy(true_phonemes)
      elif phone_label == "multilingual":
          phonemes = [phn for phn in word["predicted_segments_multilingual"] if phn["text"] != SIL]
      elif phone_label == "multilingual_phones":
          phonemes = deepcopy(word["multilingual_phones"])
      elif phone_label == "predicted":
          phonemes = [phn for phn in word["predicted_segments"] if phn["text"] != SIL]
      elif phone_label == "predicted_wav2vec2":
          if not "predicted_segments_wav2vec2" in word:
              continue
          phonemes = [phn for phn in word["predicted_segments_wav2vec2"] if phn["text"] != SIL]
      else:
          raise ValueError(f"Invalid phone label type: {phone_label}")

      phonemes = [phn for phn in phonemes if round(phn['end'] - phonemes[0]['begin'], 3) <= dur]
      if not len(phonemes):
        print(f'Skip example without segments: {phonemes}')
        continue

      if phonemes[-1]['end'] - phonemes[0]['begin'] != dur:
        phonemes[-1]['end'] = phonemes[0]['begin'] + dur
      example = {"audio": audio_path,
                 "text": label,
                 "phonemes": phonemes,
                 "true_phonemes": true_phonemes} 
      examples.append(example)
    word_f.close()
  return examples

   
if __name__ == "__main__":
    preproc = SpokenWordPreprocessor(num_features=80, data_path="/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/")
