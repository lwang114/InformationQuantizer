import torch, os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from datasets.flickr8k_word_image import FlickrWordImageDataset, FlickrWordImagePreprocessor
from datasets.librispeech import LibriSpeechDataset, LibriSpeechPreprocessor
from datasets.spoken_word_dataset import SpokenWordDataset, SpokenWordPreprocessor, collate_fn_spoken_word

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    collate_fn = None

    if 'FLICKR_WORD_IMAGE' == name :
        preprocessor = FlickrWordImagePreprocessor(dset_dir, 80,
                                                   audio_feature=args.audio_feature, 
                                                   image_feature=args.image_feature,
                                                   phone_label=args.phone_label,
                                                   min_class_size=args.min_class_size,
                                                   ignore_index=args.ignore_index,
                                                   use_blank=args.use_blank,
                                                   debug=args.debug)
        train_data = FlickrWordImageDataset(dset_dir,
                                            preprocessor, 
                                            'train',
                                            audio_feature=args.audio_feature,  
                                            image_feature=args.image_feature,
                                            phone_label=args.phone_label,
                                            min_class_size=args.min_class_size,
                                            use_segment=args.use_segment,
                                            ds_method=args.downsample_method,
                                            debug=args.debug)
        test_data = FlickrWordImageDataset(dset_dir,
                                           preprocessor, 
                                           'test',
                                           audio_feature=args.audio_feature,
                                           image_feature=args.image_feature,
                                           phone_label=args.phone_label,
                                           min_class_size=args.min_class_size,
                                           use_segment=args.use_segment,
                                           ds_method=args.downsample_method,
                                           debug=args.debug) 
    elif args.dataset in ['librispeech', 'mboshi', 'TIMIT']:     
        preprocessor = LibriSpeechPreprocessor(
                         dset_dir, 80,
                         splits=args.splits,
                         audio_feature=args.audio_feature,
                         phone_label=args.phone_label,
                         ignore_index=args.ignore_index,
                         debug=args.debug)
        train_data = LibriSpeechDataset(
                         dset_dir, 
                         preprocessor,
                         'train',
                         splits=args.splits, 
                         augment=True,
                         audio_feature=args.audio_feature,
                         phone_label=args.phone_label,
                         use_segment=args.use_segment,
                         debug=args.debug) 
        test_data = LibriSpeechDataset(
                         dset_dir, 
                         preprocessor,
                         'test',
                         splits=args.splits,
                         augment=True,
                         audio_feature=args.audio_feature,
                         phone_label=args.phone_label,
                         use_segment=args.use_segment,
                         debug=args.debug) 
    else:
        preprocessor = SpokenWordPreprocessor(
                         args.dataset, 
                         dset_dir, 80,
                         splits=args.splits,
                         audio_feature=args.audio_feature,
                         phone_label=args.phone_label,
                         ignore_index=args.ignore_index,
                         use_blank=args.use_blank,
                         debug=args.debug)
        train_data = SpokenWordDataset(
                         dset_dir,
                         preprocessor,
                         'train',
                         splits=args.splits,
                         use_segment=args.use_segment,
                         audio_feature=args.audio_feature,
                         phone_label=args.phone_label,
                         ds_method=args.downsample_method,
                         n_positives=args.get('n_positives', 0),
                         debug=args.debug)
        test_data = SpokenWordDataset(
                         dset_dir,
                         preprocessor,
                         'test',
                         splits=args.splits,
                         use_segment=args.use_segment,
                         audio_feature=args.audio_feature,
                         phone_label=args.phone_label,
                         ds_method=args.downsample_method,
                         n_positives=args.get('n_positives', 0),
                         debug=args.debug)
        collate_fn = collate_fn_spoken_word

    if collate_fn is not None:
      train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0,
                                drop_last=False,
                                collate_fn=collate_fn)

      test_loader = DataLoader(test_data,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=0,
                               drop_last=False,
                               collate_fn=collate_fn)
    else:
      train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                drop_last=False)

      test_loader = DataLoader(test_data,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=4,
                               drop_last=False)

    data_loader = dict()
    data_loader['train']=train_loader
    data_loader['test']=test_loader

    return data_loader


if __name__ == '__main__' :
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST', type=str)
    parser.add_argument('--dset_dir', default='datasets', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()

    data_loader = return_data(args)
    import ipdb; ipdb.set_trace()
