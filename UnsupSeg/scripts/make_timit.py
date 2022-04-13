import argparse
import os
import shutil
from tqdm import tqdm

def main(inpath, outpath, sph2wav):
    if not os.path.exists(inpath): 
        print('Error: input path does not exist!!')
        return -1 
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)

    for _f in tqdm(os.listdir(inpath)):
        parent_f = os.path.join(inpath, _f)
        if os.path.isdir(parent_f):
            for _ff in os.listdir(parent_f):
                parent_ff = os.path.join(parent_f, _ff)
                if os.path.isdir(parent_ff):
                    for ex in os.listdir(parent_ff):
                        if ex.endswith('.WAV'):
                            src_name = os.path.join(parent_ff, ex)
                            ex_new = ex.split('.')[0]+'.wav'
                            tgt_name = os.path.join(outpath, _f+'_'+_ff+'_'+ex_new)
                            #os.system(f'awk -v sph2wav={sph2wav} -v wav_dir={outpath} \'\{print sph2wav " -f wav " $1 " > " wav_dir "/" $2 ".wav')
                            os.system(f'{sph2wav} -f wav {src_name} {tgt_name}')
                            shutil.copy(src_name, tgt_name)

                        if ex.endswith('.PHN'): 
                            src_name = os.path.join(parent_ff, ex)
                            ex_new = ex.split('.')[0]+'.phn'
                            tgt_name = os.path.join(outpath, _f+'_'+_ff+'_'+ex_new)
                            shutil.copy(src_name, tgt_name)

parser = argparse.ArgumentParser(description='Make TIMIT dataset ready for unsupervised segmentation.')
parser.add_argument('--inpath', type=str, required=True, help='the path to the base timit dir.')
parser.add_argument('--outpath', type=str, required=True, help='the path to save the new format of the data.')
parser.add_argument('--sph2wav')

args = parser.parse_args()

main(args.inpath, args.outpath, args.sph2wav)

