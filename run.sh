#!/bin/bash/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate zerospeech2021_baseline

gpu_num=3
stage=4
stop_stage=4

iq_root=/ws/ifp-53_2/hasegawa/lwang114/spring2022/InformationQuantizer
data_root=${iq_root}/resources
timit_root=/ws/ifp-53_2/hasegawa/lwang114/data/TIMIT
eval_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/beer/recipes/aud
cpc_root=${iq_root}/zerospeech2021_baseline
unsup_seg_root=${iq_root}/UnsupSeg
model_root=${iq_root}/checkpoints/debug ###-------> Change this

word_dataset=librispeech_word
k=10 ###-------> Change this to between 30-70 for the actual dataset

# Extract CPC features for the spoken word dataset extracted from LibriSpeech
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  cwd=$(pwd)
  cd ${cpc_root}
  for split in train-clean-100 dev-clean; do
    CUDA_VISIBLE_DEVICES=0 python scripts/build_CPC_features.py \
        ${cpc_root}/checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt \
        ${data_root}/${word_dataset}/${split} \
        ${data_root}/${word_dataset}_cpc_txt
  done
  cd $cwd
fi

# VAD on the spoken word dataset
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  for split in train-clean-100 dev-clean; do
    python vad.py --out_extension .wav \
                  --audio_path resources/librispeech_word/${split}  
  done
fi

# Extract predicted segmentation for the spoken word dataset
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  # Extract unsupervised phone-level segmentations
  cd ${unsup_seg_root}
  conda activate unsup_seg
  for seg_conf in conf/librispeech_word_boundary_detection.json; do 
    CUDA_VISIBLE_DEVICES=$gpu_num python predict_whole_dataset.py --config $seg_conf
  done
  conda deactivate

  # Include predicted boundaries as part of the meta data
  cd $iq_root
  for seg_conf in configs/librispeech_word.conf; do
    python utils/extract_pred_boundaries.py 9 --config $seg_conf 
  done
  cd $cwd
fi

# Train IQ
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  config_file=$iq_root/configs/librispeech_word.conf
  CUDA_VISIBLE_DEVICES=$gpu_num python solver.py $config_file --setting ${k}clusters
fi

# Evaluate in-domain phoneme discovery performance of IQ with gold segmentation
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  ref_ali=$data_root/${word_dataset}/test.ali
  in_dirs=""
  for seed in 1234; do      
      hyp_ali=$model_root/pred_test_$seed.ali
      python utils/convert_item_to_beer_format.py \
        --item_file $data_root/${word_dataset}/dev-clean/dev-clean_nonoverlap.item \
        --gold_beer_file $ref_ali \
        --pred_file $model_root/outputs_quantized_$seed.txt
      python utils/convert_output_to_beer_format.py \
        --output_file $model_root/outputs_quantized_$seed.txt \
        --beer_file $hyp_ali 
  done

  for seed in 1234; do
      score_root=$model_root/score_aud_$seed
      if [ -d $score_root ]; then
        rm -r $score_root
      fi

      hyp_ali=$model_root/pred_test_$seed.ali
      
      cwd=$(pwd)
      cd $eval_root
      conda activate beer
      bash $eval_root/steps/score_aud.sh $ref_ali $hyp_ali $model_root/score_aud_$seed
      conda deactivate
      cd $cwd

      # Token F1 score
      python ${iq_root}/utils/evaluate.py 1 $ref_ali $hyp_ali $score_root
      in_dirs="${score_root},${in_dirs}"
  done
  in_dirs=${in_dirs%%,}

  # Compute average and standard deviation
  python ${iq_root}/utils/average_performance.py --in_dirs ${in_dirs} --out_path $model_root/average_performance
fi

# Evaluate performance on TIMIT for models trained on LibriSpeech+TIMIT with gold segmentation
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  seed=1234
  ref_ali=${eval_root}/local/timit/timit.ali
  
  for vocab in 500word; do # 800word visual_word 
    cwd=$(pwd)
    cd ${iq_root}
    config_file=${iq_root}/configs/timit_librispeech_${vocab}_segment_cpc_info_quantizer.json
    model_root=""
    while read -r l; do
      if [[ $l =~ $re ]]; then
        value="${BASH_REMATCH[2]}"
        model_root=$value
      fi
    done < $config_file
    echo "Evaluating model from ${model_root}"

    in_dirs=""
    for seed in 1234 2341 3412; do
      hyp_ali=${model_root}/pred_timit_${seed}.ali
      python ${iq_root}/utils/utils.py 3 $model_root/TIMIT_outputs_quantized_$seed.txt $hyp_ali 

      score_root=$model_root/score_aud_timit_$seed
      if [ -d $score_root ]; then
        rm -r $score_root
      fi
      conda activate beer
      cd ${eval_root}
      bash ${eval_root}/steps/score_aud.sh $ref_ali $hyp_ali $model_root/score_aud_timit_$seed
      conda deactivate
      cd $cwd
      
      # Token F1 score
      python ${iq_root}/utils/evaluate.py 1 $ref_ali $hyp_ali $score_root

      in_dirs="${score_root},${in_dirs}"
    done    
    in_dirs=${in_dirs%%,}

    # Compute average and standard deviation
    python ${iq_root}/utils/average_performance.py --in_dirs ${in_dirs} --out_path $model_root/average_performance
  done
fi
