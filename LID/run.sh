#!/usr/bin/bash

export PYTHONPATH=$(pwd)/..:$PYTHONPATH

# files args
source_folder=../../test_channels_dir/
multi_lang=true
output_dir=./output

# prediction args
batch_size=1
max_trial=5
chunk_sec=10
ground_truth=haw
vad=false
se=false

python3 -m run_lid \
    -s "${source_folder}" \
    --batch_size "${batch_size}" \
    --max_trial "${max_trial}" \
    --chunk_sec "${chunk_sec}" \
    --multi_lang "${multi_lang}" \
    --ground_truth "${ground_truth}" \
    --output_dir "$output_dir" \
    --vad "${vad}" \
    --se "${se}" \
    --audio_ext ogg \
    --lid_whisper_enable \
    --lid_voxlingua_enable \
    --lid_silero_enable
