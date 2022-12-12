# lang_ids=("id" "ur" "de")
lang_ids=("es")

export CUDA_VISIBLE_DEVICES=2

for lid in "${lang_ids[@]}"; do
    for channel in $(ls ../vox100_${lid}); do
        if [[ ! -f output/vox100_${lid}/${channel}_output.txt ]]; then
            
            echo "now predicting "vox100_${lid}/${channel}
            python run_lid.py \
                -s ../vox100_${lid}/${channel} \
                -w 4 \
                --batch_size 8 \
                --max_trial 3 \
                --chunk_sec 10
            
            if [[ $? == "2" ]]; then
                echo "Error: vox100_${lid}/${channel}" >> error_channels.txt
            fi
        fi
        if [[ ! -f output/vox100_${lid}/${channel}_vad_output.txt ]]; then
            
            echo "now predicting "vox100_${lid}/${channel}" with vad"
            python run_lid.py \
                -s ../vox100_${lid}/${channel} \
                -w 4 \
                --batch_size 8 \
                --max_trial 3 \
                --chunk_sec 10 \
                -v \
                --vad_output ../stage1_vad_timestamps/vad_time_stamp_${lid}_${channel}.txt
            
            if [[ $? == "2" ]]; then
                echo "Error: vox100_${lid}/${channel} (vad)" >> error_channels.txt
            fi
        fi
    done
done
