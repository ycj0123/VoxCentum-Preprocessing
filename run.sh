export CHANNEL_FOLDER=$1
export VAD_OUTPUT=$2

echo "now predicting "$CHANNEL_FOLDER
python run_lid.py \
    -s $CHANNEL_FOLDER \
    -w 4 \
    --batch_size 8 \
    --max_trial 3 \
    --chunk_sec 10 \
    -v \
    --vad_path $VAD_OUTPUT