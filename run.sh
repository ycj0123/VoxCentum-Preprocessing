export CUDA_VISIBLE_DEVICES=$1
CHANNEL_NAME=$2
VAD_OUTPUT=$3

echo "now predicting "$CHANNEL_NAME
python run_lid.py \
    -s $CHANNEL_NAME \
    -w 4 \
    --batch_size 8 \
    --max_trial 3 \
    --chunk_sec 10 \
    -v \
    --vad_output $VAD_OUTPUT