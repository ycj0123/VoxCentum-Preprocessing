# audio-preprocessing-pipeline

## Preparation

```shell
apt-get install libsox-fmt-all libsox-dev sox
add-apt-repository -y ppa:savoury1/ffmpeg4
apt-get -qq install -y ffmpeg
```

## Usage

### Running Language Identification (LID) with Multi-Thread

Run `multi_run.sh` will predict LID for all 70 channels of 14 languages whose output under `./output/vox100_LID/` does not exist.

`sh multi_run.sh`

Run `run.sh` with path to channel_folder and vad_time_stamp_LID_CHANNEL.txt will predict LID for the given channel. e.g.

```bash
sh run.sh \
    ../cmd_download/vox100_ar/aljazeera \
    ../stage1_vad_timestamps/vad_time_stamp_ar_aljazeera.txt
```

Run `run_lid.py` will predict LID for a channel.

```bash
python run_lid.py \
    -s path/to/vox100_lid/channel \  # path to channel folder containing *.ogg
    -w 4 \                          # number of workers
    --batch_size 8 \                # batch size
    --max_trial 3 \                 # number of random samples
    --chunk_sec 10 \                # size of a random sample
    -v \                            # Use vad
    --vad_path path/to/vad_time_stamp_LID_CHANNEL.txt  # path to vad time stamp dictionary
```

### Convert audio format to ogg and sampling to 16k

`python convert_format_sampling.py -s /audio_folder/ -w 30`

### Language Identification (LID) and speech enhancement

```python
from lid_enhancement import AudioLIDEnhancer

ase = AudioLIDEnhancer(enable_enhancement=False)
print(ase('test.ogg'))
```

## References

Denoiser copied
from [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_synthesis/preprocessing/denoiser)
