# VoxCentum

## Data Structure
```bash=
VoxCentum
├── data
│   ├── code2label.json
│   ├── iso_639-1.json
│   └── ...
├── LID
│   ├── modules
│   |   ├── log_results.py
│   |   └── ...
│   ├── run_lid.py
│   ├── run_lid.sh
│   └── ...
├── scripts
│   ├── cal_family_json.py
│   ├── cal_num.py
│   └── ...
└── VAD
    ├── modules
    |   ├── utils.py
    |   └── ...
    ├── run_vad.py
    └── ...

```

## Preparation

```shell
apt-get install libsox-fmt-all libsox-dev sox
add-apt-repository -y ppa:savoury1/ffmpeg4
apt-get -qq install -y ffmpeg
```

## Usage

### Running Language Identification (LID) with Multi-Thread

Jush execute `LID/run.sh` with correct settings.

1. Predict LID for a multi-lingual dataset:

```python
python run_lid.py \
    -s path/to/dataset \         # path to the folder of the output of VAD.
    -v path/to/vad/output \      # path to the root of the audio data.
    --lid_voxlingua_enable \
    --lid_whisper_enable \
    --lid_silero_enable
```


The structure of the dataset should be

```bash=
dataset
├── en
│   ├── channel_1
│   │   ├── audio-1.ogg
│   │   ├── audio-2.ogg
│   │   └── ...
│   └── channel_2
│       ├── audio-1.ogg
│       ├── audio-2.ogg
│       └── ...
├── af
│   ├── channel_1
│   │   ├── audio-1.ogg
│   │   ├── audio-2.ogg
│   └── ...
...
```

### Running Voice Activity Detection (VAD) with Multi-Process

```bash
python run_vad.py \
    -v path/to/vad_result \         # path to the folder of the output of VAD.
    -a path/to/audio_data_root \    # path to the root of the audio data.
    -f audio format                 # default: .wav
```

The structure of the input audio should be:

```bash=
vox100 (audio data root)
├── ar
│   ├── channel_1
│   │   ├── audio-1.wav
│   │   ├── audio-2.wav
│   │   └── ...
│   └── channel_2
│       ├── audio-1.wav
│       ├── audio-2.wav
│       └── ...
├── bn
│   ├── channel_1
│   │   ├── audio-1.wav
│   │   ├── audio-2.wav
│   └── ...
...
```

or providing the list of the audio path you want to process.

```bash
python run_vad.py \
    -v path/to/vad_result \         # path to the folder of the output of VAD.
    -a path/to/audio_data_root \    # path to the root of the audio data.
    -f audio format                 # default: .wav
    -d path/to/[lang_code]-filtered-list.txt
```

The list should be named as "[lang_code]-filtered-list.txt"
The structure of the list should be:

```bash=
[lang-code]-filtered-list.txt
├── path/to/audio-1.wav
├── path/to/audio-2.wav
├── path/to/audio-2.wav
├── path/to/audio-2.wav
│ 
...
```

The output of the VAD result would be named as "vad_time_stamp_[lang_code].json".
The structure of the .json file should be:

```bash=
vad_time_stamp_[lang_code].json
├── path/to/audio-1.wav
│   ├── timestamp1
│   │   ├── {"start": start1, "end": end1}
│   │   ├── {"start": start2, "end": end2}
│   │   └── ...
│   └── timestamp2
│       ├── {"start": start1, "end": end1}
│       ├── {"start": start2, "end": end2}
│       └── ...
├── path/to/audio-1.wav
│   ├── timestamp1
│   │   ├── {"start": start1, "end": end1}
│   │   ├── {"start": start2, "end": end2}
│   └── ...
...
```

### Preprocessd Audio Data

```bash=
vox100
├── en
│   ├── 30sec
│   │   ├── audio-1.wav
│   │   ├── audio-2.wav
│   │   └── ...
│   ├── 10sec
│   │   ├── audio-1.wav
│   │   ├── audio-2.wav
│   │   └── ...
│   └── 3sec
│       ├── audio-1.wav
│       ├── audio-2.wav
│       └── ...
├── zh
│   ├── 30sec
│   │   ├── audio-1.wav
│   │   ├── audio-2.wav
│   │   └── ...
│   ├── 10sec
│   │   ├── audio-1.wav
│   │   ├── audio-2.wav
│   │   └── ...
│   └── 3sec
│       ├── audio-1.wav
│       ├── audio-2.wav
│       └── ...
...
```

```bash=
vox100
├── am
│   ├── 30sec
│   │   ├── audio-1.wav
│   │   ├── audio-2.wav
│   │   └── ...
│   ├── 10sec
│   │   ├── audio-1.wav
│   │   ├── audio-2.wav
│   │   └── ...
│   └── 3sec
│       ├── audio-1.wav
│       ├── audio-2.wav
│       └── ...
├── as
│   ├── 30sec
│   │   ├── audio-1.wav
│   │   ├── audio-2.wav
│   │   └── ...
│   ├── 10sec
│   │   ├── audio-1.wav
│   │   ├── audio-2.wav
│   │   └── ...
│   └── 3sec
│       ├── audio-1.wav
│       ├── audio-2.wav
│       └── ...
...
```

### Utils

(1) utils/stat/cal_vad_time.py  
Calculating and recording the raw duration and post-VAD duration for various channels in different languages.  

(2) utils/change_1channel.py  
Converting audio files to mono channel and a sample rate of 16k.  

(3) utils/check_lang_channel.py  
Calculating the original duration for each channel.  

(4) utils/data_filter_manual.py  
Making filtered data list for each language, each language has about 300 hours.  

(5) utils/mp_construct_metadata.py  
Creating data statistics for each channel in different languages across all videos.  

(6) utils/split_audio.py  
Segmenting each audio based on the filtered data list after VAD into 10-second segments.  

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
