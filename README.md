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

1. Predict LID for a single channel folder.

Set 
```bash
# in run.sh
source_folder=path/to/channel_folder
multi_lang=false
ground_truth=lid
```

The structure of the dataset should be

```bash=
channel_folder
├── audio-1.ogg
├── audio-2.ogg
└── ...
```

2. Predict LID for a multi-lingual dataset:

Set 
```bash
# in run.sh
source_folder=path/to/dataset
multi_lang=true
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
    -v path/to/vad_result \         # path to the folder of VAD result
    -a path/to/audio_data_root \    # path to the root of the audio data
    -d path/to/data_list \          # path to the list of audio data we used
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
