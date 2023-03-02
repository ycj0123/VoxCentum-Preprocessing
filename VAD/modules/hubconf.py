dependencies = ['torch', 'torchaudio']
import torch
import json
from utils_vad import (init_jit_model,
                       get_speech_timestamps,
                       get_number_ts,
                       get_language,
                       get_language_and_group,
                       save_audio,
                       read_audio,
                       VADIterator,
                       collect_chunks,
                       drop_chunks,
                       Validator,
                       OnnxWrapper)

def versiontuple(v):
    return tuple(map(int, (v.split('+')[0].split("."))))


def silero_vad(onnx=False, force_onnx_cpu=False):
    """Silero Voice Activity Detector
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """

    if not onnx:
        installed_version = torch.__version__
        supported_version = '1.12.0'
        if versiontuple(installed_version) < versiontuple(supported_version):
            raise Exception(f'Please install torch {supported_version} or greater ({installed_version} installed)')

    hub_dir = torch.hub.get_dir()
    # /snakers4_silero-vad_master/files/silero_vad.onnx
    print(f'{hub_dir}/snakers4_silero-vad_master/files/silero_vad.onnx')
    if onnx:
        silero_vad_path = "/home/meta-531-216/kuanyi_stage1/silero-vad/files/silero_vad.onnx"
        model = OnnxWrapper(silero_vad_path, force_onnx_cpu)
    else:
        model = init_jit_model(model_path=f'{hub_dir}/snakers4_silero-vad_master/files/silero_vad.jit')

    utils = (get_speech_timestamps,
             save_audio,
             read_audio,
             VADIterator,
             collect_chunks)

    return model, utils


