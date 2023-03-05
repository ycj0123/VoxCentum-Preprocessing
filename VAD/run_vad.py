import torch
import torchaudio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing import Pool

from modules.vad import vad
from modules.hubconf import silero_vad

import argparse
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score
import nlp2
from tqdm import tqdm
import json

# load silero vad model.
model, utils = silero_vad(onnx=True)

# get utils.
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# our audio format: .wav, sr=16k, audio bit rate=16, channel num=2.
sampling_rate = 16000

# run vad.
def running_vad(file_path):
    wav = read_audio(file_path)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    return [file_path, speech_timestamps]

def format_transform(result_jsons, root_path, format="wav"):
    ''' Transform Data path '''
    modified_list = []
    split_path = "/home/meta-531-216/mount-4/stage1/vox100/"
    for item in result_jsons:
        audio_name = root_path + "/" + item.split("/home/meta-531-216/mount-4/stage1/vox100/")[1] + "." + format
        # print(audio_name)
        modified_list.append(audio_name)
        
    return modified_list

if __name__ ==  '__main__':

    # vad result.
    vad_output_dir = "/home/meta-531-216/kuanyi_stage1/stage1_vad_result"

    # audio file root directory.
    audio_root_path = "/home/meta-531-216/mount-4/preprocessed_16000/wav_fileter-list/vox100"

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vad_output_path", type=str, default=vad_output_dir, help="vad_output_result_path")
    parser.add_argument("-a", "--audio_path", type=str, default=audio_root_path, help="audio_data_root_path")
    parser.add_argument("-d", "--data_list_path", type=str, default=None, help="data_list_path")
    parser.add_argument("-f", "--audio_format", type=str, default="wav", help="audio_format")

    # which audio data being used.
    # data_list_dir_path = "/home/meta-531-216/kuanyi_stage1/stage1_filtered_data_list"

    args = parser.parse_args()
    config = vars(args)
    vad_output_dir = config["vad_output_path"]
    audio_root_path = config["audio_path"]
    data_list_dir_path = config["data_list_path"]
    audio_format = config["audio_format"]

    if data_list_dir_path != None:
        # collect which audio data we are going to use.
        for data_list_lang_path in os.listdir(data_list_dir_path):
            lang_code = data_list_lang_path
            data_list_lang_path = data_list_dir_path + "/" + data_list_lang_path

            for data_list_path in os.listdir(data_list_lang_path):
                channel_name = data_list_path.split(f"{lang_code}-")[-1]
                channel_name = channel_name.split("-filtered-list.txt")[0]
                data_list_path = data_list_lang_path + "/" + data_list_path 
                
                result_jsons = []
                with open(data_list_path, "r") as f:
                    for line in f.readlines():
                        line = line.strip()
                        result_jsons.append(line)
        
                result_jsons = format_transform(result_jsons, audio_root_path) 
            
                print(f"Processing {lang_code}-{channel_name}...")
                output_dict = dict()
                
                # run multi-processing vad.
                pool = mp.Pool(8)
                results = []
                start = time.time()
                for result in tqdm(pool.imap_unordered(running_vad, result_jsons),total=len(result_jsons)):
                    results.append(result)
                end = time.time()
                print(f"VAD Time = {end - start}")

                # collect vad result, key: file_path, value: speech_timestamps.
                for pair in results:
                    dict_key = pair[0]
                    dict_value = pair[1]
                    output_dict[dict_key] = dict_value
        
                # output path: output_dir/vad_time_stamp_{lang_code}.json
                output_file_name = vad_output_dir + "/" + "vad_time_stamp_" + lang_code + "_" + channel_name + ".json"
                with open(output_file_name, "w") as fp:
                    json.dump(output_dict, fp, indent=4)
    
    else: # processing all audio data in audio path. (every language and evary channel).
        for lang_code in os.listdir(audio_root_path):
            # lang_code: which language.
            lang_channel_path = os.path.join(audio_root_path, lang_code)
            for channel in os.listdir(lang_channel_path):
                # channel: which channel.
                channel_path = os.path.join(lang_channel_path, channel)
                result_jsons = []

                # collect files in the channel.
                source_dir = channel_path
                for i in tqdm(nlp2.get_files_from_dir(source_dir, match=audio_format)):
                    try:
                        result_jsons.append(i)
                    except:
                        pass
                
                print(f"Processing {lang_code}-{channel}...")
                output_dict = dict()
                
                # run multi-processing vad.
                pool = mp.Pool(8)
                results = []
                start = time.time()
                for result in tqdm(pool.imap_unordered(running_vad, result_jsons),total=len(result_jsons)):
                    results.append(result)
                end = time.time()
                print(f"VAD Time = {end - start}")

                # collect vad result, key: file_path, value: speech_timestamps.
                for pair in results:
                    dict_key = pair[0]
                    dict_value = pair[1]
                    output_dict[dict_key] = dict_value
        
                # output path: output_dir/vad_time_stamp_{lang_code}.json
                output_file_name = vad_output_dir + "/" + "vad_time_stamp_" + lang_code + "_" + channel_name + ".json"
                with open(output_file_name, "w") as fp:
                    json.dump(output_dict, fp, indent=4)
                
