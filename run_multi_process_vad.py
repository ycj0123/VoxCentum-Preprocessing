import time
from multiprocessing import Pool
import torch
import torchaudio
from scripts.hubconf import silero_vad
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import argparse
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score
import nlp2
from tqdm import tqdm
import json

model, utils = silero_vad(onnx=True)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

sampling_rate = 16000

def running_vad(file_path):
    wav = read_audio(file_path)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    return [file_path, speech_timestamps]

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", type=str, default='en', help="Language")
    parser.add_argument("-c", "--channel", type=str, default='CNN', help="Channel Name")
    args = parser.parse_args()
    config = vars(args)
    language_type = config['language']
    channel_name = config['channel']

    data_path = "/home/meta-531-216/nas-60022/cmd_download/vox100_" + language_type + "/" + channel_name 

    result_jsons = []
    start = time.time() 

    for i in tqdm(nlp2.get_files_from_dir(data_path, match='ogg')): 
        try:
            result_jsons.append(i)
        except:
            pass
        
    end = time.time() 
    print("Checking audio files costs %f seconds" % (end - start))

    file_list_len = len(result_jsons)
    output_dict = dict()

    # handle vad using multi-process.
    pool = mp.Pool()
    results = []
    start = time.time()
    for result in tqdm.tqdm(pool.imap_unordered(running_vad, result_jsons), total=len(result_jsons)):
        results.append(result)
    end = time.time()
    print(f"VAD Time = {end - start}")

    # write output.
    for pair in result:
        dict_key = pair[0]
        dict_value = pair[1]
        output_dict[dict_key] = dict_value
    
    output_file_name = "vad_time_stamp_" + language_type + "_" + channel_name + ".txt"
    with open(output_file_name, 'w') as f:
        print(output_dict, file=f)
       
