import json
import nlp2
import random
from tqdm import tqdm
import os

from multiprocessing import Pool
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import argparse

''' Creating data statistics for each channel in different languages across all videos. '''

def convert_time(second):
    minute = int(second / 60)
    second = second - minute * 60
    hour = int(minute / 60)
    minute = minute - hour * 60
    time_string = f"{hour} hour " + f"{minute} minute " + f"{second} second."
    return time_string

def get_meta_json_file(dir_path, shuffle=False):
    result_jsons = []
    for i in tqdm(nlp2.get_files_from_dir(dir_path, match='.json')): 
        try:
            result_jsons.append(i)
        except:
            pass
    
    if shuffle == True:
        random.shuffle(result_jsons)
    
    return result_jsons

def get_audio_duration(data_path):

    duration = 0
    try:
        with open(data_path, "r") as f:
            duration = json.load(f)['duration']
    except:
        pass
   
    return duration

def running_construct_dict(data_path):
    duration = get_audio_duration(data_path)
    dir_name = data_path.split("/")[-2]
    audio_name = data_path.split("/")[-1]
    key = language_code + "/" + dir_name + "/" + audio_name

    return [key, duration]

def construct_dict(result_jsons, language_code, process_num):
    ''' construct dict -> key: audio file, value: duration (sec) '''

    audio_duration_dict = dict()
    # run multi-processing vad.
    pool = mp.Pool(process_num)
    results = []
    start = time.time()
    for result in tqdm(pool.imap_unordered(running_construct_dict, result_jsons),total=len(result_jsons)):
        results.append(result)
    end = time.time()
    print(f"Construct Metadata Time = {end - start}")

    # construct dict.
    for info in results:
        audio_name = info[0]
        duration = info[1]
        audio_duration_dict[audio_name] = duration

    return audio_duration_dict

def cli_main(dir_path, language_code, output_file_name, process_num):
    result_jsons = get_meta_json_file(dir_path, shuffle=True)
    audio_duration_dict = construct_dict(result_jsons, language_code, process_num)
    # print(audio_duration_dict)
    output_file_name = output_file_name + ".json"
    with open(output_file_name, 'w') as fp:
        json.dump(audio_duration_dict, fp, indent=4)
    
if __name__ ==  '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", help="nas_root", type=str, default="/home/meta-531-216/mount-4")
    parser.add_argument("-s", "--stage", help="which_stage", type=str, default="1")
    parser.add_argument("-p", "--process", help="multi_process_num", type=int, default=4)
    
    args = parser.parse_args()

    stage_index = args.stage

    nas_root_path = args.root

    process_num = args.process

    root_path = f"{nas_root_path}/stage{stage_index}/vox100" # path of nas root directory.
    stage_dir = f"./stage{stage_index}_metadata" # output path of metadata directory.
    stage_dir = f"/home/meta-531-216/kuanyi_stage1/stage{stage_index}_metadata"

    for lang_code in os.listdir(root_path):

        dir_path = f"{root_path}/{lang_code}"
        global language_code
        language_code = lang_code

        for channel_name in os.listdir(dir_path):
            channel_path = dir_path + "/" + channel_name
            print(f"Processing Lang={lang_code}. Channel={channel_name}")
            output_file_name = f"{stage_dir}/{lang_code}-{channel_name}-metadata"
            cli_main(channel_path, lang_code, output_file_name, process_num)