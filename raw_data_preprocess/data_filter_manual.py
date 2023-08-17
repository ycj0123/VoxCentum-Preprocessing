import json
import nlp2
import random
from tqdm import tqdm
import os
import pandas as pd

''' Make filtered data list for each language, each language has about 300 hours'''
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

def construct_dict(result_jsons, language_code="ar"):
    ''' construct dict -> key: audio file, value: duration (sec) '''
    audio_duration_dict = dict()
    for data_path in tqdm(result_jsons):
        duration = get_audio_duration(data_path)

        dir_name = data_path.split("/")[-2]
        audio_name = data_path.split("/")[-1]
        key = language_code + "/" + dir_name + "/" + audio_name

        audio_duration_dict[key] = duration

    return audio_duration_dict

def check_audio_exist(path):

    # .webm/.m4a/.mp4/.webm.part
    extension = ["webm", "m4a", "mp4", "webm.part"]
    audio_path = "/home/meta-531-216/mount-4/stage3/vox100/" + path.split(".info.json")[0]
    path_extension = [(audio_path + "." + ext) for ext in extension]
    for path_ext in path_extension:
        # print(path_ext)
        if os.path.isfile(path_ext) == True:
            return True, path_ext

    return False, "not/existed"

def audio_filter(audio_duration_dict, channel_name, lang_code, channel_list, time_list, limit_hour):
    limit_sec = limit_hour * 60 * 60 # hour to sec.
    time_accumulation = 0

    min_exceed = limit_sec
    min_exceed_audio = []

    item = audio_duration_dict.items()

    audio_list = []

    used_num = 0
    not_used_num = 0
    
    total_time = 0 # record total time of all audio.
    
    time_constrain = 4 * 60 * 60 # 2hr.
    
    if limit_hour == -1: # -1 represents for using all audio.
        for i in item:
            ''' i[0]: audio file, i[1]: duration '''
            total_time = total_time + i[1]

            check, audio_ext = check_audio_exist(i[0])
            if check == True:
                audio_list.append(audio_ext)
                time_accumulation = time_accumulation + i[1]
                used_num = used_num + 1
                    
            else:
                continue
        limit_hour = time_accumulation
        limit_sec = limit_hour

    else:
        for i in item:
            ''' i[0]: audio file, i[1]: duration '''
            total_time = total_time + i[1]

            check, audio_ext = check_audio_exist(i[0])
            if check == True:
                if i[1] != 0 and i[1] <= time_constrain:    
                    if time_accumulation + i[1] <= limit_sec:
                        # audio_list.append(i[0])
                        audio_list.append(audio_ext)
                        time_accumulation = time_accumulation + i[1]
                        used_num = used_num + 1
                    else:
                        # print(f"<{i[1]}, {time_accumulation + i[1]}, {limit_sec}")
                        not_used_num = not_used_num + 1
                else:
                    # print(i[1])
                    not_used_num = not_used_num + 1
                    continue
    log_information(channel_name, lang_code, used_num, not_used_num, time_accumulation, limit_sec, total_time, len(item), channel_list, time_list)
    
    return audio_list
        
def log_information(channel_name, lang_code, used_num, not_used_num, time_accumulation, limit_sec, total_time, total_file_num, channel_list, time_list):
    log_file_name = "./log_stage3_ceb/" + lang_code + "-" + channel_name + "-" "log-info.txt"

    with open(log_file_name, "w") as f:
        print(f"Channel Name: {channel_name}", file=f)
        print("File Information:", file=f)
        print(f"Total Number of Audio File: {total_file_num}", file=f)
        print(f"Number of Audio File Used: {used_num}", file=f)
        print(f"Number of Audio File Not Used: {not_used_num}", file=f)
        
        print("", file=f)
        
        print("Time Information:", file=f)
        print(f"Channel total time: {total_time} sec = {convert_time(total_time)}", file=f)
        print(f"Goal of Time Distribution: {limit_sec} sec = {convert_time(limit_sec)}", file=f)
        print(f"Time Accumulation: {time_accumulation} sec = {convert_time(time_accumulation)}", file=f)
        if limit_sec != 0:
            print(f"Time Ratio: {time_accumulation / limit_sec}", file=f)
        
        data_information(channel_list, time_list, channel_name, total_time)

def data_information(channel_list, time_list, channel_name, time_length):
    channel_list.append(channel_name)
    time_list.append(time_length)

def output_list(audio_filtered_list, output_file_path):
    data_root_path = "/home/meta-531-216/mount-4/stage3/vox100/" 
        
    with open(output_file_path, "w") as f:
        for item in audio_filtered_list:
            print(item, file=f)

def get_channel_name(file_name, lang_code):
    split_file_ext = file_name.split("-metadata.json")[0]
    split_lang_code = split_file_ext.split(f"{lang_code}-")[-1]
    return split_lang_code

def cli_main(metadata_dir, lang_code, channel_name, limit_hour):
    
    metadata_file = lang_code + "-" + channel_name + "-metadata.json"
    output_file_name = lang_code + "-" + channel_name + "-" + "filtered-list.txt"
    log_file_name = "./log_stage3_ceb/" + lang_code + "-" + channel_name + "-" "log-info.txt"
    channel_list = []
    time_list = []
    with open(metadata_dir + "/" + metadata_file, "r") as f:
        audio_duration_dict = json.load(f)
    audio_filtered_list = audio_filter(audio_duration_dict, channel_name, lang_code, channel_list, time_list, limit_hour)
        
    lang_output_dir = output_dir + "/" + lang_code
    if not os.path.isdir(lang_output_dir):
        os.makedirs(lang_output_dir)
    output_file_path = lang_output_dir + "/" + output_file_name
    output_list(audio_filtered_list, output_file_path)
    
if __name__ ==  '__main__':
    
    dir_path = "/home/meta-531-216/kuanyi_stage1/stage3_metadata"             # metadata path
    output_dir = "/home/meta-531-216/kuanyi_stage1/stage3_filtered_data_ceb" # modify some lang
    
    '''Read Channel Hours Limit'''
    hour_infor_path = "/home/meta-531-216/kuanyi_stage1/stage1_hours/stage3_hour_ceb.csv"
    hour_infor_dict = pd.read_csv(hour_infor_path).to_dict()

    dict_len = len(hour_infor_dict["Channel"])
    result_list = [i for i in range(dict_len)]
    
    for index in tqdm(result_list):
        channel_name = hour_infor_dict["Channel"][index]
        lang_code = hour_infor_dict["Lang"][index]
        limit_hour = hour_infor_dict["Hour"][index]
        cli_main(dir_path, lang_code, channel_name, limit_hour)
   