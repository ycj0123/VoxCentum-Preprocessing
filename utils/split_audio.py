# split audio into fixed segment, e.g. 10-sec segemnts
import os
import torchaudio
import torch
import ast
from tqdm import tqdm
import multiprocessing as mp
import time
import json
import random
import pandas as pd

sampling_rate = 16000

''' Segmenting each audio based on the filtered data list after VAD into 10-second segments. '''

def hour2sec(hour):
    return hour * 60 * 60

def random_shuffle_dict(vad_time_dict):
    vad_time_list = list(vad_time_dict.items())
    random.seed(1216) # fix.
    random.shuffle(vad_time_list)
    vad_time_dict_shuffle = dict(vad_time_list)
    
    return vad_time_dict_shuffle

def get_split_audio_list(vad_time_dict, limit_segment_30=0, limit_segment_10=180, limit_segment_3=0):
    ''' Get 30/10/3 sec split data list '''
    # random shuffle dict
    vad_time_dict = random_shuffle_dict(vad_time_dict)
    
    # initial
    num_segment_10 = 0
    
    # limit_segment (hour) -> (sec)
    limit_num_segment_10 = hour2sec(limit_segment_10) / 10
    
    # segment_10 data path list.
    list_segment_10 = []
    
    if limit_num_segment_10 < 0: # all used.
        for item in vad_time_dict:
            vad_segment_time = vad_time_dict[item]
            
            if vad_segment_time >= 10:
                num_segment_10 = num_segment_10 + int(vad_segment_time / 10)
                list_segment_10.append(item)

            else:
                continue

    else: # hours limit.
        for item in vad_time_dict:
            vad_segment_time = vad_time_dict[item]
            
            if vad_segment_time >= 10 and (num_segment_10 + int(vad_segment_time / 10)) < limit_num_segment_10:
                num_segment_10 = num_segment_10 + int(vad_segment_time / 10)
                list_segment_10.append(item)

            else:
                continue

    print(f"10-sec: {num_segment_10 * 10 / 3600} hr.")

    return list_segment_10

# collect vad segment time in one channel
def get_total_duration_vad_segment_in_channel(vad_result_path, extension="json"):
    
    sampling_rate = 16000
    vad_result_time_dict = dict()
    
    try:
        with open(vad_result_path, "r") as f:
            vad_result = eval(f.readline())
    except:
        with open(vad_result_path, "r") as f:
            vad_result = json.load(f)
   
    for item in vad_result:
        channel_sum = 0
        for time_stamp in vad_result[item]:
            start = time_stamp["start"]
            end = time_stamp["end"]
            segment = (end - start) / sampling_rate
            channel_sum = channel_sum + segment
        vad_result_time_dict[item] = int(channel_sum)
    
    return vad_result_time_dict
   

# collect different channel vad segment time in one language.
def get_total_duration_vad_segment_for_lang(lang_code, vad_result_dir_path="/home/meta-531-216/kuanyi_stage1/stage3_vad_result"):
    
    vad_time_dict = dict() # store (audio path, sec) for the specific language.
    for result_path in os.listdir(vad_result_dir_path):
        if f"_{lang_code}_" not in result_path and f"_{lang_code}" not in result_path: continue
        
        result_path = vad_result_dir_path + "/" + result_path
        vad_time_dict.update(get_total_duration_vad_segment_in_channel(result_path))
        
    return vad_time_dict

def get_vad_result_dict_for_lang(lang_code, vad_result_dir_path="/home/meta-531-216/kuanyi_stage1/stage3_vad_result", extension="json"):
    vad_result_dict = dict()
    for result_path in os.listdir(vad_result_dir_path):
        if f"_{lang_code}_" not in result_path and f"_{lang_code}" not in result_path: continue
        result_path = vad_result_dir_path + "/" + result_path
        
        try:
            with open(result_path, "r") as f:
                vad_result_for_channel = eval(f.readline())
        except:
            with open(result_path, "r") as f:
                vad_result_for_channel = json.load(f)
        
        vad_result_dict.update(vad_result_for_channel)
    
    return vad_result_dict

def concat_vad_segment(audio_path, audio_data, vad_result_dict):
    vad_chunks = []
    
    for segment in vad_result_dict[audio_path]:
        vad_chunks.append(audio_data[:, segment['start']: segment['end']])
    
    if len(vad_chunks) != 0: # if chunk list is not empty.
        out = torch.cat(vad_chunks, dim=1)
    else:
        out = audio_data
        print(f"{audio_path} has no speech.")
    
    return out

def split_audio(audio_path, segment_duration, output_dir):
    
    audio_name = audio_path.split("/")[-1].split(".wav")[0]
    
    # out, sr = torchaudio.load(audio_path.replace("vox100_stage2", "vox100_stage2_abor")) # out: (channel_num, length)
    out, sr = torchaudio.load(audio_path) # out: (channel_num, length)
    
    if len(vad_result_dict) != 0: # if audio has speech.
        out = concat_vad_segment(audio_path, out, vad_result_dict)
    else: # if audio has no speech.
        return 0
    
    channel_num = out.size()[0]
    audio_duration = int(out.size()[1] / sr)
    split_num = int(audio_duration / segment_duration)
    
    for split in range(split_num):
        output_audio_segment_name = f"{audio_name}-{segment_duration}-{split}.wav"
        output_audio_segment_path = output_dir + "/" + output_audio_segment_name
        out_split = out[:, split * segment_duration * sr: (split + 1) * segment_duration * sr]
        try:
            torchaudio.save(output_audio_segment_path, out_split, sr, format="wav")
        except:
            print(f"Error in {audio_path}.")

    return 0    

def multi_wrapper(args):
    return split_audio(*args)       
    
def split_log(split_log_dir, lang_code, list_segment_10):
    split_log_path = f"./{split_log_dir}/{lang_code}-used.txt"
    with open (split_log_path, "w") as fp:
        for path in list_segment_10:
            print(path, file=fp)

if __name__ ==  '__main__':
    stage1 = ["en", "zh", "hi", "es", "fr", "ar", "bn", "ru", "pt", "ur", "id", "de", "ja", "mr"] # done.
    stage2_step1 = ["am", "as", "az", "ca", "cs", "el", "et", "fi", "gu", "ht"] # done.
    stage2 = ["it", "kk", "km", "kn", "ko", "mg", "ml", "mn", "ms", "mt", "ne", "nl", "no", "ro", "sd", "si", "sk", "sl", "sn", "so", "sq", "ta", "tg", "th", "tl", "vi", "zh-CN"]
    stage2_step2 = ["it", "kk", "km", "kn", "ko", "mg", "ml", "mn", "ms", "mt", "ne", "nl", "no", "ro"] # done.
    stage2_step3 = ["sd", "si", "sk", "sl", "sn", "so", "sq", "ta", "tg", "th", "tl", "vi", "zh-CN"]
    stage2_abor_lang = ["ami", "bnn", "ckv", "dru", "pwn", "pyu", "ssf", "sxr", "szy", "tao", "tay", "trv", "trv_", "tsu", "xnb", "xsy"]
    stage2_part2 = ["af", "be", "bg", "bs", "fa", "gl", "hr", "hu", "hy", "lo", "my", "pa", "ps", "sr", "sv", "sw", "te", "uk", "uz", "yue", "zh-TW"]
    stage2_part3 = ["da", "la", "mk", "pl", "tr"]
    stage2_part4 = ["hak", "oan"]
    stage3_part1 = ["ba", "bo", "ceb", "cy", "eo", "eu", "fo", "ha", "is", "ka", "lt", "lv", "oc", "tk", "tt", "yi", "yo", "ckb", "ce", "bm", "gl"]
    stage3_ceb = ["ceb"]

    split_hour_path = "/home/meta-531-216/kuanyi_stage1/stage1_hours/stage3_split.csv"
    split_limit_dict = pd.read_csv(split_hour_path).to_dict()
    split_hour_dict = {}

    length = len(split_limit_dict["Lang"])

    for index in range(length):
        split_hour_dict[split_limit_dict["Lang"][index]] = split_limit_dict["Split"][index]
    
    for lang_code in stage3_ceb:
        # vad_time_dict: all vad segments in one language.
        # vad_time_dict = get_total_duration_vad_segment_for_lang(lang_code, vad_result_dir_path="/home/meta-531-216/kuanyi_stage1/stage2_vad_result")
        
        # aboriginal.
        vad_time_dict = get_total_duration_vad_segment_for_lang(lang_code, vad_result_dir_path="/home/meta-531-216/kuanyi_stage1/stage3_vad_result")

        list_segment_10 = get_split_audio_list(vad_time_dict, limit_segment_10=split_hour_dict[lang_code])

        # get the segments we are goint to split into 10-sec segments.
        # if lang_code in ["hak"]:
        #     list_segment_10 = get_split_audio_list(vad_time_dict, limit_segment_10=-1)
        # else:
        #     list_segment_10 = get_split_audio_list(vad_time_dict, limit_segment_10=180)
        
        # output log.
        split_log_dir = "stage3_split_log"
        split_log(split_log_dir, lang_code, list_segment_10)

        # get each segment start and end time.
        global vad_result_dict
        vad_result_dict = get_vad_result_dict_for_lang(lang_code, vad_result_dir_path="/home/meta-531-216/kuanyi_stage1/stage3_vad_result")
      
        # using multi-processing
        # split into 10-sec-segment
        duration_list = [10] * len(list_segment_10)
        print(f"Processing 10-sec segments of {lang_code}.")
        # output_dir_path = f"/home/meta-531-216/mount-4/VoxCentum_stage3/{lang_code}"
        output_dir_path = f"/home/meta-531-216/mount-4/VoxCentum_stage3_ceb/{lang_code}"

        if not os.path.isdir(output_dir_path):
            os.makedirs(output_dir_path)
                
        output_dir_list = [output_dir_path] * len(list_segment_10)
        zip_args = list(zip(list_segment_10, duration_list, output_dir_list))
            
        pool = mp.Pool(4)
        results = []
        for result in tqdm(pool.imap_unordered(multi_wrapper, zip_args),total=len(list_segment_10)):
            results.append(result)
    
    print("Done, all the audio has been split.")
    