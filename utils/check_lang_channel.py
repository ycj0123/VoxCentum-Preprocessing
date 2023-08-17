import os
import json
import pandas as pd
from tqdm import tqdm

''' Calculating the original duration for each channel. '''

def sec2hour(sec):
    return round(sec / 3600, 2)

def get_channel_hours(metadata_dir_path, lang_code, channel_name):
    
    metadata_name = f"{lang_code}-{channel_name}-metadata.json"
    metadata_path = os.path.join(metadata_dir_path, metadata_name)
    with open(metadata_path, "r") as fp:
        metadata_dict = json.load(fp)
    
    total_duration = 0
    for item in metadata_dict:
        audio_duration = metadata_dict[item]
        total_duration = total_duration + audio_duration

    total_duration = int(sec2hour(total_duration))
    
    return total_duration

if __name__ ==  '__main__':
    stage2_audio_path = "/home/meta-531-216/mount-4/stage3/vox100"
    stage2_metadata_dir_path = "/home/meta-531-216/kuanyi_stage1/stage3_metadata"
    
    lang_list = []
    channel_list = []
    channel_duration_list = []
    for lang_code in os.listdir(stage2_audio_path):
        if lang_code not in ["ceb"]: continue
        channel_path = os.path.join(stage2_audio_path, lang_code)
        for channel in os.listdir(channel_path):
            try:
                channel_duration = get_channel_hours(stage2_metadata_dir_path, lang_code, channel)
                # print(channel_duration)
                channel_duration_list.append(channel_duration)
                lang_list.append(lang_code)
                channel_list.append(channel)
            except:
                pass
    
    data_infor = {
        "Channel": channel_list,
        "Lang": lang_list,
        "Hours": channel_duration_list
    }
    df = pd.DataFrame(data_infor)
    df.to_csv("stage3_ceb_channel.csv")

    lang_total_hour = dict()
    for index in range(len(channel_list)):
        if lang_list[index] not in lang_total_hour:
            lang_total_hour[lang_list[index]] = 0
        lang_total_hour[lang_list[index]] += channel_duration_list[index]
    lang_list = (list(lang_total_hour))
    hour_list = []
    for item in lang_total_hour:
        hour_list.append(lang_total_hour[item])
    
    data_infor = {
        "Lang": lang_list,
        "Hours": hour_list
    }

    df = pd.DataFrame(data_infor)
    df.to_csv("stage3_ceb.csv")