from pydub import AudioSegment
import torchaudio
import os
import json
import pandas as pd
from tqdm import tqdm

''' Calculating and recording the raw duration and post-VAD duration 
for various channels in different languages. '''

def get_vad_time_dict(path):
    
    with open(path, "r") as fp:
        vad_time_dict = json.load(fp)
        
    return vad_time_dict

def sec2hour(sec):
    return round((sec / 3600), 2)

def cal_vad_time_in_channel(vad_result_dir, lang_code):

    channel_list = []
    segment_time_list = []
    raw_hour_list = []
    for vad_result in os.listdir(vad_result_dir):
        if f"_{lang_code}_" not in vad_result: continue
        
        channel_name = vad_result.split(f"vad_time_stamp_{lang_code}_")[-1]
        
        if channel_name.endswith("json"):
            channel_name = channel_name.split(".json")[0]
        else:
            channel_name = channel_name.split(".txt")[0]
        
        raw_hour_list.append(get_channel_hours(metadata_dir_path, lang_code, channel_name))

        channel_list.append(channel_name)
        vad_result_path = vad_result_dir + "/" + vad_result
        
        try:
            vad_time_dict = get_vad_time_dict(vad_result_path)
        except:
            with open(vad_result_path, "r") as f:
                vad_time_dict = eval(f.readline())

        segment = 0
        total = 0
        for item in vad_time_dict:
            audio_duration = 0
            for audio_segment in vad_time_dict[item]:
                segment_time = (audio_segment["end"] - audio_segment["start"]) / 16000
                audio_duration = audio_duration + (segment_time)
                
            total = total + audio_duration
        total = sec2hour(total)
        
        segment_time_list.append(total)
    
    return segment_time_list, channel_list, raw_hour_list

def get_audio_duration(fileName):
    waveform, sample_rate = torchaudio.load(fileName)
    duration = waveform.size()[-1] / sample_rate
    duration = duration / 3600
    return duration
    
# cal raw hours.
def cal_raw_hours(meta_dir, lang_code):
    meta_input = f"{meta_dir}/{lang_code}-list.txt"
    duration_list = []
    with open(meta_input, "r") as fp:
        for line in fp.readlines():
            line = line.strip()
            duration = get_audio_duration(line)
            duration_list.append(duration)
    
    total_time = sum(duration_list)
    total_time = round(total_time, 2)

    return total_time

def cal_raw_hours2(lang_code, meta_dir="/home/meta-531-216/kuanyi_stage1/stage3_metadata"):
    # meta_input = f"{meta_dir}/{lang_code}-list.txt"
    total_time = 0
    for metadata in os.listdir(meta_dir):
        if f"{lang_code}-" not in metadata: continue
        # print(lang_code)
        metadata = meta_dir + "/" + metadata
        with open(metadata, "r") as fp:
            meta_dict = json.load(fp)

        for item in meta_dict:
            duration = meta_dict[item]
            total_time = total_time + duration
    
    return round(total_time / 3600, 2)

def get_lang_vad_hour(lang_code):
    vad_result_path = f"/home/meta-531-216/kuanyi_stage1/stage3_vad_result/vad_time_stamp_{lang_code}.json"
    channel_list = []
    segment_time_list = []
    vad_time_dict = get_vad_time_dict(vad_result_path)
    
    segment = 0
    total = 0
    for item in vad_time_dict:
        audio_duration = 0
        for audio_segment in vad_time_dict[item]:
            # print(audio_segment["end"], audio_segment["start"])
            segment_time = (audio_segment["end"] - audio_segment["start"]) / 16000
            audio_duration = audio_duration + (segment_time)
                
        total = total + audio_duration
    total = sec2hour(total)
     
    segment_time_list.append(total)
    
    return segment_time_list, channel_list

def get_channel_hours(metadata_dir_path, lang_code, channel_name):
    
    metadata_name = f"{lang_code}-{channel_name}-metadata.json"
    metadata_path = os.path.join(metadata_dir_path, metadata_name)
    
    with open(metadata_path, "r") as fp:
        metadata_dict = json.load(fp)

    total_duration = 0
    for item in metadata_dict:
        audio_duration = metadata_dict[item]
        total_duration = total_duration + audio_duration

    total_duration = round((sec2hour(total_duration)), 2)
    
    return total_duration

def read_limit():
    hour_infor_path = "/home/meta-531-216/kuanyi_stage1/stage1_hours/stage3_hour.csv"
    hour_infor_dict = pd.read_csv(hour_infor_path).to_dict()

    dict_len = len(hour_infor_dict["Channel"])
    result_list = [i for i in range(dict_len)]
    limit_dict = {}
    for index in tqdm(result_list):
        channel_name = hour_infor_dict["Channel"][index]
        lang_code = hour_infor_dict["Lang"][index]
        limit_hour = hour_infor_dict["Hour"][index]
        limit_dict[channel_name] = limit_hour
    return limit_dict

if __name__ ==  '__main__':

    stage = 3
    
    # stage 3.
    lang_list = ["am", "as", "az", "ca", "cs", "el",
    "et", "fi", "gu", "ht", "it", "kk", "km", "kn", "ko", "mg",
    "ml", "mn", "ms", "mt", "ne", "nl", "no", "ro", "sd", "si",
    "sk", "sl", "sn", "so", "sq", "ta", "tg", "th", "tl", "vi",
    "zh-CN"]

    # stage 2.
    lang_list = ["af", "be", "bg", "bs", "fa", "gl",
    "hr", "hu", "hy", "lo", "my", "pa", "ps", "sr", "sv", "sw", "te",
    "uk", "uz", "yue", "zh-TW"]

    # stage 1.
    lang_list = ["en", "zh", "hi", "es", "fr", "ar", "bn", "ru", "pt", "ur", "id", 
              "de", "ja", "mr"]

    lang_list.sort()

    print(f"Total Language: {len(lang_list)}")

    global metadata_dir_path
    metadata_dir_path = f"/home/meta-531-216/kuanyi_stage1/stage{str(stage)}_metadata"

    vad_result_dir_path = f"/home/meta-531-216/kuanyi_stage1/stage{str(stage)}_vad_result"
    
    total_vad_time = []
    total_raw_time = []
    channel_map_list = []
    lang_map_list = []
    raw_hour_map_list = []
    filtered_list = []
    limit_dict = read_limit()

    for lang in lang_list:
        vad_time, channel_list, raw_hour_list = cal_vad_time_in_channel(vad_result_dir_path, lang)
        for i in range(len(vad_time)):
            total_vad_time.append(vad_time[i])
            channel_map_list.append(channel_list[i])
            lang_map_list.append(lang)
            raw_hour_map_list.append(raw_hour_list[i])
            if limit_dict[channel_list[i]] == -1:
                filtered_list.append(raw_hour_list[i])
            else:
                filtered_list.append(limit_dict[channel_list[i]])

    data_infor = {
        "Channel": channel_map_list,
        "Lang": lang_map_list,
        "Raw hours": raw_hour_map_list,
        "Filtered hours": filtered_list,
        "Vad hours": total_vad_time 
    }

    df = pd.DataFrame(data_infor)
    df.to_csv("channel_time_stage3_ceb.csv", encoding="utf_8_sig") 

    # total VAD hour in lang.
    lang_total_dict = dict()
    length = len(channel_map_list)
    for index in range(length):
        if lang_map_list[index] not in lang_total_dict:
            lang_total_dict[lang_map_list[index]] = 0
        lang_total_dict[lang_map_list[index]] = lang_total_dict[lang_map_list[index]] + total_vad_time[index]
    
    total_time_list = []
    for item in lang_total_dict:
        time = float(lang_total_dict[item])
        time = round(time, 2)
        total_time_list.append(time)

    # total raw hour in lang.
    lang_raw_total_dict = dict()
    length = len(raw_hour_map_list)
    for index in range(length):
        if lang_map_list[index] not in lang_raw_total_dict:
            lang_raw_total_dict[lang_map_list[index]] = 0
        lang_raw_total_dict[lang_map_list[index]] = lang_raw_total_dict[lang_map_list[index]] + raw_hour_map_list[index]
    
    total_raw_time_list = []
    for item in lang_raw_total_dict:
        time = float(lang_raw_total_dict[item])
        time = round(time, 2)
        total_raw_time_list.append(time)

    data_infor = {
        "Lang": list(dict.fromkeys(lang_total_dict)),
        "Total Raw hours": total_raw_time_list,
        "Total VAD hours": total_time_list
    }

    df = pd.DataFrame(data_infor)
    df.to_csv("lang_stage3_ceb.csv", encoding="utf_8_sig") 
    
     
    
            
            
    
    