import argparse
import os

from sklearn.metrics import precision_score, recall_score, accuracy_score
import nlp2
from tqdm import tqdm
import json

from lid_enhancement import AudioLIDEnhancer

import torch
from scripts.multi_thread_loader import MultiThreadLoader
import pickle, time, threading
from math import ceil

code2label = json.load(open("code2label.json"))

Voxlingua107_dataset_path = '/storage/PromptGauguin/kuanyi/dev'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", type=str, default=Voxlingua107_dataset_path, help="Source directory")
    parser.add_argument("-w", "--workers", type=int, default=3, help="Number of workers")
    parser.add_argument("-v", "--vad", type=bool, default=False, help="Voice Activity Detection")
    parser.add_argument("--vad_path", type=str, default='./vad_timestamp_ja.txt', help="vad timestamps file path")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_sec", type=int, default=30)
    parser.add_argument("--max_trial", type=int, default=10)
    parser.add_argument("--wav", action='store_true', help="Use .wav file extension")
    args = parser.parse_args()
    config = vars(args)
    source_dir = config['src']
    voice_activity_detection = config['vad']
    vad_timestamp_path = config['vad_path']
    n_workers = config['workers']
    batch_size = config['batch_size']
    chunk_sec = config['chunk_sec']
    max_trial = config['max_trial']

    if voice_activity_detection == True:
        with open(vad_timestamp_path, 'r') as f:
            vad_time_stamp_dict = eval(f.readline())

        print("With using VAD")
    else:
        vad_time_stamp_dict = dict()
        print("Without using VAD")

    result_jsons = []
    # result_jsons = ["../nas-60022/vox100_ar/aljazeera/" + s.strip() for s in open("../vox100_ar-aljazeera.txt", "r").readlines()]
    # Voxlingua107, match='wav'
    # raw, match='ogg'

    audio_ext = '.ogg' if not config['wav'] else ".wav"

    for i in tqdm(nlp2.get_files_from_dir(source_dir, match=audio_ext)): 
        try:
            result_jsons.append(i)
        except:
            pass

    # Set up multi-thread loader
    loader = MultiThreadLoader(
        n_workers = n_workers, 
        batch_size = batch_size, 
        n_files = len(result_jsons),
        max_trial = max_trial,
        chunk_sec = chunk_sec,
        vad = voice_activity_detection,
        vad_time_stamp_dict = vad_time_stamp_dict,
    )
    loader.start(files=result_jsons)
    ready_data_idx = 0
    
    wrong = []
    preds = []
    labels = []
    possible_langs = list(code2label.keys())
    lid_model = AudioLIDEnhancer(device='cuda', enable_enhancement=False, lid_voxlingua_enable=True, lid_silero_enable=False, lid_whisper_enable=True, voice_activity_detection=voice_activity_detection)
    skip_audio_list = []

    progress = tqdm(total=ceil(len(result_jsons) / batch_size), desc="PRED", position=1)
    while True:
        X, y, l, r = loader.get_data() # X is list of batched data

        if loader.no_more_data:
            break

        elif X is not None:
            for i in range(len(X)):
                pred = lid_model.forward(X[i])
                preds.extend([code2label[code] for code in pred])
                labels.extend(y[i])
                progress.update(1)

            loader.free_data(l, r)

        time.sleep(1)

        # if no_voice_detect == True: # no voice in the audio input
        #     # print("Skip this unvoiced audio file.")
        #     skip_audio_list.append(file_dir)
        #     continue

        # if result == None:
        #     result = 0
        # if result != label:
        #     wrong.append((result_code, label_code))
        # preds.append(result)
        # labels.append(label)
    
    print(preds)
    print(labels)

    print('Wrong predictions: ', [f'pred: {w[0]}, gt: {w[1]}' for w in wrong])
    print('Precision: ', precision_score(labels, preds, average=None, labels=list(dict.fromkeys(code2label.values()))))
    print('Recall: ', recall_score(labels, preds, average=None, labels=list(dict.fromkeys(code2label.values()))))
    print('Accuracy: ', accuracy_score(labels, preds))

    # with open('unvoiced.txt', 'w') as output_file:
    #     for item in skip_audio_list:
    #         output_file.write("%s\n" % item)
    

