import argparse
import os

from sklearn.metrics import precision_score, recall_score, accuracy_score
import nlp2
from tqdm import tqdm
import json

from lid_enhancement import AudioLIDEnhancer

import torch

code2label = json.load(open("code2label.json"))

Voxlingua107_dataset_path = '/storage/PromptGauguin/kuanyi/dev'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", type=str, default=Voxlingua107_dataset_path, help="Source directory")
    parser.add_argument("-w", "--workers", type=int, default=30, help="Number of workers")
    parser.add_argument("-v", "--vad", type=bool, default=False, help="Voice Activity Detection")
    args = parser.parse_args()
    config = vars(args)
    source_dir = config['src']
    voice_activity_detection = config['vad']

    if voice_activity_detection == True:
        print("With using VAD")
    else:
        print("Without using VAD")

    result_jsons = []
    # Voxlingua107, match='wav'
    # raw, match='ogg'
    for i in tqdm(nlp2.get_files_from_dir(source_dir, match='ogg')): 
        try:
            result_jsons.append(i)
        except:
            pass
    
    wrong = []
    preds = []
    labels = []
    possible_langs = list(code2label.keys())
    lid_model = AudioLIDEnhancer(device='cuda', enable_enhancement=False, lid_voxlingua_enable=True, lid_silero_enable=True, lid_whisper_enable=True, voice_activity_detection=voice_activity_detection)
    skip_audio_list = []

    for file_dir in tqdm(result_jsons, desc="LID"):
        rel_dir = os.path.relpath(file_dir, config['src'])
        label = code2label[rel_dir.split('/')[0]]
        label_code = rel_dir.split('/')[0]

        result_code, _, no_voice_detect = lid_model(file_dir, possible_langs=possible_langs)
        result = code2label.get(result_code)
        
        if no_voice_detect == True: # no voice in the audio input
            # print("Skip this unvoiced audio file.")
            skip_audio_list.append(file_dir)
            continue

        if result == None:
            result = 0
        if result != label:
            wrong.append((result_code, label_code))
        preds.append(result)
        labels.append(label)

    print('Wrong predictions: ', [f'pred: {w[0]}, gt: {w[1]}' for w in wrong])
    print('Precision: ', precision_score(labels, preds, average=None))
    print('Recall: ', recall_score(labels, preds, average=None))
    print('Accuracy: ', accuracy_score(labels, preds))

    with open('unvoiced.txt', 'w') as output_file:
        for item in skip_audio_list:
            output_file.write("%s\n" % item)
    

