import argparse
import os

from sklearn.metrics import precision_score, recall_score, accuracy_score
import nlp2
from tqdm import tqdm
import json
import ast

from lid_enhancement import AudioLIDEnhancer

import torch
from scripts.multi_thread_loader import MultiThreadLoader
from scripts.log_results import gen_pred_results
import pickle, time, threading
from math import ceil

code2label = json.load(open("code2label.json"))

Voxlingua107_dataset_path = '/storage/PromptGauguin/kuanyi/dev'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", type=str, default=Voxlingua107_dataset_path, help="Source directory")
    parser.add_argument("-w", "--workers", type=int, default=3, help="Number of workers")
    parser.add_argument("-v", "--vad", action='store_true', help="Voice Activity Detection")
    parser.add_argument("--vad_output", type=str, default=None, help="Timestamps from VAD output")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_sec", type=int, default=30)
    parser.add_argument("--max_trial", type=int, default=10)
    parser.add_argument("--wav", action='store_true', help="Use .wav file extension")
    args = parser.parse_args()
    config = vars(args)
    source_dir = config['src']
    voice_activity_detection = config['vad']
    n_workers = config['workers']
    batch_size = config['batch_size']
    chunk_sec = config['chunk_sec']
    max_trial = config['max_trial']

    if voice_activity_detection == True:
        print("With using VAD")
        vad_output = ast.literal_eval(open(config['vad_output'], "r").readline())
        vad_output = {k.split('/')[-1]: v for k, v in vad_output.items()}
    else:
        print("Without using VAD")

    result_jsons = []

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
        vad_output = vad_output if voice_activity_detection else None,
    )
    loader.start(files=result_jsons)
    ready_data_idx = 0
    
    wrong = []
    preds = []
    labels = []
    possible_langs = list(code2label.keys())
    lid_model = AudioLIDEnhancer(device='cuda', enable_enhancement=False, lid_voxlingua_enable=True, lid_silero_enable=False, lid_whisper_enable=True, voice_activity_detection=False)
    skip_audio_list = []

    progress = tqdm(total=ceil(len(result_jsons) / batch_size), desc="PRED", position=1)
    timeout_cnt = 5
    while True:
        X, y, l, r = loader.get_data() # X is list of batched data

        if loader.no_more_data:
            if timeout_cnt < 0:
                exit(2)
            if len(result_jsons) - loader.num_unvoiced != len(preds):
                print(f"waiting: {len(result_jsons)} != {len(preds)}") # 880 878or872
                print(loader.last_batched_idx) # 110
                print(len(loader.batched_inputs)) # 110
                print(len(loader.audio_tensors)) # 878
                timeout_cnt -= 1
            else:
                break

        elif X is not None:
            for i in range(len(X)):
                try:
                    pred = lid_model.forward(X[i])
                except RuntimeError:
                    exit(2)
                preds.extend([code2label[code] for code in pred])
                labels.extend(y[i])
                progress.update(1)

            loader.free_data(l, r)

        time.sleep(1)
    
    if voice_activity_detection:
        output_file_name = f"{source_dir.split('/')[-1].strip()}_vad_output.txt"
    else:
        output_file_name = f"{source_dir.split('/')[-1].strip()}_output.txt"

    gen_pred_results(
        labels=labels,
        preds=preds,
        total_sec=loader.total_sec,
        config=config,
        result_jsons=result_jsons, 
        src_dir=source_dir, 
        loading_time=loader.loading_time,
        file_idx=loader.file_idx,
        num_unvoiced=loader.num_unvoiced,
        unvoiced_idx=loader.unvoiced_idx,
        predicting_time=progress.format_dict['elapsed'],
        output_file=output_file_name,
        output_dir="output/" + source_dir.split('/')[-2].strip()
    )
