import json
from sklearn.metrics import accuracy_score
import os
from modules.code2label import code2label

def gen_pred_results(labels=None, preds=None, total_sec=None, config=None, audio_filenames=None, src_dir=None, loading_time=None, file_idx=None, num_unvoiced=0, unvoiced_idx=None, predicting_time=None, output_file=None, output_dir="output", output_lang='en', is_whisper=False, is_silero=False, is_vox=False):
    
    pred_pair = {}

    for i in range(len(file_idx)):
        try:
            pred_pair[audio_filenames[file_idx[i]]] = int(preds[i])
        except:
            print(audio_filenames)
            print(file_idx)
            print(preds)
            exit()
    
    data = {}
    data['channel'] = src_dir
    data['hours'] = total_sec / 3600
    data['n_files'] = len(audio_filenames) - num_unvoiced
    data['config'] = config
    data['loading_time'] = loading_time
    data['num_unvoiced'] = num_unvoiced
    data['predicting_time'] = predicting_time
    data['accuracy'] = accuracy_score(labels, preds)
    data['pred'] = pred_pair
    data['is_whisper'] = is_whisper
    data['is_vox'] = is_vox
    data['is_silero'] = is_silero


    if output_file is not None:
        output_dir = os.path.join(output_dir, output_lang)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, output_file), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
