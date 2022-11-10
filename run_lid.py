import argparse
import os

from sklearn.metrics import precision_score, recall_score, accuracy_score
import nlp2
from tqdm import tqdm
import json

from lid_enhancement import AudioLIDEnhancer

code2label = json.load(open("code2label.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", type=str, default="/home/itk0123/crnn-lid/data/raw", help="Source directory")
    parser.add_argument("-w", "--workers", type=int, default=30, help="Number of workers")
    args = parser.parse_args()
    config = vars(args)
    source_dir = config['src']
    result_jsons = []
    for i in tqdm(nlp2.get_files_from_dir(source_dir, match='ogg')):
        try:
            result_jsons.append(i)
        except:
            pass
    
    wrong = []
    preds = []
    labels = []
    possible_langs = list(code2label.keys())
    lid_model = AudioLIDEnhancer(device='cuda', enable_enhancement=False, lid_voxlingua_enable=True, lid_silero_enable=True, lid_whisper_enable=True)
    for file_dir in tqdm(result_jsons, desc="LID"):
        rel_dir = os.path.relpath(file_dir, config['src'])
        label = code2label[rel_dir.split('/')[0]]
        label_code = rel_dir.split('/')[0]
        result_code = lid_model(file_dir, possible_langs=possible_langs)[0]
        result = code2label.get(result_code)
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

