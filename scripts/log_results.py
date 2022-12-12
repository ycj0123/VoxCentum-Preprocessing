import json
from sklearn.metrics import precision_score, recall_score, accuracy_score
import os

def gen_pred_results(labels=None, preds=None, total_sec=None, config=None, result_jsons=None, src_dir=None, loading_time=None, file_idx=None, num_unvoiced=0, unvoiced_idx=None, predicting_time=None, output_file=None, output_dir="output"):
    
    pred_pair = {}

    for i in range(len(file_idx)):
        pred_pair[result_jsons[file_idx[i]]] = int(preds[i])
    
    code2label = json.load(open("code2label.json"))
    
    data = {}
    data['channel'] = src_dir
    data['hours'] = total_sec / 3600
    data['n_files'] = len(result_jsons)
    data['config'] = config
    data['loading_time'] = loading_time
    data['num_unvoiced'] = num_unvoiced
    data['predicting_time'] = predicting_time
    data['precision'] = list(precision_score(labels, preds, average=None, labels=list(dict.fromkeys(code2label.values()))))
    data['recall'] = list(recall_score(labels, preds, average=None, labels=list(dict.fromkeys(code2label.values()))))
    data['accuracy'] = accuracy_score(labels, preds)
    data['pred'] = pred_pair


    if output_file is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{output_file}", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
