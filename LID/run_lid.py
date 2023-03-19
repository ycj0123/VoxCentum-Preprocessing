import argparse
import os
import torch
import nlp2
import json
from torch.utils.data import Dataset, DataLoader
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics import accuracy_score
import torchaudio
import numpy as np
import whisper
from tqdm import tqdm
from multiprocessing import Process, Pool

SAMPLE_RATE = 16000
mel_transform = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, hop_length=160, n_mels=80, norm='slaney', mel_scale="slaney")

def collate_padding_fn(batch):
    out = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return out

def batched_log_mel_spectrogram(audio):
    mel_spec = mel_transform(audio)
    
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec[:,:,:-1]  # Pop the last column

# def silero_predict(audio_path):
#     silero_model, silero_lang_dict, silero_lang_group_dict, silero_utils = torch.hub.load(
#         repo_or_dir='snakers4/silero-vad',
#         model='silero_lang_detector_95',
#         onnx=False)
#     silero_get_language_and_group, silero_read_audio = silero_utils
#     audio = silero_read_audio(audio_path, sampling_rate=SAMPLE_RATE)
#     lang, lang_group = silero_get_language_and_group(audio, silero_model, silero_lang_dict, silero_lang_group_dict, top_n=1)
#     return lang[0][0]

class AudioSamplesDataset(Dataset):
    def __init__(self, audio_files, root_dir, vad_path, chunk_sec, max_trial):
        """
        Args:
            audio_files (list): Paths to the audio files
            root_dir (str): folders containing the audio files
            vad_path (str): Path to the vad result folder
        """
        self.audio_files = audio_files
        self.root_dir = root_dir
        self.vad_path = vad_path
        self.chunk_sec = chunk_sec
        self.max_trial = max_trial

        if self.vad_path is not None:
            print(self.vad_path)
            with open(self.vad_path, 'r') as f:
                self.vad_time_stamp_dict = json.load(f)
            print("With using VAD")
        else:
            print("Without using VAD")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = os.path.join(self.root_dir, self.audio_files[idx])
        audio, sr = torchaudio.load(path)
        if len(audio.shape) == 2:
            audio = audio.mean(0)

        if sr != SAMPLE_RATE:
            print(f"ERROR: Wrong sample rate {sr}")
            exit()
        
        if self.vad_path is not None:
            vad_chunks = []
            speech_timestamps = self.vad_time_stamp_dict[path] # key: path, value: time stamps.
            for i in speech_timestamps:
                vad_chunks.append(audio[i['start']: i['end']].squeeze())

            if len(vad_chunks) != 0:
                audio = torch.cat(vad_chunks)
            else:
                audio = torch.zeros((1))

        # split audio into chunks
        chunks = list(torch.split(audio, self.chunk_sec*SAMPLE_RATE))
        if chunks[-1].shape[-1] < SAMPLE_RATE:
            concat_index = -2 if len(chunks) >= 2 else 0
            chunks[concat_index] = torch.cat(chunks[-2:])
            chunks = chunks[:concat_index + 1]
        
        idx = list(range(len(chunks)))
        np.random.shuffle(idx)
        idx = idx[:self.max_trial]
        audio = torch.concat([chunks[i] for i in idx])

        return audio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", type=str, help="Source directory")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("-v", "--vad_path", type=str, default=None, help="vad timestamps file path")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_sec", type=int, default=5)
    parser.add_argument("--max_trial", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default='output/')
    parser.add_argument("--audio_ext", type=str, default='.ogg')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--lid_whisper_enable", action='store_true')
    parser.add_argument("--lid_voxlingua_enable", action='store_true')
    parser.add_argument("--lid_silero_enable", action='store_true')
    args = parser.parse_args()

    lid_task_list = []

    for ground_truth in os.listdir(args.source_dir):
        lang_dir = os.path.join(args.source_dir, ground_truth)
        if os.path.isdir(lang_dir):
            for channel in os.listdir(lang_dir):
                channel_dir = os.path.join(lang_dir, channel)
                if os.path.isdir(channel_dir):
                    lid_task_list.append((channel_dir, ground_truth))
    
    # ECAPA-TDNN voxlingua107 model
    if args.lid_voxlingua_enable:
        voxlingua_model = EncoderClassifier.from_hparams(source="TalTechNLP/voxlingua107-epaca-tdnn",
                                                                run_opts={"device": args.device},
                                                                savedir="tmp")
        voxlingua_model = voxlingua_model.to(args.device)
        voxlingua_model.eval()
    
    # Whisper lid model
    if args.lid_whisper_enable:
        whisper_model = whisper.load_model("base")
    
    # Silero lid model
    if args.lid_silero_enable:
        silero_model, silero_lang_dict, silero_lang_group_dict, silero_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_lang_detector_95',
            onnx=False)
        silero_get_language_and_group, silero_read_audio = silero_utils

    for source_dir, ground_truth in lid_task_list:

        audio_filenames = []
        for i in nlp2.get_files_from_dir(source_dir, match=args.audio_ext): 
            try:
                audio_filenames.append(i)
            except:
                pass
        print(f"now predicting {source_dir} (ground truth: {ground_truth})")
        print("Number of audio file:", len(audio_filenames))

        audio_dataset = AudioSamplesDataset(
            audio_files = audio_filenames,
            root_dir = source_dir,
            vad_path = os.path.join(args.vad_path, "vad_time_stamp_" + ground_truth + "_" + os.path.basename(source_dir) + ".json") if args.vad_path else None,
            chunk_sec = args.chunk_sec,
            max_trial = args.max_trial
        )

        audio_dataloader = DataLoader(audio_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_padding_fn)

        vox_pred, whisper_pred, silero_pred = [], [], []
        vox_prob, whisper_prob, silero_prob = [], [], []
        

        for batch_tensor in tqdm(audio_dataloader):
            if args.lid_silero_enable:
                for audio in batch_tensor:
                    lang, lang_group = silero_get_language_and_group(audio, silero_model, silero_lang_dict, silero_lang_group_dict, top_n=1)
                    silero_pred.append(lang[0][0].split(',')[0])
                    silero_prob.append(lang[0][-1])

            if args.lid_whisper_enable:
                pad_tensor = whisper.pad_or_trim(batch_tensor)
                batch_mel_spec = batched_log_mel_spectrogram(pad_tensor).to(args.device)
                _, probs = whisper_model.detect_language(batch_mel_spec)
                whisper_pred.extend([max(prob, key=prob.get) for prob in probs])
                whisper_prob.extend([max(prob.values()) for prob in probs])
            
            batch_tensor = batch_tensor.to(args.device)

            if args.lid_voxlingua_enable:
                vox_results = voxlingua_model.classify_batch(batch_tensor)
                vox_pred.extend(vox_results[3])
                vox_prob.extend([p.item() for p in vox_results[1]])

        
        y_true = [ground_truth] * len(audio_dataset)

        voxlingua_prob = {audio_filenames[i]: vox_prob[i] for i in range(len(vox_prob))}
        whisper_prob = {audio_filenames[i]: whisper_prob[i] for i in range(len(whisper_prob))}
        silero_prob = {audio_filenames[i]: silero_prob[i] for i in range(len(silero_prob))}

        results = {
            "channel": os.path.basename(source_dir),
            "n_files": len(audio_filenames),
            "args": vars(args),
            "voxlingua_acc": accuracy_score(y_true, vox_pred) if args.lid_voxlingua_enable else "nan",
            "whisper_acc": accuracy_score(y_true, whisper_pred) if args.lid_whisper_enable else "nan",
            "silero_acc": accuracy_score(y_true, silero_pred) if args.lid_silero_enable else "nan",
            "voxlingua_prob": voxlingua_prob if args.lid_voxlingua_enable else "nan",
            "whisper_prob": whisper_prob if args.lid_whisper_enable else "nan",
            "silero_prob": silero_prob if args.lid_silero_enable else "nan"
        }

        json_obj = json.dumps(results, indent=4)
        
        os.makedirs(os.path.join(args.output_dir, ground_truth), exist_ok=True)
        with open(os.path.join(args.output_dir, ground_truth, f"{os.path.basename(source_dir)}.json"), "w") as outfile:
            outfile.write(json_obj)
        
        
