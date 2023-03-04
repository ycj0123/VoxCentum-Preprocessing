import gc
from pathlib import Path

import torch
import torchaudio
from collections import Counter
from speechbrain.pretrained import EncoderClassifier
import whisper
import librosa
import numpy as np

import scripts.denoiser as denoiser
from scripts.denoiser.pretrained import master64
from scripts.utility import shuffle_gen, write, collate_fn_padd
torch.backends.quantized.engine = 'qnnpack'
import json
from scripts.cal_num import WHISPER_LANGS, VOX_LANGS, SILERO_LANGS
from tqdm import tqdm
from modules.code2label import code2label

class AudioLIDEnhancer:
    def __init__(self, device='cuda', dry_wet=0.01, sampling_rate=16000, chunk_sec=30, max_batch=3,
                 lid_return_n=5,
                 lid_silero_enable=True,
                 lid_voxlingua_enable=True,
                 lid_whisper_enable=True,
                 enable_enhancement=False,
                 voice_activity_detection=False):
        torchaudio.set_audio_backend("sox_io")  # switch backend
        self.device = device
        self.dry_wet = dry_wet
        self.sampling_rate = sampling_rate
        self.chunk_sec = chunk_sec
        self.chunk_length = sampling_rate * chunk_sec
        self.lid_return_n = lid_return_n
        self.lid_silero_enable = lid_silero_enable
        self.lid_voxlingua_enable = lid_voxlingua_enable
        self.lid_whisper_enable = lid_whisper_enable
        self.enable_enhancement = enable_enhancement
        self.voice_activity_detection = voice_activity_detection
        self.code2label = code2label
        self.label2code = {v:k for k, v in self.code2label.items()}
        self.whisper_langs = set([self.code2label[lang] for lang in WHISPER_LANGS.keys()])
        self.vox_langs = set([self.code2label[lang] for lang in VOX_LANGS.keys()])
        self.silero_langs = set([self.code2label[lang] for lang in SILERO_LANGS.keys()])
        self.mel_transform = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, hop_length=160, n_mels=80, norm='slaney', mel_scale="slaney")

        # Speech enhancement model
        if enable_enhancement:
            self.enhance_model = master64()
            self.enhance_model = self.enhance_model.to(self.device)
            self.enhance_model.eval()
            self.max_batch = self.get_max_batch()

        # LID model
        if lid_silero_enable:
            self.silero_model, self.silero_lang_dict, self.silero_lang_group_dict, silero_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_lang_detector_95',
                onnx=False)
            self.silero_get_language_and_group, self.silero_read_audio = silero_utils

        # LID model
        if lid_voxlingua_enable:
            self.voxlingua_language_id = EncoderClassifier.from_hparams(source="TalTechNLP/voxlingua107-epaca-tdnn",
                                                                        run_opts={"device": self.device},
                                                                        savedir="tmp")
            self.voxlingua_language_id = self.voxlingua_language_id.to(self.device)
            self.voxlingua_language_id.eval()
        
        # LID model: 99 language
        if lid_whisper_enable:
            self.whisper_model = whisper.load_model("base")
        
        # Load VAD model if using voice activity detection.
        if voice_activity_detection == True:
            USE_ONNX = False
            self.vad_model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=True,
                                              onnx=USE_ONNX)

    def get_max_batch(self):
        print("calculating max batch size...")
        batch = 1
        with torch.no_grad():
            try:
                while True:
                    self.enhance_model(torch.rand([batch, self.chunk_length]).cuda())
                    batch += 1
                    gc.collect()
                    torch.cuda.empty_cache()
            except:
                pass

        batch = max(batch - 5, 1)
        print("maximum batch size will be", batch)
        return batch

    # performance language identification on input audio,
    # if the language is one of the possible language, perform language enhancement
    # otherwise we just return the original audio
    
    def __call__(self, filepath='', input_values=[], result_path='', possible_langs=[], max_trial=10,
                 hit_times=5):
        
        no_voice_detect = 0 # initial.

        if len(filepath) > 0:
            # loading audio file and generating the enhanced version

            if self.voice_activity_detection == True: # if using VAD.
                (get_speech_timestamps,
                save_audio,
                read_audio,
                VADIterator,
                collect_chunks) = self.utils

                SAMPLING_RATE = 16000
                wav = read_audio(filepath, sampling_rate=SAMPLING_RATE)
                # get speech timestamps from full audio file
                speech_timestamps = get_speech_timestamps(wav, self.vad_model, sampling_rate=SAMPLING_RATE)

                # collect_chunks(tss: List[dict], wav: torch.Tensor):
                chunks = []
                for i in speech_timestamps:
                    chunks.append(wav[i['start']: i['end']])
                
                if len(chunks) == 0: # empty chunk list, which is unvoiced audio.
                    # print(f"No voice detected in {filepath}")
                    no_voice_detect = 1
                    out, sr = torchaudio.load(filepath)
                    out = out.mean(0).unsqueeze(0)

                else: # if chunk list is not empty.
                    out = torch.cat(chunks).unsqueeze(0)
                    sr = SAMPLING_RATE

            else: # without using vad.
                out, sr = torchaudio.load(filepath)
                out = out.mean(0).unsqueeze(0)
        else:
            out = input_values

        # split audio into chunks
        chunks = list(torch.split(out, self.chunk_length, dim=1))
        if chunks[-1].shape[-1] < self.sampling_rate:
            concat_index = -2 if len(chunks) >= 2 else 0
            chunks[concat_index] = torch.cat(chunks[-2:], dim=-1)
            chunks = chunks[:concat_index + 1]

        # total probability
        audio_langs = Counter({})

        # randomly select chunk for language detection
        for s_i in list(shuffle_gen(len(chunks)))[:max_trial]:
            # segment probability
            lid_result = Counter({})
            if self.lid_silero_enable:
                languages, language_groups = self.silero_get_language_and_group(chunks[s_i].squeeze(),
                                                                                self.silero_model,
                                                                                self.silero_lang_dict,
                                                                                self.silero_lang_group_dict,
                                                                                top_n=self.lid_return_n)
                # add the ('2 char lang_code': probability) pair to lid_result
                for l in languages:
                    lang_code = l[0].split(',')[0][:2]
                    if lang_code in lid_result:
                        lid_result[lang_code] += l[-1]
                    else:
                        lid_result[lang_code] = l[-1]

            if self.lid_voxlingua_enable:
                self.voxlingua_language_id = self.voxlingua_language_id.to(self.device)
                prediction = self.voxlingua_language_id.classify_batch(chunks[s_i].squeeze().to(self.device))
                values, indices = torch.topk(prediction[0], self.lid_return_n, dim=-1)
                # add the ('2 char lang_code': probability) pair to lid_result
                for i, l in enumerate(self.voxlingua_language_id.hparams.label_encoder.decode_torch(indices)[0]):
                    lang_code = l[:2]
                    if lang_code in lid_result:
                        lid_result[lang_code] += values[0][i].item()
                    else:
                        lid_result[lang_code] = values[0][i].item()

            if self.lid_whisper_enable:
                audio = whisper.pad_or_trim(chunks[s_i].squeeze())
                mel = whisper.log_mel_spectrogram(audio.to(self.device)).to(self.device)
                
                _, probs = self.whisper_model.detect_language(mel)
                probs_keys = list(probs.keys())
                probs_values = [probs[k] for k in probs_keys]
                values, indices = torch.topk(torch.tensor(probs_values), self.lid_return_n)

                for i in indices:
                    lang_code = probs_keys[i]
                    if lang_code in lid_result:
                        lid_result[lang_code] += probs_values[i]
                    else:
                        lid_result[lang_code] = probs_values[i]

            # add segment probability to total probability
            if len(possible_langs) == 0:
                audio_langs += lid_result
            else:
                audio_langs += dict(filter(lambda x: x[0] in possible_langs, lid_result.items()))
        
        audio_lang = max(audio_langs, key=audio_langs.get, default='na')
        # print(audio_lang)

        if self.enable_enhancement and (len(possible_langs) == 0):
            batch_data = []
            cache_batch = []
            for c in chunks:
                if len(cache_batch) >= self.max_batch:
                    batch_data.append(cache_batch)
                    cache_batch = []
                cache_batch.append(c)
            if len(cache_batch) > 0:
                batch_data.append(cache_batch)

            enhance_result = []
            for bd in batch_data:
                batch, lengths, masks = collate_fn_padd([i[0] for i in bd], self.device)
                estimate = (1 - self.dry_wet) * self.enhance_model(batch).squeeze(1) + self.dry_wet * batch
                enhance_result.append(torch.masked_select(estimate, masks).detach().cpu())

            denoise = torch.cat(enhance_result, dim=-1).unsqueeze(0)

            p = Path(filepath)
            write(denoise, str(Path(p.parent, f"{p.stem}_enhanced{p.suffix}")), sr)
            snr = denoiser.utils.cal_snr(out, denoise)
            snr = snr.cpu().detach().numpy()[0]

            return audio_lang, snr, no_voice_detect
            
        else:
            return audio_lang, 0, no_voice_detect
    
    def batched_log_mel_spectrogram(self, audio):
        # mel_spec = torch.tensor(librosa.feature.melspectrogram(y=np.array(audio.cpu()), sr=self.sampling_rate, n_fft=400, hop_length=160, n_mels=80))
        mel_spec = self.mel_transform(audio)
        
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def forward(self, X, possible_langs=None):

        lid_result = [{} for _ in range(X.shape[0])]

        if self.lid_voxlingua_enable:
            out_prob, score, index, text_lab = self.voxlingua_language_id.classify_batch(X.to(self.device))
            values, indices = torch.topk(out_prob, self.lid_return_n, dim=1)

            # print("voxlingua")
            # add the ('2 char lang_code': probability) pair to lid_result
            for b_i in range(values.shape[0]):
                for i, l in enumerate(self.voxlingua_language_id.hparams.label_encoder.decode_torch(indices[b_i])):
                    lang_code = l
                    # print(f"{lang_code}:{values[b_i][i]}")
                    if lang_code in lid_result[b_i]:
                        lid_result[b_i][lang_code] += values[b_i][i].item()
                    else:
                        lid_result[b_i][lang_code] = values[b_i][i].item()
        
        if self.lid_whisper_enable:
            audio = whisper.pad_or_trim(X)
            mel = self.batched_log_mel_spectrogram(audio).to(self.device)
            tokens, probs = self.whisper_model.detect_language(mel[:,:,:-1]) # Pop the last column

            # print("whisper")
            for b_i in range(len(probs)):
                probs_keys = list(probs[b_i].keys())
                probs_values = [probs[b_i][k] for k in probs_keys]
                values, indices = torch.topk(torch.tensor(probs_values), self.lid_return_n)
                for i in indices:
                    lang_code = probs_keys[i]
                    # print(f"{lang_code}:{probs_values[i]}")
                    if lang_code in lid_result[b_i]:
                        lid_result[b_i][lang_code] += probs_values[i]
                    else:
                        lid_result[b_i][lang_code] = probs_values[i]

        pred = [max(batch_pred, key=batch_pred.get) for batch_pred in lid_result]
        return pred

    def update_lid_result(self, audio_idx, lid_result, pred_keys, pred_probs):
        pred_probs = [pred_probs[i] / sum(pred_probs) for i in range(self.lid_return_n)]
        for pred_i in range(self.lid_return_n):
            lang_code = pred_keys[pred_i]
            prob = pred_probs[pred_i]
            if lang_code in lid_result[audio_idx]:
                lid_result[audio_idx][lang_code] += prob
            else:
                lid_result[audio_idx][lang_code] = prob

    
    def seq_forward(self, X, Y):
        lid_result = [{} for _ in range(len(X))]

        for b_i in range(len(X)):
            audio_tensor = X[b_i]
            if audio_tensor.size()[0] == 2:
                audio_tensor = audio_tensor.mean(0)
            if len(audio_tensor.size()) == 1:
                audio_tensor = audio_tensor.reshape(1, -1)
            if len(audio_tensor.size()) != 2 or audio_tensor.size()[0] != 1:
                print(f"wrong input size: {audio_tensor.size()}")
                exit()

            if self.lid_silero_enable and Y[b_i] in self.silero_langs:
                pred_lang, _ = self.silero_get_language_and_group(audio_tensor.squeeze(), self.silero_model, self.silero_lang_dict, self.silero_lang_group_dict, top_n=self.lid_return_n)
                pred_keys, pred_probs = [pred_lang[i][0].split(',')[0] for i in range(self.lid_return_n)], [pred_lang[i][1] for i in range(self.lid_return_n)]
                # print("silero:", pred_keys, [pred_probs[i] / sum(pred_probs) for i in range(self.lid_return_n)])
                self.update_lid_result(b_i, lid_result, pred_keys, pred_probs)

            if self.lid_voxlingua_enable and Y[b_i] in self.vox_langs:
                out_prob, score, index, text_lab = self.voxlingua_language_id.classify_batch(audio_tensor.to(self.device))
                values, indices = torch.topk(out_prob, self.lid_return_n, dim=1)

                pred_keys = []
                pred_probs = []
                for i, l in enumerate(self.voxlingua_language_id.hparams.label_encoder.decode_torch(indices[0])):
                    pred_keys.append(l)
                    pred_probs.append(values[0][i].item())
                
                # print("voxlingua:", pred_keys, [pred_probs[i] / sum(pred_probs) for i in range(self.lid_return_n)])
                self.update_lid_result(b_i, lid_result, pred_keys, pred_probs)
            
            if self.lid_whisper_enable and Y[b_i] in self.whisper_langs:
                audio = whisper.pad_or_trim(audio_tensor.reshape(1, -1))
                mel = self.batched_log_mel_spectrogram(audio).to(self.device)
                tokens, probs = self.whisper_model.detect_language(mel[:,:,:-1]) # Pop the last column

                pred_keys = list(probs[0].keys())
                pred_probs = [probs[0][k] for k in pred_keys]
                values, indices = torch.topk(torch.tensor(pred_probs), self.lid_return_n)

                pred_keys = [pred_keys[i] for i in indices]
                pred_probs = [pred_probs[i] for i in indices]
                
                # print("whisper:", pred_keys, [pred_probs[i] / sum(pred_probs) for i in range(self.lid_return_n)])
                self.update_lid_result(b_i, lid_result, pred_keys, pred_probs)
            
            if (Y[b_i] not in self.whisper_langs) and (Y[b_i] not in self.vox_langs) and (Y[b_i] not in self.silero_langs):
                print("[ERR] language not found.")
                print(Y[b_i])
                exit()

        pred = [max(batch_pred, key=batch_pred.get) for batch_pred in lid_result]
        # pred = [self.code2label[lid] for lid in pred]
        # print(pred)
        return pred
