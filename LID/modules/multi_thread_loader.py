import threading, signal, torchaudio, json, torch
from time import sleep
from tqdm import tqdm
import numpy as np
from math import ceil
from scripts.utility import shuffle_gen
from modules.code2label import code2label

LOCK_TIMEOUT = 3
SAMPLE_RATE = 16000

class MultiThreadLoader():

    def __init__(self, n_workers=3, batch_size=16, n_files=None, max_trial=10, chunk_sec=30, vad_path=False, ground_truth='en', use_vad=False):

        # Locks
        self.lock = {
            "data": threading.Lock(),
            "worker_cnt": threading.Lock(),
        }

        # Shared resources: data
        self.audio_tensors = []
        self.lid_labels = []
        self.batched_inputs = []
        self.batched_labels = []
        self.unvoiced_idx = []
        self.file_idx = []
        self.total_sec = 0
        self.num_unvoiced = 0
        self.ground_truth = ground_truth

        # Shared resources: worker_cnt
        self.n_workers = n_workers

        self.interrupt = False
        self.no_more_data = False
        self.grouper_left = False

        self.threads = []
        self.code2label = code2label
        self.batch_size = batch_size
        self.n_files = n_files
        self.n_steps = ceil(self.n_files / self.batch_size)
        self.last_batched_idx = 0
        self.loading_time = None
        self.vad_path = vad_path
        
        # Chunk split
        self.max_trial = max_trial
        self.chunk_sec = chunk_sec
        self.chunk_len = chunk_sec * SAMPLE_RATE
        
        # Use VAD
        self.use_vad = use_vad
        if self.use_vad:
            USE_ONNX = False
            self.vad_model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=True,
                                              trust_repo=True,
                                              onnx=USE_ONNX)
            (self.get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks) = self.utils

    def get_data_lock(self, lock):
        if self.interrupt:
            # print("A worker left.")
            exit()
        get_lock = lock.acquire(LOCK_TIMEOUT)
        while not get_lock:
            if self.interrupt:
                # print("A worker left.")
                exit()
            get_lock = lock.acquire(LOCK_TIMEOUT)
    
    def split_into_chunks(self, audio):
        # split audio into chunks
        chunks = list(torch.split(audio, self.chunk_len))
        if chunks[-1].shape[-1] < SAMPLE_RATE:
            concat_index = -2 if len(chunks) >= 2 else 0
            chunks[concat_index] = torch.cat(chunks[-2:], dim=-1)
            chunks = chunks[:concat_index + 1]
        return chunks

    def start(self, files):
        # files: list of audio file paths to be loaded

        def int_handler(signum, frame):
            print("Recieve SIGINT. Exiting")
            self.interrupt = True
            exit()
        
        def load_worker_func(l, r):
            buffer = [[], [], []] # X, y, file_idx
            buffer_sec = 0
            for i in range(l, r):
                path = files[i]
                try:
                    out, sr = torchaudio.load(path)
                    sec = out.shape[0] / SAMPLE_RATE if len(out.shape) == 1 else out.shape[1] / SAMPLE_RATE
                except:
                    print(path)
                    self.num_unvoiced += 1
                    continue
                try:

                    if sr != SAMPLE_RATE:
                        print(f"ERROR: Wrong sample rate {sr}")
                        self.interrupt = True
                        exit()

                    if self.vad == True:
                        vad_chunks = []
                        speech_timestamps = self.vad_time_stamp_dict['path'] # key: path, value: time stamps.
                        for i in speech_timestamps:
                            vad_chunks.append(out[i['start']: i['end']])

                        if len(vad_chunks) != 0: # if chunk list is not empty.
                            out = torch.cat(vad_chunks).unsqueeze(0)
                            sr = SAMPLE_RATE
                            
                    else: # not using vad
                        out = out.mean(0).squeeze()
                except:
                    pass
                    
                out = out.mean(0).squeeze()

                if self.vad_path is not None:
                    timestamp = self.vad_path[path.split('/')[-1]]
                    if len(timestamp) > 0:
                        out = torch.concat([out[obj['start']: obj['end']] for obj in timestamp])
                    else:
                        self.get_data_lock(self.lock["data"])
                        self.num_unvoiced += 1
                        self.unvoiced_idx.append(i)
                        self.lock["data"].release()
                        continue
                
                
                chunks = self.split_into_chunks(out)
                audio = torch.concat([chunks[s_i] for s_i in list(shuffle_gen(len(chunks)))[:self.max_trial]])
                if self.use_vad:
                    self.get_data_lock(self.lock["data"])
                    speech_timestamps = self.get_speech_timestamps(audio, self.vad_model, sampling_rate=SAMPLE_RATE)
                    chunks = []
                    for j in speech_timestamps:
                        chunks.append(audio[j['start']: j['end']])
                    
                    if len(chunks) == 0: # empty chunk list, which is unvoiced audio.
                        print("unvoiced")
                        self.num_unvoiced += 1
                        self.unvoiced_idx.append(i)
                        self.lock["data"].release()
                        continue

                    else: # if chunk list is not empty.
                        audio = torch.cat(chunks).unsqueeze(0)
                    self.lock["data"].release()

                buffer[0].append(audio)
                # buffer[1].append(self.code2label[path.split('/')[-3].split('_')[-1]])
                buffer[1].append(self.code2label[self.ground_truth])
                buffer[2].append(i)
                buffer_sec += sec

                if len(buffer[0]) >= self.batch_size:
                    self.get_data_lock(self.lock["data"])
                    self.audio_tensors.extend(buffer[0])
                    self.lid_labels.extend(buffer[1])
                    self.file_idx.extend(buffer[2])
                    self.total_sec += buffer_sec
                    self.lock["data"].release()
                    buffer = [[], [], []]
                    buffer_sec = 0
            
            if len(buffer[0]) > 0:
                self.get_data_lock(self.lock["data"])
                self.audio_tensors.extend(buffer[0])
                self.lid_labels.extend(buffer[1])
                self.file_idx.extend(buffer[2])
                self.total_sec += buffer_sec
                self.lock["data"].release()

            self.lock["worker_cnt"].acquire()
            self.n_workers -= 1
            self.lock["worker_cnt"].release()
        
        def group_worker_func():
            t0 = 0
            progress = tqdm(total=self.n_steps, desc="LOAD", position=0)

            while self.num_worker_left() > 0:
                # Check whether data is ready for grouping
                self.get_data_lock(self.lock["data"])
                num_audio = len(self.audio_tensors)
                self.lock["data"].release()

                if num_audio-t0 >= self.batch_size:
                    # print(f"Grouping idx from {t0} to {t0+self.batch_size} (Now loaded: {num_audio})")
                    self.get_data_lock(self.lock["data"])
                    num_audio = len(self.audio_tensors)
                    while num_audio-t0 >= self.batch_size:
                        batched_input = torch.nn.utils.rnn.pad_sequence(self.audio_tensors[t0:t0+self.batch_size], batch_first=True)
                        batched_label = self.lid_labels[t0:t0+self.batch_size]
                        self.batched_inputs.append(batched_input)
                        self.batched_labels.append(batched_label)
                        t0 += self.batch_size
                        progress.update(1)
                    self.lock["data"].release()

                sleep(1)
            
            # All loading workers are done
            self.get_data_lock(self.lock["data"])
            num_audio = len(self.audio_tensors)
            self.lock["data"].release()
            while num_audio != self.n_files - self.num_unvoiced:
                print("crucial wait", num_audio, self.n_files, self.num_unvoiced)
                sleep(1)
                self.get_data_lock(self.lock["data"])
                num_audio = len(self.audio_tensors)
                self.lock["data"].release()
            while num_audio > t0:
                if num_audio-t0 >= self.batch_size:
                    batched_input = torch.nn.utils.rnn.pad_sequence(self.audio_tensors[t0:t0+self.batch_size], batch_first=True)
                    batched_label = self.lid_labels[t0:t0+self.batch_size]
                    self.batched_inputs.append(batched_input)
                    self.batched_labels.append(batched_label)
                    t0 += self.batch_size
                else: # The last batch
                    batched_input = torch.nn.utils.rnn.pad_sequence(self.audio_tensors[t0:num_audio], batch_first=True)
                    batched_label = self.lid_labels[t0:num_audio]
                    self.batched_inputs.append(batched_input)
                    self.batched_labels.append(batched_label)
                    t0 = num_audio
                progress.update(1)
            
            self.loading_time = progress.format_dict['elapsed']
            
            for trd in self.threads:
                trd.join()
            
            self.grouper_left = True
            # print("Grouper left.")
        
        signal.signal(signal.SIGINT, int_handler)

        self.n_workers = min(self.n_workers, len(files))
        for i in range(self.n_workers):
            trd = threading.Thread(
                target = load_worker_func, 
                args = (int(len(files)/self.n_workers*i), int(len(files)/self.n_workers*(i+1)))
            )
            trd.start()
            self.threads.append(trd)
        
        control_trd = threading.Thread(
            target=group_worker_func,
            args=()
        )
        control_trd.start()
    
    def num_worker_left(self):
        self.lock["worker_cnt"].acquire()
        n = self.n_workers
        self.lock["worker_cnt"].release()
        return n
    
    def get_data(self):
        self.get_data_lock(self.lock["data"])
        last_batched_idx = self.last_batched_idx
        num_batched_data = len(self.batched_inputs)
        self.lock["data"].release()

        X, y = None, None
        l, r = None, None

        if last_batched_idx < num_batched_data:
            l = last_batched_idx
            r = num_batched_data-1
            self.get_data_lock(self.lock["data"])
            X = self.batched_inputs[last_batched_idx: num_batched_data]
            y = self.batched_labels[last_batched_idx: num_batched_data]
            # print(f"Now using batch id: {last_batched_idx} to {num_batched_data}")
            self.last_batched_idx = num_batched_data
            self.lock["data"].release()
        
        elif self.num_worker_left() == 0 and self.grouper_left and last_batched_idx == num_batched_data:
            self.no_more_data = True
        
        return X, y, l, r
    
    def free_data(self, l, r):
        # print(len(self.audio_tensors), len(self.lid_labels))
        self.get_data_lock(self.lock["data"])
        for i in range(l, r+1):
            self.batched_inputs[i] = None
            self.batched_labels[i] = None
            for j in range(l*self.batch_size, (r+1)*self.batch_size):
                if j < len(self.audio_tensors) and j < len(self.lid_labels):
                    self.audio_tensors[j] = None
                    self.lid_labels[j] = None
        self.lock["data"].release()
