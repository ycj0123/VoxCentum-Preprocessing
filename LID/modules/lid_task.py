import ast, nlp2, os, time
from tqdm import tqdm
from math import ceil

from modules.code2label import code2label
from modules.multi_thread_loader import MultiThreadLoader
from modules.lid_enhancement import AudioLIDEnhancer
from modules.log_results import gen_pred_results
from modules.code2label import code2label

# given all the arguments, will predict the language id of all the audio
# and log to a result file
class LIDTask():
    def __init__(self, args):
        self.config = vars(args)
        self.source_dir = args.source_dir
        self.workers = args.workers
        self.vad = args.vad
        self.se = args.se
        self.multi_lang = args.multi_lang
        self.vad_path = args.vad_path
        self.batch_size = args.batch_size
        self.chunk_sec = args.chunk_sec
        self.max_trial = args.max_trial
        self.ground_truth = args.ground_truth if not self.multi_lang else None
        self.output_dir = args.output_dir
        self.audio_ext = args.audio_ext
        self.lid_whisper_enable = args.lid_whisper_enable
        self.lid_silero_enable = args.lid_silero_enable
        self.lid_voxlingua_enable = args.lid_voxlingua_enable
        self.lid_model = AudioLIDEnhancer(
            device='cpu', 
            lid_voxlingua_enable=self.lid_voxlingua_enable,
            lid_silero_enable=self.lid_silero_enable, 
            lid_whisper_enable=self.lid_whisper_enable, 
            enable_enhancement=self.se, 
            voice_activity_detection=self.vad
        )

        if self.multi_lang is False and self.ground_truth not in code2label.keys():
            print(self.ground_truth, "is not in our lang list")
            print(f"Language {self.ground_truth} not found")
            raise NotImplementedError

        if self.vad == True:
            with open(self.vad_path, 'r') as f:
                vad_time_stamp_dict = eval(f.readline())

            print("With using VAD")
            self.vad_path = ast.literal_eval(open(self.vad_path, "r").readline())
            self.vad_path = {k.split('/')[-1]: v for k, v in self.vad_path.items()}
        else:
            vad_time_stamp_dict = dict()
            print("Without using VAD")
    
    def predict(self):
        lid_task_list = []

        if self.multi_lang:
            for ground_truth in os.listdir(self.source_dir):
                lang_dir = os.path.join(self.source_dir, ground_truth)
                if os.path.isdir(lang_dir):
                    for channel in os.listdir(lang_dir):
                        channel_dir = os.path.join(lang_dir, channel)
                        if os.path.isdir(channel_dir):
                            lid_task_list.append((channel_dir, ground_truth))
        else:
            lid_task_list.append((self.source_dir, self.ground_truth))

        for source_dir, ground_truth in lid_task_list:

            audio_filenames = []
            for i in nlp2.get_files_from_dir(source_dir, match=self.audio_ext): 
                try:
                    audio_filenames.append(i)
                except:
                    pass
            print(f"now predicting {source_dir} (ground truth: {ground_truth})")
            print("Number of audio file:", len(audio_filenames))

            loader = MultiThreadLoader(
                n_workers = self.workers, 
                batch_size = self.batch_size, 
                n_files = len(audio_filenames),
                max_trial = self.max_trial,
                chunk_sec = self.chunk_sec,
                vad_path = self.vad_path,
                ground_truth = ground_truth,
                use_vad = self.vad
            )
            loader.start(files=audio_filenames)
            ready_data_idx = 0
            
            preds = []
            labels = []
            possible_langs = list(code2label.keys())
            skip_audio_list = []

            progress = tqdm(total=ceil(len(audio_filenames) / self.batch_size), desc="PRED", position=1)
            while True:
                X, y, l, r = loader.get_data() # X is list of batched data

                if loader.no_more_data:
                    break

                elif X is not None:
                    for i in range(len(X)):
                        try:
                            pred = self.lid_model.seq_forward(X[i], y[i])
                        except RuntimeError:
                            exit(2)
                        preds.extend([code2label[code] for code in pred])
                        labels.extend(y[i])
                        progress.update(1)

                    loader.free_data(l, r)

                if not loader.no_more_data:
                    time.sleep(0.5)
            
            channel_name = os.path.basename(os.path.normpath(source_dir))
            if self.vad:
                output_file_name = f"{channel_name}_vad_output.txt"
            else:
                output_file_name = f"{channel_name}_output.txt"

            print(preds)

            gen_pred_results(
                labels=labels,
                preds=preds,
                total_sec=loader.total_sec,
                config=self.config,
                audio_filenames=audio_filenames, 
                src_dir=source_dir, 
                loading_time=loader.loading_time,
                file_idx=loader.file_idx,
                num_unvoiced=loader.num_unvoiced,
                unvoiced_idx=loader.unvoiced_idx,
                predicting_time=progress.format_dict['elapsed'],
                output_file=output_file_name,
                output_dir=self.output_dir,
                output_lang=ground_truth,
                is_whisper=True if (code2label[ground_truth] in self.lid_model.whisper_langs and self.lid_model.lid_whisper_enable) else False,
                is_vox=True if (code2label[ground_truth] in self.lid_model.vox_langs and self.lid_model.lid_voxlingua_enable) else False,
                is_silero=True if (code2label[ground_truth] in self.lid_model.silero_langs and self.lid_model.lid_silero_enable) else False
            )