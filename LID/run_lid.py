import argparse
import os

from modules.utils import str2bool
from modules.lid_task import LIDTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", type=str, help="Source directory")
    parser.add_argument("-w", "--workers", type=int, default=10, help="Number of workers")
    parser.add_argument("--vad", type=str2bool, default=False, help="Voice Activity Detection")
    parser.add_argument("--se", type=str2bool, default=False, help="Speech Enhancement")
    parser.add_argument("--multi_lang", type=str2bool, default=False)
    parser.add_argument("--vad_path", type=str, default=None, help="vad timestamps file path")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_sec", type=int, default=30)
    parser.add_argument("--max_trial", type=int, default=10)
    parser.add_argument("--ground_truth", type=str, default='en')
    parser.add_argument("--output_dir", type=str, default='output/')
    parser.add_argument("--audio_ext", type=str, default='.ogg')
    parser.add_argument("--lid_whisper_enable", action='store_true')
    parser.add_argument("--lid_voxlingua_enable", action='store_true')
    parser.add_argument("--lid_silero_enable", action='store_true')
    args = parser.parse_args()
    
    lid_task = LIDTask(args=args)
    lid_task.predict()
