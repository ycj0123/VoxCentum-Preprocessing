from sox import file_info
import argparse
import glob
import os
from multiprocessing import Pool
from tqdm import tqdm

''' From tsu-yuan bro.'''
''' Converting audio files to mono channel and a sample rate of 16k. '''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/meta-531-216/mount-4/VoxCentum_stage3_ceb')

    parser.add_argument('--new_root', default='/home/meta-531-216/mount-4/VoxCentum_stage3_ceb_pcm16')
    parser.add_argument('--log', default='VoxCentum_stage3_1channel.log')
    
    args = parser.parse_args()
    return args

def get_wav_channel(file):
    return file_info.channels(file)

def main(args):
    root = args.root
    new_root = args.new_root
    log = args.log

    # files = glob.glob(f'{root}/@KatikAiraDenielle/*.webm')
    files = glob.glob(f'{root}/ceb/*.wav')
    print(files)
    commands = []
    count = {}
    for file in tqdm(files):
        new_file = file.replace(root, '')
        new_file = f'{new_root}{new_file}'
        
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        file_ch = get_wav_channel(file)
        count[file_ch] = count.get(file_ch, 0) + 1
        if file_ch == 1 or file_ch == 2:
            commands.append(f'ffmpeg -i {file} -ac 1 -ar 16000 {new_file}')
        else:
            with open(log, 'a') as f:
                f.write(f'[Wrong Channel Number]{file}\n')

    pool = Pool(8)
    n_batch = 100
    batch = len(commands) // n_batch + 1
    for i in tqdm(range(n_batch)):
        batch_commands = commands[batch * i : batch * (i + 1)]
        batch_results = pool.map(os.system, batch_commands)
        with open(log, 'a') as f:
            f.write(f'[Batch {i}] {batch_results}\n')

if __name__ == '__main__':
    args = get_args()
    main(args)
