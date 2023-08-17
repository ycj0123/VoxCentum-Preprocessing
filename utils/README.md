# VoxCentum_Preprocess

## Pipeline Overview
(0). Make use of the our LID pipeline to test the LID accuracy on different channels by sampling some audio data of each channel. If LID accuracy is too low, there might be some problem in that channel.  

(1). Select 200~300 hours of audio data in each language（by metadata）  
* Record the audio data we have used.  
* There are 5 channels in each language.  
* Select 300 hours from each language.  
* Format：wav file、sampling rate = 16k, audio bit rate = 16, channel_num = 2.

(2). Do VAD (Voice Activity Detection)  
* The ratio of Hours of post-VAD audio data to Hours of pre-VAD audio data is 65%~80%.  
* After doing VAD, the hours of audio data is 196-hr ~ 250-hr.

(3). Split the audio data preprocessed by VAD into segments of fixed length. In our setting, the length is 10 sec.  

(4). Extract feature offline from our preprocessed audio data, e.g. Mel-spectrogram.  

(5). Train a simple x-vector based model.  

(6). On the language-level, filtering 110-hr audio data by the loss computed by our x-vector model.  

(7). Random split the filtered data into train/evaluation/test set, e.g. 100-hr/5-hr/5-hr.  

## Details

### Stage 1.

(0). Step0: 14 Languages.  

(1). Step1: Randomly select 300 hours of audio data from the crawled dataset for each language.  

(2). Step2: Standardize the audio formats and store them in:  
/home/meta-531-216/mount-4/preprocessed_16000/wav_fileter-list/vox100_stage1  

(3). Step3: Perform Voice Activity Detection (VAD).  

(4). Step4: Randomly extract 180 hours of data from the VAD results and split them into 10-second segments.  

(5). Step5: Store the data at:   
/home/meta-531-216/mount-4/VoxCentum_stage1  

* Note: If the data duration is insufficient, use all available data.  

### Stage 2.

(0). Step0: 63 Languages + Indigenous Languages (16) + Taiwanese + Hakka.  

(1). Step1: Utilize LID Pipeline to pre-filter low-resource languages.  

(2). Step2: Randomly select 300 hours of audio data from the crawled dataset for each language.  

(3). Step3: Standardize the audio formats and store them in:  

/home/meta-531-216/mount-4/preprocessed_16000/wav_fileter-list/vox100_stage2  
/home/meta-531-216/mount-4/preprocessed_16000/wav_fileter-list/vox100_stage2_abor  

(4). Step4: Perform Voice Activity Detection (VAD).  

(5). Step5: Randomly extract 180 hours of data from the VAD results and split them into 10-second segments.  

(6). Step6: Store the data at:  
/home/meta-531-216/mount-4/VoxCentum_stage2  

(7). Step7: For Indigenous languages, apply LID pipeline for filtering; if predicted as "zh," exclude them.  

* Note: If the data duration is insufficient, use all available data.  

### Stage 3.

(0). Step0: 43 Languages.  

(1). Step1: Utilize LID Pipeline to pre-filter low-resource languages.  

(2). Step2: Randomly select 300 hours of audio data from the crawled dataset for each language.  

(3). Step3: Standardize the audio formats and store them in:  
/home/meta-531-216/mount-4/preprocessed_16000/wav_fileter-list/vox100_stage3  

(4). Step4: Perform Voice Activity Detection (VAD).  

(5). Step5: Randomly extract 180 hours of data from the VAD results and split them into 10-second segments.  

(6). Step6: Store the data at:  
/home/meta-531-216/mount-4/VoxCentum_stage3  

(7). Step7: For "haw", “mi”, “gd”, “ga”, “ba”, “tt” languages, apply LID pipeline for filtering. If predicted as "en", “en”, “en”, “en”, “ru”, “ru” respectively, exclude them.  

* Note: If the data duration is insufficient, use all available data.  

### Stereo-to-Mono.

(0). Due to the limitations in audio data transfer, we have converted our stereo audio data into mono audio data.  

(1). Stored in:  
/home/meta-531-216/mount-4/VoxCentum_stage{index}1channel  
(index = 1, 2, 3)  

## Code
### Utils

(1) stat/cal_vad_time.py  
Calculating and recording the raw duration and post-VAD duration for various channels in different languages.  

(2) change_1channel.py  
Converting audio files to mono channel and a sample rate of 16k.  

(3) check_lang_channel.py  
Calculating the original duration for each channel.  

(4) data_filter_manual.py  
Making filtered data list for each language, each language has about 300 hours.  

(5) mp_construct_metadata.py  
Creating data statistics for each channel in different languages across all videos.  

(6) split_audio.py  
Segmenting each audio based on the filtered data list after VAD into 10-second segments.  
