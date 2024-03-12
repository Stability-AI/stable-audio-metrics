# navigate up one directory to get to stable-audio-metrics
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# import packages
import pandas as pd
from src.clap_score import clap_score
from src.passt_kld import passt_kld
from src.openl3_fd import openl3_fd

# the audio in generated_path should be named according to the 'caption_id' in the csv_file_path below
generated_path = 'your_model_outputs_folder' # path with the audio to evaluate
csv_file_path = 'load/song_describer.csv' # file with ids and prompts correspondences

# path with the recorded/reference/ground truth audio for FDopenl3 and KLpasst
reference_path_trackid = 'path_to_songdescriber_folder_with_track_id'
reference_path_captionid = 'path_to_songdescriber_folder_with_caption_id'
# note that, differently from musiccaps, for song describer dataset we need two reference_paths because it contains multiple captiosn per audio in the dataset
# for KL our assumed data structure is an audiofile per caption_id unique caption (1106 caption_id) 
# for FD our assumed data structure is an audiofile per track_id unique audio (706 track_id)
# - the caption_id path contains as many (copied) music files as captions it exists, each representing a prompt and named with caption_ids
# - the track_id path contains the music files named by track_ids
# caption_id folder can be symlinks to the track_id folder


print('Computing CLAP score..')
# in this songdescriber case here, our audios are named with the caption_ids in csv_file_path
df = pd.read_csv(csv_file_path)
# create a dictionary to get the text prompt (used to generate audio) given the caption_id (audio file name)
id2text = df.set_index('caption_id')['caption'].to_dict()
# compute clap score from the id2text (prompts) and generated_path (audio)
clp = clap_score(id2text, generated_path, audio_files_extension='.wav')
print('[songdescriber] CLAP score (630k-audioset-fusion-best.pt): ', clp, generated_path)


print('Computing KLpasst..')
# list all ids that are in both ref_path (reference audio) and eval_path (generated audio)
# in this songdescriber case here, our audios are named with the caption_ids in csv_file_path
sdd_ids = df['caption_id'].tolist()
# compute KLpasst between ref_path (reference audio) and eval_path (generated audio)
kl = passt_kld(ids=sdd_ids, 
              eval_path=generated_path, 
              eval_files_extension='.wav',
              ref_path=reference_path_captionid,
              ref_files_extension='.mp3', # song describer dataset is in .mp3
              collect='mean')
print('[songdescriber] KLpasst: ', kl, generated_path)


print('Computing FDopenl3..')
model_channels = 2 # 1 or 2 channels
model_sr = 44100 # maximum bandwidth at which we evaluate, up to 48kHz
type = 'music' # openl3 model trained on 'music' or 'env'
hop = 0.5 # openl3 hop_size in seconds (openl3 window is 1 sec)
batch = 4
# compute the FDopenl3 given the parameters above
fd = openl3_fd(
    channels=model_channels,
    samplingrate=model_sr,
    content_type=type,
    openl3_hop_size=hop,
    eval_path=generated_path,
    eval_files_extension='.wav',
    ref_path=reference_path_trackid,
    ref_files_extension='.mp3',
    batching=batch
)


# print all the results
print('\n\n[songdescriber] CLAP score: ', clp, generated_path)
print('[songdescriber] KLpasst: ', kl, generated_path)
print('[songdescriber] FDopenl3: ', fd, generated_path)
