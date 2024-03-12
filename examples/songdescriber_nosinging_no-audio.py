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
csv_file_path = 'load/song_describer-nosinging.csv' # file with ids and prompts correspondences


print('Computing CLAP score..')
# in this songdescriber case here, our audios are named with the 'caption_id' in csv_file_path
df = pd.read_csv(csv_file_path)
# create a dictionary to get the text prompt (used to generate audio) given the 'caption_id' (audio file name)
id2text = df.set_index('caption_id')['caption'].to_dict()
# compute clap score from the id2text (prompts) and generated_path (audio)
clp = clap_score(id2text, generated_path, audio_files_extension='.wav')
print('[song describer dataset] CLAP score (630k-audioset-fusion-best.pt): ', clp, generated_path)


print('Computing KLpasst..')
# list all ids that are in both ref_path (reference audio that is loaded) and eval_path (generated audio)
# in this songdescriber case here, our audios are named with the 'caption_id' in csv_file_path
sdd_ids = df['caption_id'].tolist()
# compute KLpasst between ref_path (reference audio that is loaded) and eval_path (generated audio)
kl = passt_kld(ids=sdd_ids, 
              eval_path=generated_path, 
              load_ref_probabilities='load/passt_kld/stable-audio__song-describer-nosinging__collectmean__reference_probabilities.pkl', 
              collect='mean')
print('[song describer dataset] KLpasst: ', kl, generated_path)


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
    load_ref_embeddings='load/openl3_fd/stable-audio__song-describer-nosinging__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz',
    batching=batch
)


# print all the results
print('\n\n[song describer dataset] CLAP score: ', clp, generated_path)
print('[song describer dataset] KLpasst: ', kl, generated_path)
print('[song describer dataset] FDopenl3: ', fd, generated_path)
