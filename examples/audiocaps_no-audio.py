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

# the audio in generated_path should be named according to the csv_file_path below
generated_path = 'your_model_outputs_folder' # path with the audio to evaluate
csv_file_path = 'load/audiocaps-test.csv' # file with ids and prompts correspondences

# these are the audiocaps ids that we could not download from Youtube – we ignore them for KLpasst computation
# at the time of downloading audiocaps, 979 out of 881 audios were available, this is the list of audios that were not available:
NOT_IN_AUDIOCAPS = ['gQMTOKsCIyk', '9HVgYs8OOLc', 'WHRnyGXcdy8', 'hFCmq9pCBbM', 'D9GHUPGWsV0', 'fmEft49sPfE', 'OTLtzk0W4zg', 'ZNEZLlDVgrE', 'HUwXtfYRFwk', 'g0l8ArPOTvA', 'orgwzt45ojE', '1oOYqBroWoA', 'lYhwCRX2wNc', '31WGUPOYS5g', 'XIooZl1QdM4', 'vEWmHtiznF8', 'c0IggDOisOo', 'lTJLvSvjUZk', 'QoEal_hKz4Q', 'xnVqvc7N7Po', 'YqYCDis3EUA', 'PtW0cZVprJQ', 'Yk274Wr5iIE', '3n05BjV7r7M', 'NDaVSIJaXVs', 'Ep72tyiL3as', 'cK2kSVR1d2o', 't3VFlDiEKgY', 'jPayI-MTnag', 'cr0GiZr0TNY', '-mhFGevxLUg', 'QARuiRtfy-k', 'A2mcp0N__7U', 'qWYncqPSy9A', 'OmmPaIAXN0s', 'GSHcgY6ATkQ', '2GehEKSOgc8', 'S_3aeOvniZc', '3IguMJkqpl4', '2bq2lc3DLwM', '0UJtGcoPYa0', 'Q3vkJMVMbC8', 'hhSqQN1Ou68', '7-HCqJFwHoI', '5eSRL3PRHzo', 'bmEF-c-M174', 'zFzPOsOKog4', 'SNIaYhri76w', 'hpDltmawxIM', 'ram-QPKSQYc', 'zEM94PH29VQ', '_w2pA1VeB40', 'rgrmLLhxoCQ', '1OyEgzXCkYE', 'NtQiduPRiRg', '8BPTQO_cx7E', '4bUL_ttiOdw', 'pPLisQ_QXxw', 'NX0gR9Eebp0', 'yau2WIRkxb8', 'XP1L5k-Zxro', 'B-gTt3_rceQ', '2ItTq2JShdU', '7WkB6pflr6o', '77nElZGi5NU', '404cD3bVXDc', 'ITP7tMt1BDg', 'qPYwp1K4sZE', 'jso1tv-zG-E', 'PuLuZ_TXv-0', 'MSziND26UTA', 'eJCaRgf1M20', 'cN-oYKd-M4E', 'ETb9EIQOMAA', '4_Cak7gvly4', 'MOxddxW5PXs', 'Af4a-9rcnP0', 'BrPFQDr99Gg', '9XqkKuTqEOM', '5rh5-MCjqq8', 'nuZEAuAl8hQ', '3iLGu2Omgrw', 'vsy1IpYmrSY', 'q46VXJ6JN9M', 'XL8JV9qXGLE', 'Lh0UmwRgA7s', 'EQVWhHmT_cE', 'Z7yDwpdGelM', 'dvY_HUaRgW8', 'baThGFuiYys', 'awxrHOpt-sE', '3kBlVLkN0zo', 'mVjub3o_IxE', 'RNBoH2LHQEM']

# note that this script allows comparing against Stable Audio without downloading the dataset
# if all parameters are kept as in this file, the results are comparable with the ones in Stable Audio
# all ref_probabilities and ref_embeddings are already precomputed in load folder


print('Computing CLAP score..')
# in this audiocaps case here, our audios are named with the ytid in csv_file_path
df = pd.read_csv(csv_file_path)
# create a dictionary to get the text prompt (used to generate audio) given the ytid (audio file name)
id2text = df.set_index('audiocap_id')['caption'].to_dict()
# compute clap score from the id2text (prompts) and generated_path (audio)
clp = clap_score(id2text, generated_path, audio_files_extension='.wav')
print('[audiocaps] CLAP score (630k-audioset-fusion-best.pt): ', clp, generated_path)


print('Computing KLpasst..')
# list all ids that are in both ref_path (reference audio that is loaded) and eval_path (generated audio)
# in this audiocaps case here, our audios are named with the ytid in csv_file_path
audiocaps_ids = df['audiocap_id'].tolist()
# map youtube_ids to audiocaps_ids, because KL operates with audiocaps_ids instead of youtube_ids
ids_not_in_audiocaps = df[df['youtube_id'].isin(NOT_IN_AUDIOCAPS)]['audiocap_id'].tolist() # omit those
# compute KLpasst between ref_path (reference audio that is loaded) and eval_path (generated audio)
kl = passt_kld(ids=audiocaps_ids, 
              eval_path=generated_path, 
              load_ref_probabilities='load/passt_kld/audiocaps-test__collectmean__reference_probabilities.pkl', 
              no_ids=ids_not_in_audiocaps,
              collect='mean')
print('[audiocaps] KLpasst: ', kl, generated_path)


print('Computing FDopenl3..')
model_channels = 2 # 1 or 2 channels
model_sr = 44100 # maximum bandwidth at which we evaluate, up to 48kHz
type = 'env' # openl3 model trained on 'music' or 'env'
hop = 0.5 # openl3 hop_size in seconds (openl3 window is 1 sec)
batch = 4
# compute the FDopenl3 given the parameters above. Keep them as they are to compare with Stable Audio!
fd = openl3_fd(
    channels=model_channels,
    samplingrate=model_sr,
    content_type=type,
    openl3_hop_size=hop,
    eval_path=generated_path,
    eval_files_extension='.wav',
    load_ref_embeddings='load/openl3_fd/audiocaps-test__channels2__44100__openl3env__openl3hopsize0.5__batch4.npz',
    batching=batch
)


print('\n\n[audiocaps] CLAP score: ', clp, generated_path)
print('[audiocaps] KLpasst: ', kl, generated_path)
print('[audiocaps] FDopenl3: ', fd, generated_path)
