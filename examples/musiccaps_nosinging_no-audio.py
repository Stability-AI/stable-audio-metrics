# Navigate up one directory to get to stable-audio-metrics
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import packages
import pandas as pd
from src.clap_score import clap_score
from src.passt_kld import passt_kld
from src.openl3_fd import openl3_fd


# the audio in generated_path should be named according to the csv_file_path below
generated_path = '/fsx/jordipons/audio/nosinging/musiccaps-stableaudio/' # path with the audio to evaluate
csv_file_path = 'load/musiccaps-public-nosinging.csv' # file with ids and prompts correspondences

# these are the musiccaps ids that we could not download from Youtube – we ignore them for KLpasst computation
NOT_IN_MUSICCAPS = ['NXuB3ZEpM5U', 'C7OIuhWSbjU', 'Rqv5fu1PCXA', 'WvEtOYCShfM', '25Ccp8usBtE', 'idVCpQaByc4', 'tpamd6BKYU4', 'bpwO0lbJPF4', 'We0WIPYrtRE', 'kiu-40_T5nY', '5Y_mT93tkvQ', 'zCrpaLEq1VQ', '8olHAhUKkuk', '6xxu6f0f0e4', 'B7iRvj8y9aU', 'rrAIuGMTqtA', 'UdA6I_tXVHE', 'm-e3w2yZ6sM', 'Xy7KtmHMQzU', 'd6-bQMCz7j0', 'BeFzozm_H5M', 't5fW1-6iXZY', 'jd1IS7N3u0I', '_hYBs0xee9Y', 'EhFWLbNBOxc', '63rqIYPHvlc', 'Jk2mvFrdZTU', 'IbJh1xeBFcI', 'HAHn_zB47ig', 'j9hAUlz5kQs', 'Vu7ZUUl4VPc', 'asYb6iDz_kM', 'fZyq2pM2-dI', 'vOAXAoHtl7o', 'go_7i6WvfeE', 'iXgEQj1Fs7g', 'dcY062mkf9g', '_ACyiKGpD8Y', '_DHMdtRRJzE', 'zSSIGv82318', '2dyxjGTXSpA', '7WZwlOrRELI', 'g8USMvt9np0', '374R7te0ra0', 'CCFYOw8keiI', 'eHeUipPZHIc', '0J_2K1Gvruk', 'MYtq46rNsCA', 'NIcsJ8sEd0M', '8vFJX7NcSbI', 'TkclVqlyKx4', 'T6iv9GFIVyU', 'ChqJYrmQIN4', 'ZZrvO__SNtA', 'fwXh_lMOqu0', '0khKvVDyYV4', '-sevczF5etI', 'qc1DaM4kdO0', 'wBe5tW8iJew', 'vQHKa69Mkzo', 'Fv9swdLA-lo', 'Ah_aYOGnQ_I', 'nTtxF9Wyw6o', '7B1OAtD_VIA', 'OS4YFp3DiEE', 'lTLsL94ABRs', 'jmPmqzxlOTY', 'k-LkhT4HAiE', 'Hvs6Xwc6-gc', 'xxCnmao8FAs', 'BiQik0xsWxk', 'L5Uu_0xEZg4', 'cADT8fUucLQ', 'ed-1zAOr9PQ', 'zSq2D_GF00o', 'gdtw54I8soM', 'lrk00BNiuD4', 'RQ0-sjpAPKU', 'SLq-Co_szYo', '0fqtA_ZBn_8', 'Xoke1wUwEXY', 'LRfVQsnaVQE', 'p_-lKpxLK3g', 'AaUZb-iRStE', '0pewITE1550', 'JNw0A8pRnsQ', 'vVNWjq9byoQ']


print('Computing CLAP score..')
# in this musiccaps case here, our audios are named with the ytid in csv_file_path
df = pd.read_csv(csv_file_path)
# create a dictionary to get the text prompt (used to generate audio) given the ytid (audio file name)
id2text = df.set_index('ytid')['caption'].to_dict()
# compute clap score from the id2text (prompts) and generated_path (audio)
clp = clap_score(id2text, generated_path, audio_files_extension='.wav')
print('[musiccaps] CLAP score (630k-audioset-fusion-best.pt): ', clp, generated_path)


print('Computing KLpasst..')
# list all ids that are in both ref_path (reference audio) and eval_path (generated audio)
# in this musiccaps case here, our audios are named with the ytid in csv_file_path
df = pd.read_csv(csv_file_path)
musiccaps_ids = df['ytid'].tolist()
# compute KLpasst between ref_path (reference audio) and eval_path (generated audio)
kl = passt_kld(ids=musiccaps_ids, 
              eval_path=generated_path, 
              load_ref_probabilities='load/passt_kld/stable-audio__musiccaps-public-nosinging__collectmean__reference_probabilities.pkl', 
              no_ids=NOT_IN_MUSICCAPS,
              collect='mean')
print('[musiccaps] KLpasst: ', kl, generated_path)


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
    load_ref_embeddings='load/openl3_fd/stable-audio__musiccaps-public-nosinging__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz',
    batching=batch
)


print('\n\n[musiccaps] CLAP score: ', clp, generated_path)
print('[musiccaps] KLpasst: ', kl, generated_path)
print('[musiccaps] FDopenl3: ', fd, generated_path)
