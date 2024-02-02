# navigate up one directory to get to stable-audio-metrics
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# import packages
import pandas as pd
from src.passt_kld import passt_kld

"""
Compute KL-divergence between the label probabilities of the generated audio with respect to the original audio.
Both generated audio (in eval_path) and original audio (in ref_path) are represented by the same prompt/description.
Audios are identified by an id, that is the name of the file in both directories and links the audio with the prompt/description.

For inputs longer that the 10 sec PaSST was trained on, we aggregate/collect via 'mean' (default) or 'max' pooling along the logits vector.
We split the inpot into overlapping analysis windows. Subsequently, we aggregate/collect (accross windows) the generated logits and then apply a softmax. 

This evaluation script assumes that ids are in both ref_path and eval_path.

We label probabilities via the PaSST model: https://github.com/kkoutini/PaSST

GPU-based computation. Run: CUDA_VISIBLE_DEVICES=0 python examples_passt_kld.py

Params:
-- ids: list of ids present in both eval_path and ref_path. 
-- eval_path: path where the generated audio files to evaluate are available.
-- eval_files_extenstion: files extension (default .wav) in eval_path.
-- ref_path: path where the reference audio files are available. (instead of load_ref_probabilities)
-- ref_files_extenstion: files extension (default .wav) in ref_path.
-- load_ref_probabilities: path to the reference probabilities. (inestead of ref_path)
-- no_ids: it is possible that some reference audio is corrupted or not present. Ignore some this list of ids.
-- collect (default='mean'): for longer inputs, aggregate/collect via 'mean' or 'max' pooling along logits vector.
Returns:
-- KL divergence
"""

# these are the musiccaps ids that we could not download from Youtube – we ignore them for our evaluation
# at the time of downloading musiccaps, 5434 out of 5521 audios were available, this is the list of audios that were not available:
NOT_IN_MUSICCAPS = ['NXuB3ZEpM5U', 'C7OIuhWSbjU', 'Rqv5fu1PCXA', 'WvEtOYCShfM', '25Ccp8usBtE', 'idVCpQaByc4', 'tpamd6BKYU4', 'bpwO0lbJPF4', 'We0WIPYrtRE', 'kiu-40_T5nY', '5Y_mT93tkvQ', 'zCrpaLEq1VQ', '8olHAhUKkuk', '6xxu6f0f0e4', 'B7iRvj8y9aU', 'rrAIuGMTqtA', 'UdA6I_tXVHE', 'm-e3w2yZ6sM', 'Xy7KtmHMQzU', 'd6-bQMCz7j0', 'BeFzozm_H5M', 't5fW1-6iXZY', 'jd1IS7N3u0I', '_hYBs0xee9Y', 'EhFWLbNBOxc', '63rqIYPHvlc', 'Jk2mvFrdZTU', 'IbJh1xeBFcI', 'HAHn_zB47ig', 'j9hAUlz5kQs', 'Vu7ZUUl4VPc', 'asYb6iDz_kM', 'fZyq2pM2-dI', 'vOAXAoHtl7o', 'go_7i6WvfeE', 'iXgEQj1Fs7g', 'dcY062mkf9g', '_ACyiKGpD8Y', '_DHMdtRRJzE', 'zSSIGv82318', '2dyxjGTXSpA', '7WZwlOrRELI', 'g8USMvt9np0', '374R7te0ra0', 'CCFYOw8keiI', 'eHeUipPZHIc', '0J_2K1Gvruk', 'MYtq46rNsCA', 'NIcsJ8sEd0M', '8vFJX7NcSbI', 'TkclVqlyKx4', 'T6iv9GFIVyU', 'ChqJYrmQIN4', 'ZZrvO__SNtA', 'fwXh_lMOqu0', '0khKvVDyYV4', '-sevczF5etI', 'qc1DaM4kdO0', 'wBe5tW8iJew', 'vQHKa69Mkzo', 'Fv9swdLA-lo', 'Ah_aYOGnQ_I', 'nTtxF9Wyw6o', '7B1OAtD_VIA', 'OS4YFp3DiEE', 'lTLsL94ABRs', 'jmPmqzxlOTY', 'k-LkhT4HAiE', 'Hvs6Xwc6-gc', 'xxCnmao8FAs', 'BiQik0xsWxk', 'L5Uu_0xEZg4', 'cADT8fUucLQ', 'ed-1zAOr9PQ', 'zSq2D_GF00o', 'gdtw54I8soM', 'lrk00BNiuD4', 'RQ0-sjpAPKU', 'SLq-Co_szYo', '0fqtA_ZBn_8', 'Xoke1wUwEXY', 'LRfVQsnaVQE', 'p_-lKpxLK3g', 'AaUZb-iRStE', '0pewITE1550', 'JNw0A8pRnsQ', 'vVNWjq9byoQ']

# we use musiccaps' ytid, and audio files with those ids are in both ref_path and eval_path.
csv_file_path = 'load/musiccaps-public.csv'
df = pd.read_csv(csv_file_path)
musiccaps_ids = df['ytid'].tolist()

kl = passt_kld(ids=musiccaps_ids, 
              eval_path='your_model_outputs_folder', 
              ref_path='musiccaps_folder', 
              no_ids=NOT_IN_MUSICCAPS,
              collect='mean')

print('KLpasst of YourModel', kl)

"""
Extracting the probabilities is timeconsuming. After being computed once, we store them.
We store pre-computed reference probabilities in load/ 
To load those and save computation, just set the path in load_ref_probabilities.
If load_ref_probabilities is set, ref_path is not required.

load/passt_kld/stable-audio__musiccaps-public__collectmean__reference_probabilities.pkl
These reference probabilities are already available and were used to compute Stable Audio's metrics.
"""

kl = passt_kld(ids=musiccaps_ids, 
              eval_path='your_model_outputs_folder', 
              load_ref_probabilities='load/passt_kld/stable-audio__musiccaps-public__collectmean__reference_probabilities.pkl', 
              no_ids=NOT_IN_MUSICCAPS,
              collect='mean')

print('KLpasst of YourModelLoaded', kl)
