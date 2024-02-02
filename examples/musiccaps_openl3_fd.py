# navigate up one directory to get to stable-audio-metrics
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# import packages
from src.openl3_fd import openl3_fd

"""
Compute the Fréchet Distance between files in eval_path and ref_path.

Fréchet distance computed on top of openl3 embeddings.

GPU-based computation. Run: CUDA_VISIBLE_DEVICES=0 python examples_openl3_fd.py
Important: requires cuda 11.8 (e.g.: module load cuda/11.8)

openl3_fd params:
-- channels: 1 (mono), or 2 (stereo) to get the Fréchet Distance over mono or stereo embeddings.
-- samplingrate: max bandwith at wich we evaluate the given signals. Up to 48kHz.
-- content_type: 'music' or 'env' to select a content type for openl3.
-- openl3_hop_size: analysis resolution of openl3 in seconds. Openl3's input window is 1 sec.
-- eval_path: path where the generated audio files to evaluate are available.
-- eval_files_extenstion: files extension (default .wav) in eval_path.
-- ref_path: path where the reference audio files are available. (instead of load_ref_embeddings)
-- ref_files_extension: files extension (default .wav) in ref_path.
-- load_ref_embeddings: path to the reference embedding statistics. (inestead of ref_path)
-- batching: set batch size (with an int) or set to False (default False).
Returns:
-- Fréchet distance.
"""

reference_path = 'musiccaps_folder'
evaluation_path = 'your_model_outputs_folder'
model_channels = 2 # 1 or 2 channels
model_sr = 44100 # maximum bandwidth at which we evaluate, up to 48kHz
type = 'music' # openl3 model trained on 'music' or 'env'
hop = 0.5 # openl3 hop_size in seconds (openl3 window is 1 sec)
batch = 16

fd = openl3_fd(
    channels=model_channels,
    samplingrate=model_sr,
    content_type=type,
    openl3_hop_size=hop,
    eval_path=evaluation_path,
    eval_files_extension='.flac',
    ref_path=reference_path,
    ref_files_extension='.wav',
    batching=batch
)

file_path = (
    'MusicCaps' +
    '_vs_YourModel' +
    '__ch' + str(model_channels) +
    '__sr' + str(model_sr) +
    '__type' + str(type) +
    '__hop' + str(hop) +
    '__batch' + str(batch) +    
    '.txt'
)

with open(file_path, 'w') as file:
    file.write(file_path + '\n')
    file.write('Fréchet distance: ' + str(fd) + '\n')

"""
Extracting the embeddings is timeconsuming. After being computed once, we store them.
We store pre-computed reference embedding statistics in load/openl3_fd/ 
To load those and save computation, just set the path in load_ref_embeddings.
If load_ref_embeddings is set, ref_path is not required.

load/openl3_fd/stable-audio__musiccaps-public__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz
These reference embeddings are already available and were used to compute Stable Audio's metrics.
"""

load_ref_embeddings = 'load/openl3_fd/stable-audio__musiccaps-public__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz'
evaluation_path = 'your_model_outputs_folder'
model_channels = 2 # 1 or 2 channels
model_sr = 44100 # maximum bandwidth at which we evaluate, up to 48kHz
type = 'music' # openl3 model trained on 'music' or 'env'
hop = 0.5 # openl3 hop_size in seconds (openl3 window is 1 sec)
batch = 4

fd = openl3_fd(
    channels=model_channels,
    samplingrate=model_sr,
    content_type=type,
    openl3_hop_size=hop,
    eval_path=evaluation_path,
    eval_files_extension='.flac',
    load_ref_embeddings=load_ref_embeddings,
    batching=batch
)

file_path = (
    'MusicCapsLoaded' +
    '_vs_YourModel' +
    '__ch' + str(model_channels) +
    '__sr' + str(model_sr) +
    '__type' + str(type) +
    '__hop' + str(hop) +
    '__batch' + str(batch) +    
    '.txt'
)

with open(file_path, 'w') as file:
    file.write(file_path + '\n')
    file.write('Fréchet distance: ' + str(fd) + '\n')
