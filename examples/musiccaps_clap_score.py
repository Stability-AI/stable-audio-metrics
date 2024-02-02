# Navigate up one directory to get to stable-audio-metrics
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import packages
import pandas as pd
from src.clap_score import clap_score

"""
The CLAP_score runs the cosine similarity between the LAION-CLAP text embedding of the given prompt and 
the LAION-CLAP audio embedding of the generated audio. LION-CLAP: https://github.com/LAION-AI/CLAP

This evaluation script assumes that audio_path files are identified with the ids in id2text.

clap_score() evaluates all ids in id2text.

GPU-based computation. Run: CUDA_VISIBLE_DEVICES=0 python examples_clap_score.py

Params:
-- id2text: dictionary with the mapping between id (generated audio filenames in audio_path) 
            and text (prompt used to generate audio). clap_score() evaluates all ids in id2text.
-- audio_path: path where the generated audio files to evaluate are available.
-- audio_files_extension: files extension (default .wav) in eval_path.
-- clap_model: choose one of the above clap_models (default: '630k-audioset-fusion-best.pt').
Returns:
-- CLAP-LION score
"""

csv_file_path = 'load/musiccaps-public.csv'
df = pd.read_csv(csv_file_path)
id2text = df.set_index('ytid')['caption'].to_dict()

generated_path = 'your_model_outputs_folder'

"""
IMPORTANT: the audios in generated_path should have the same ids as in id2text.
For musiccaps, you can load id2text as above and each generated_path audio file
corresponds to a prompt (text description) in musiccaps. Files are named with ids, as follows:
- your_model_outputs_folder/_-kssA-FOzU.wav
- your_model_outputs_folder/_0-2meOf9qY.wav
- your_model_outputs_folder/_1woPC5HWSg.wav
...
- your_model_outputs_folder/ZzyWbehtt0M.wav
"""

clp = clap_score(id2text, generated_path, audio_files_extension='.wav')
print('CLAP score (630k-audioset-fusion-best.pt): ', clp)

"""
By default, the clap_model is '630k-audioset-fusion-best.pt' (with fusion, to handle longer inputs).

But you can select any of the following models:
    - music_speech_audioset_epoch_15_esc_89.98.pt (used by musicgen)
    - music_audioset_epoch_15_esc_90.14.pt
    - music_speech_epoch_15_esc_89.25.pt
    - 630k-audioset-fusion-best.pt (our default, with "fusion" to handle longer inputs)

To know more about those models see: https://github.com/LAION-AI/CLAP
"""
clp = clap_score(id2text, generated_path, audio_files_extension='.wav', clap_model='music_speech_audioset_epoch_15_esc_89.98.pt')
print('CLAP score (music_speech_audioset_epoch_15_esc_89.98.pt): ', clp)

