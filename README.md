# stable-audio-metrics
Collection of metrics for evaluating generative audio models:
- Fréchet Distance, based on Openl3: https://github.com/marl/openl3
- Kullback–Leibler divergence, based on PaSST: https://github.com/kkoutini/PaSST
- CLAP score, based on CLAP-LION: https://github.com/LAION-AI/CLAP

## Installation 
Clone this repository: `git clone https://github.com/jordipons/stable-audio-quality.git`.

Create a python virtual environment `python3 -m venv env`, activate it `source env/bin/activate`, and install the dependencies `pip install -r requirements.txt`.

**TROUBLESHOOTING**: `stable-audio-metrics` might require an older version of cuda because of Openl3 dependencies. Try cuda 11.8 if you find it does not run on GPU as expected. Slurm users might want to run `module load cuda/11.8`.

## Usage
We only support GPU usage, because it can be too slow on CPU. 

Run any of the examples included in the repository, for example: `CUDA_VISIBLE_DEVICES=0 python examples/musiccaps_openl3_fd.py`.

Each example script (`examples/musiccaps_openl3_fd.py`, `examples/musiccaps_passt_kld.py`, and `example/musiccapss_clap_score.py`) details how to use it.

Additional documentation is available in: `src/openl3_fd.py`, `src/passt_kld.py`, and `clap_score.py`.

The code is readable and includes comments, and we encourage users to incorporate this in their codebase and understand/adapt those metrics for their use case.

## Comparing with Stable Audio

This repository allows to easily compare the results of your generative model against Stable Audio.

### MusicCaps evaluation

`stable-audio-metrics` were used to evaluate Stable Audio model witn MusicCaps.

**IMPORTANT**: The pre-computed statistics and probabilities allows comparing against Stable Audio (with the exact same conditions) without the need to download MusicCaps audio. Further, you don't need to download the MusicCaps text/prompts/captions. They are available at `load/musiccaps-public.csv`.

**IMPORTANT**: If you want to compare against Stable Audio, you must set all parameters as below. Even if your model outputs mono audio at a different sampling rate. `stable-audio-metrics` will do the resampling and mono/stereo handling to deliver a fair comparison against Stable Audio.

At the time of downloading MusicCaps, 5434 out of 5521 audios were available. The list of audios that were not availalbe are listed in `examples/musiccaps_passt_kld.py`.

**DATA STRUCTURE**: Generate an audio for every prompt in MusicCaps (5521 audios), and name each audio by its MusicCaps id. `stable-audio-metrics` assumes this structure:
- your_model_outputsfolder/-kssA-FOzU.wav
- your_model_outputs_folder/_0-2meOf9qY.wav
- your_model_outputs_folder/_1woPC5HWSg.wav
- ...
- your_model_outputs_folder/ZzyWbehtt0M.wav

Pre-computed reference statistics for **Fréchet Distance** are in `load/openl3_fd`. To compare against Stable Audio, use the following setup:
```python
load_ref_embeddings = 'load/openl3_fd/musiccaps__channels2__44100__openl3music__openl3hopsize0.5__batch16__reference_statistics.npz'

fd = openl3_fd(
    channels=2, # 2 channels as in Stable Audio
    samplingrate=44100, # Stable Audio generates 44.1kHz audio
    content_type='music' # openl3 model trained on 'music' for MusicCaps evaluation
    openl3_hop_size=0.5, # openl3 hop_size in seconds (openl3 window is 1 sec), set to 0.5 sec for efficency 
    eval_path='your_model_outputs_folder',
    eval_files_extension='.flac', # modify this to the audio format of your generated data
    load_ref_embeddings=load_ref_embeddings,
    batching=16,
)
```

Pre-computed audio embeddings for **Kullback-Leibler divergence** are in `load/passt_kld/musiccaps__collectmean__reference_probabilities.pkl`. To compare against Stable Audio, follow the example in `examples/musiccaps_passt_kld.py`. Make sure you set the `no_ids` parameter with the `NOT_IN_MUSICCAPS` ids in `examples/musiccaps_passt_kld.py` to exclude them from the evaluation. Note that we set window_size=10 (as PaSST training) and overlap=5 (overlap analysis window, every 5 sec, for efficiency), both default values in `src/passt_kld.py`.

We do not provide any pre-computed embedding for the **CLAP score**, because are fast to compute. Make sure you use `630k-audioset-fusion-best.pt` (the default model in `src/clap_score.py`). This "fusion" model allows handling longer inputs than 10 seconds, because Stable Audio generates 90 sec outputs.

# Notes for Jordi
- loudness normalization improves metrics?
- run long stuff
- check warnings and take them out.
- try installation from scratch

FORMER VERSIONS: 
torch==1.7.1
torchaudio==0.7.2
torchvision==0.8.2

CURRENT VERSIONS:
torch==2.0.0
torchaudio==2.0.1
torchvision==0.15.1
