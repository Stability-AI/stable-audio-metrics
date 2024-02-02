# stable-audio-metrics
Collection of metrics for evaluating generative audio models:
- Fréchet Distance, based on [Openl3](https://github.com/marl/openl3).
- Kullback–Leibler divergence, based on [PaSST](https://github.com/kkoutini/PaSST).
- CLAP score, based on [CLAP-LION](https://github.com/LAION-AI/CLAP).

## Installation 
Clone this repository, and create a python virtual environment `python3 -m venv env`, activate it `source env/bin/activate`, and install the dependencies `pip install -r requirements.txt`.

- ***TROUBLESHOOTING*** – It might require an older version of cuda because of Openl3 dependencies. Try cuda 11.8 if you find it does not run on GPU as expected.

## Usage

Modify our examples such that `generated_path` points to the folder you want to evaluate and run it. For example: `CUDA_VISIBLE_DEVICES=0 python examples/musiccaps_no-audio.py` or `CUDA_VISIBLE_DEVICES=6 python examples/audiocaps_no-audio.py`. 
- Each example script (`examples/musiccaps_openl3_fd.py`, `examples/musiccaps_passt_kld.py`, and `example/musiccapss_clap_score.py`) details how to use it.
- Additional documentation is available in: `src/openl3_fd.py`, `src/passt_kld.py`, and `src/clap_score.py`.
- We only support GPU usage, because it can be too slow on CPU.
- ***IMPORTANT*** – The `no-audioaudio` examples allow running the evaluations without downloading the datasets, because reference statistics and embeddings are already computed in `load`. Check the [example's documentation](examples/README.md).
- ***COMPARING w/ STABLE AUDIO*** – The pre-computed statistics and embeddings allows comparing against Stable Audio (with the exact same conditions) without the need to download the audio. Further, you don't need to download each datasets' text prompts since they are also available in the `load` folder. We do not provide any pre-computed embedding for the CLAP score, because is fast to compute. To compare against Stable Audio, you must set all parameters as in the examples. Even if your model outputs mono audio at a different sampling rate. `stable-audio-metrics` will do the resampling and mono/stereo handling to deliver a fair comparison.

## Data structure
Generate an audio for every prompt in each dataset, and name each generated audio by its corresponding id. 

Our musiccaps examples assume the following structure, where 5,521 generations are named after the `ytid` from the prompts file `load/musiccaps-public.csv`: `your_model_outputsfolder/-kssA-FOzU.wav`,'`your_model_outputs_folder/_0-2meOf9qY.wav`, ... `your_model_outputs_folder/ZzyWbehtt0M.wav`.

Our audiocaps examples assume the following structure, where 4,875 generations are named after the `audiocap_id` from the prompts file `load/audiocaps-test.csv`:
`your_model_outputsfolder/3.wav`, `your_model_outputs_folder/481.wav`, ... `your_model_outputs_folder/107432.wav`.
