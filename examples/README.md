# Examples documentation

Examples and documentation on how using each metric separately:
- `musiccaps_clap_score.py`: CLAP metric for musiccaps.
- `musiccaps_openl3_fd.py`: FDopenl3 metric for musiccaps.
- `musiccaps_passt_kld.py`: KLpasst metric for musiccaps.

Examples on how to evaluate on audiocaps and musiccaps:
- `audiocaps.py`: run CLAP, FDopenl3 and KLpasst metrics for audiocaps.
- `musiccaps.py`: run CLAP, FDopenl3 and KLpasst metrics for musiccaps.
- `musiccaps_nosinging.py`: run CLAP, FDopenl3 and KLpasst metrics for the musiccaps subset without singing voice (vocals) prompts.

For running the previous musiccaps and audiocaps scripts, one must download the audio from those datasets.

The following examples do not required downloading the audio. To facilitate evaluating with `stable-audio-metrics`, we already provide the reference probabilities and embeddings of the audio. As a result, one can directly evaluate with those scripts without the need to download the audio:
- `audiocaps_no-audio.py`: run CLAP, FDopenl3 and KLpasst metrics for audiocaps.
- `musiccaps_no-audio.py`: run CLAP, FDopenl3 and KLpasst metrics for musiccaps.
- `musiccaps_nosinging_no-audio.py`: run CLAP, FDopenl3 and KLpasst metrics for the musiccaps subset without singing voice (vocals) prompts.

These final examples would allow you to compare against Stable Audio.
- ***COMPARING w/ STABLE AUDIO*** – The `no-audio` examples allows comparing against Stable Audio (with the exact same conditions) without the need to download the audio. Further, you don't need to download each datasets' text prompts since they are also available in the `load` folder. We do not provide any pre-computed embedding for the CLAP score, because is fast to compute. To compare against Stable Audio, you must set all parameters as in the examples. Even if your model outputs mono audio at a different sampling rate. `stable-audio-metrics` will do the resampling and mono/stereo handling to deliver a fair comparison.
