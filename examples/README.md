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

To facilitate evaluating with stable-audio-metrics, we already provide the reference probabilities and embeddings of the audio. As a result, one can directly evaluate with those scripts without the need to download the audio:
- `audiocaps_no-audio.py`: run CLAP, FDopenl3 and KLpasst metrics for audiocaps.
- `musiccaps_no-audio.py`: run CLAP, FDopenl3 and KLpasst metrics for musiccaps.
- `musiccaps_nosinging_no-audio.py`: run CLAP, FDopenl3 and KLpasst metrics for the musiccaps subset without singing voice (vocals) prompts.