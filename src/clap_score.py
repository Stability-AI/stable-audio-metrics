import os
import requests
from tqdm import tqdm
import torch
import numpy as np
import laion_clap
from clap_module.factory import load_state_dict
import librosa
import pyloudnorm as pyln

# following documentation from https://github.com/LAION-AI/CLAP
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def clap_score(id2text, audio_path, audio_files_extension='.wav', clap_model='630k-audioset-fusion-best.pt'):
    """
    Cosine similarity is computed between the LAION-CLAP text embedding of the given prompt and 
    the LAION-CLAP audio embedding of the generated audio. LION-CLAP: https://github.com/LAION-AI/CLAP
    
    This evaluation script assumes that audio_path files are identified with the ids in id2text.
    
    clap_score() evaluates all ids in id2text.

    GPU-based computation.

    Select one of the following models from https://github.com/LAION-AI/CLAP:
        - music_speech_audioset_epoch_15_esc_89.98.pt (used by musicgen)
        - music_audioset_epoch_15_esc_90.14.pt
        - music_speech_epoch_15_esc_89.25.pt
        - 630k-audioset-fusion-best.pt (our default, with "fusion" to handle longer inputs)

    Params:
    -- id2text: dictionary with the mapping between id (generated audio filenames in audio_path) 
                and text (prompt used to generate audio). clap_score() evaluates all ids in id2text.
    -- audio_path: path where the generated audio files to evaluate are available.
    -- audio_files_extension: files extension (default .wav) in eval_path.
    -- clap_model: choose one of the above clap_models (default: '630k-audioset-fusion-best.pt').
    Returns:
    -- CLAP-LION score
    """
    # load model
    if clap_model == 'music_speech_audioset_epoch_15_esc_89.98.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt'
        clap_path = 'load/clap_score/music_speech_audioset_epoch_15_esc_89.98.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
    elif clap_model == 'music_audioset_epoch_15_esc_90.14.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
        clap_path = 'load/clap_score/music_audioset_epoch_15_esc_90.14.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
    elif clap_model == 'music_speech_epoch_15_esc_89.25.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt'
        clap_path = 'load/clap_score/music_speech_epoch_15_esc_89.25.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
    elif clap_model == '630k-audioset-fusion-best.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt'
        clap_path = 'load/clap_score/630k-audioset-fusion-best.pt'
        model = laion_clap.CLAP_Module(enable_fusion=True, device='cuda')
    else:
        raise ValueError('clap_model not implemented')

    # download clap_model if not already downloaded
    if not os.path.exists(clap_path):
        print('Downloading ', clap_model, '...')
        os.makedirs(os.path.dirname(clap_path), exist_ok=True)

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(clap_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
                for data in response.iter_content(chunk_size=8192):
                    file.write(data)
                    progress_bar.update(len(data))

    # fixing CLAP-LION issue, see: https://github.com/LAION-AI/CLAP/issues/118
    pkg = load_state_dict(clap_path)
    pkg.pop('text_branch.embeddings.position_ids', None)
    model.model.load_state_dict(pkg)
    model.eval()

    if not os.path.isdir(audio_path):        
        raise ValueError('audio_path does not exist')

    if id2text:   
        print('[EXTRACTING TEXT EMBEDDINGS] ')
        batch_size = 64
        text_emb = {}
        for i in tqdm(range(0, len(id2text), batch_size)):
            batch_ids = list(id2text.keys())[i:i+batch_size]
            batch_texts = [id2text[id] for id in batch_ids]
            with torch.no_grad():
                embeddings = model.get_text_embedding(batch_texts, use_tensor=True)
            for id, emb in zip(batch_ids, embeddings):
                text_emb[id] = emb

    else:
        raise ValueError('Must specify id2text')

    print('[EVALUATING GENERATIONS] ', audio_path)
    score = 0
    count = 0
    for id in tqdm(id2text.keys()):
        file_path = os.path.join(audio_path, str(id)+audio_files_extension)
        with torch.no_grad():
            audio, _ = librosa.load(file_path, sr=48000, mono=True) # sample rate should be 48000
            audio = pyln.normalize.peak(audio, -1.0)
            audio = audio.reshape(1, -1) # unsqueeze (1,T)
            audio = torch.from_numpy(int16_to_float32(float32_to_int16(audio))).float()
            audio_embeddings = model.get_audio_embedding_from_data(x = audio, use_tensor=True)
        cosine_sim = torch.nn.functional.cosine_similarity(audio_embeddings, text_emb[id].unsqueeze(0), dim=1, eps=1e-8)[0]
        score += cosine_sim
        count += 1

    return score / count if count > 0 else 0


if __name__ == "__main__":

    import pandas as pd

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
