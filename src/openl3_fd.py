import openl3
import soundfile as sf
import numpy as np
from scipy import linalg
import glob
from tqdm import tqdm
import os
import soxr
import pyloudnorm as pyln


def calculate_embd_statistics(embd_lst):
    if isinstance(embd_lst, list):
        embd_lst = np.array(embd_lst)
    mu = np.mean(embd_lst, axis=0)
    sigma = np.cov(embd_lst, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    Adapted from: https://github.com/gudgud96/frechet-audio-distance/blob/main/frechet_audio_distance/fad.py
    
    Numpy implementation of the Frechet Distance.
    
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Params:
    -- mu1: Embedding's mean statistics for generated samples.
    -- mu2: Embedding's mean statistics for reference samples.
    -- sigma1: Covariance matrix over embeddings for generated samples.
    -- sigma2: Covariance matrix over embeddings for reference samples.
    Returns:
    --  Fréchet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
            'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def extract_embeddings(directory_path, channels, samplingrate, content_type, openl3_hop_size, batch_size=16):
    """
    Given a list of files, compute their embeddings in batches.

    If channels == 1: stereo audio is downmixed to mono. Mono embeddings are of dim=512.

    If channels == 2: mono audio is "faked" to stereo by copying the mono channel.
    Stereo embeddings are of dim=1024, since we concatenate L (dim=512) and R (dim=512) embeddings.

    Params:
    -- directory_path: path where the generated audio files are available.
    -- channels: 1 (mono), or 2 (stereo) to get mono or stereo embeddings.
    -- samplingrate: max bandwidth at which we evaluate the given signals. Up to 48kHz.
    -- content_type: 'music' or 'env' to select a content type specific openl3 model.
    -- openl3_hop_size: analysis resolution of openl3 in seconds. Openl3's input window is 1 sec. 
    -- batch_size: number of audio files to process in each batch.
    Returns:
    -- list of embeddings: [np.array[], ...], as expected by calculate_frechet_distance()
    """

    wav_files = glob.glob(directory_path)
    if len(wav_files) == 0:
        raise ValueError('No files with this extension in this path!')
    model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type=content_type, embedding_size=512)
    
    first = True
    for i in tqdm(range(0, len(wav_files), batch_size)):
        batch_files = wav_files[i:i+batch_size]
        batch_audio_l = []
        batch_audio_r = []
        batch_sr = []
        
        for file in batch_files:
            audio, sr = sf.read(file)
            audio = pyln.normalize.peak(audio, -1.0)            
            if audio.shape[0] < sr: 
                print('Audio shorter than 1 sec, openl3 will zero-pad it:', file, audio.shape, sr)

            # resample to the desired evaluation bandwidth
            audio = soxr.resample(audio, sr, samplingrate) # mono/stereo <- mono/stereo, input sr, output sr

            # mono embeddings are stored in batch_audio_l (R channel not used)
            if channels == 1:
                batch_audio_l.append(audio)

            elif channels == 2:
                if audio.ndim == 1:
                    # if mono, "fake" stereo by copying mono channel to L and R
                    batch_audio_l.append(audio)
                    batch_audio_r.append(audio)
                elif audio.ndim == 2:
                    # if it's stereo separate channels for openl3
                    batch_audio_l.append(audio[:,0])
                    batch_audio_r.append(audio[:,1])

            batch_sr.append(samplingrate)

        # extracting mono embeddings (dim=512) or the L channel for stereo embeddings
        emb, _ = openl3.get_audio_embedding(batch_audio_l, batch_sr, model=model, verbose=False, hop_size=openl3_hop_size, batch_size=batch_size)

        # format mono embedding
        if channels == 1:
            emb = np.concatenate(emb,axis=0)
        
        # extracting stereo embeddings (dim=1024), since we concatenate L (dim=512) and R (dim=512) embeddings
        elif channels == 2:
            # extract the missing R channel
            emb_r, _ = openl3.get_audio_embedding(batch_audio_r, batch_sr, model=model, verbose=False, hop_size=openl3_hop_size, batch_size=batch_size)
            emb = [np.concatenate([l, r], axis=1) for l, r in zip(emb, emb_r)]
            emb = np.concatenate(emb, axis=0)

        # concatenate embeddings
        if first:
            embeddings = emb
            first = False
        else:
            embeddings = np.concatenate([embeddings, emb], axis=0)
    
    # return as a list of embeddings: [np.array[], ...]
    return [e for e in embeddings]


def extract_embeddings_nobatching(directory_path, channels, samplingrate, content_type, openl3_hop_size):
    """
    Given a list of files, compute their embeddings one by one.

    If channels == 1: stereo audio is downmixed to mono. Mono embeddings are of dim=512.

    If channels == 2: mono audio is "faked" to stereo by copying the mono channel.
    Stereo embeddings are of dim=1024, since we concatenate L (dim=512) and R (dim=512) embeddings.

    Params:
    -- directory_path: path where the generated audio files are available.
    -- channels: 1 (mono), or 2 (stereo) to get mono or stereo embeddings.
    -- samplingrate: max bandwidth at which we evaluate the given signals. Up to 48kHz.
    -- content_type: 'music' or 'env' to select a content type specific openl3 model.
    -- openl3_hop_size: analysis resolution of openl3 in seconds. Openl3's input window is 1 sec. 
    Returns:
    -- list of embeddings: [np.array[], ...], as expected by calculate_frechet_distance()
    """

    wav_files = glob.glob(directory_path)
    if len(wav_files) == 0:
        raise ValueError('No files with this extension in this path!')    
    model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type=content_type, embedding_size=512)

    first = True
    for file in tqdm(wav_files):
        audio, sr = sf.read(file)
        audio = pyln.normalize.peak(audio, -1.0)
        if audio.shape[0] < sr: 
            print('Audio shorter than 1 sec, openl3 will zero-pad it:', file, audio.shape, sr)

        # resample to the desired evaluation bandwidth
        audio = soxr.resample(audio, sr, samplingrate) # mono/stereo <- mono/stereo, input sr, output sr

        # extracting stereo embeddings (dim=1024), since we concatenate L (dim=512) and R (dim=512) embeddings
        if channels == 2:
            if audio.ndim == 1:
                audio_l3, sr_l3 = audio, samplingrate
            elif audio.ndim == 2:
                # if it's stereo separate channels for openl3
                audio_l3 = [audio[:,0], audio[:,1]]
                sr_l3 = [samplingrate, samplingrate]
            emb, _ = openl3.get_audio_embedding(audio_l3, sr_l3, model=model, verbose=False, hop_size=openl3_hop_size)
            if audio.ndim == 1:
                # if mono audio, "fake" stereo by concatenating mono embedding as L and R embeddings
                emb = np.concatenate([emb, emb],axis=1)
            elif audio.ndim == 2:
                emb = np.concatenate(emb,axis=1)

        # or extracting mono embeddings (dim=512)
        elif channels == 1: 
            emb, _ = openl3.get_audio_embedding(audio, samplingrate, model=model, verbose=False, hop_size=openl3_hop_size)

        # concatenate embeddings
        if first:
            embeddings = emb
            first = False
        else:
            embeddings = np.concatenate([embeddings, emb], axis=0)
    
    # return as a list of embeddings: [np.array[], ...]
    return [e for e in embeddings]


def openl3_fd(channels, samplingrate, content_type, openl3_hop_size, eval_path, 
              eval_files_extension='.wav', ref_path=None, ref_files_extension='.wav', load_ref_embeddings=None, batching=False):
    """
    Compute the Fréchet Distance between files in eval_path and ref_path.
    
    Fréchet distance computed on top of openl3 embeddings.

    GPU-based computation.

    Extracting the embeddings is timeconsuming. After being computed once, we store them.
    We store pre-computed reference embedding statistics in load/openl3_fd/ 
    To load those and save computation, just set the path in load_ref_embeddings.
    If load_ref_embeddings is set, ref_path is not required.

    Params:
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

    if not os.path.isdir(eval_path):        
        raise ValueError('eval_path does not exist')

    if load_ref_embeddings:
        if not os.path.exists(load_ref_embeddings):
            raise ValueError('load_ref_embeddings does not exist')
        print('[LOADING REFERENCE EMBEDDINGS] ', load_ref_embeddings)
        loaded = np.load(load_ref_embeddings)
        mu_ref = loaded['mu_ref']
        sigma_ref = loaded['sigma_ref']

    else:
        if ref_path:
            if not os.path.isdir(ref_path):
                raise ValueError("ref_path does not exist")
            path = os.path.join(ref_path, '*'+ref_files_extension)
            print('[EXTRACTING REFERENCE EMBEDDINGS] ', path)
            if batching:
                ref_embeddings = extract_embeddings(path, channels, samplingrate, content_type, openl3_hop_size, batch_size=batching)
            else:
                ref_embeddings = extract_embeddings_nobatching(path, channels, samplingrate, content_type, openl3_hop_size)            
            mu_ref, sigma_ref = calculate_embd_statistics(ref_embeddings)

            # store statistics to load later on
            if not os.path.exists('load/openl3_fd'):
                os.makedirs('load/openl3_fd/')
            save_ref_embeddings_path = (
                'load/openl3_fd/' +
                path.replace('/', '_') +
                '__channels' + str(channels) +
                '__' + str(samplingrate) +
                '__openl3' + str(content_type) +
                '__openl3hopsize' + str(openl3_hop_size) +
                '__batch' + str(batching) +
                '.npz'
            )                
            np.savez(save_ref_embeddings_path, mu_ref=mu_ref, sigma_ref=sigma_ref)
            print('[REFERENCE EMBEDDINGS][SAVED] ', save_ref_embeddings_path)

        else:
            raise ValueError('Must specify ref_path or load_ref_embeddings')

    path = os.path.join(eval_path, '*'+eval_files_extension)
    print('[EXTRACTING EVALUATION EMBEDDINGS] ', path)
    if batching:
        eval_embeddings = extract_embeddings(path, channels, samplingrate, content_type, openl3_hop_size, batch_size=batching)
    else:
        eval_embeddings = extract_embeddings_nobatching(path, channels, samplingrate, content_type, openl3_hop_size)    
    mu_eval, sigma_eval = calculate_embd_statistics(eval_embeddings)

    fd = calculate_frechet_distance(mu_eval, sigma_eval, mu_ref, sigma_ref)
    if load_ref_embeddings:
        print('[FRéCHET DISTANCE] ', eval_path, load_ref_embeddings, fd)
    else:
        print('[FRéCHET DISTANCE] ', eval_path, ref_path, fd)

    return fd

if __name__ == "__main__":

    reference_path = 'musiccaps_folder'
    evaluation_path = 'your_model_outputs_folder'
    model_channels = 2 # 1 or 2 channels
    model_sr = 44100 # maximum bandwidth at which we evaluate, up to 48kHz
    type = 'music' # openl3 model trained on 'music' or 'env'
    hop = 0.5 # openl3 hop_size in seconds (openl3 window is 1 sec)

    _ = openl3_fd(
        channels=model_channels,
        samplingrate=model_sr,
        content_type=type,
        openl3_hop_size=hop,
        eval_path=evaluation_path,
        ref_path=reference_path,
        batching=False
    )