"""Speaker-similarity utilities (Resemblyzer, CPU, weights ship in the pip
package — fully offline).

WARNING you will rediscover the hard way if you ignore it: a search that
maximizes ONLY this similarity will find degraded, artifact-heavy audio
that the embedding scores highly and a human immediately rejects. Your
fitness function is part of the assignment. Listen to your candidates.
"""
import glob
import os

import numpy as np
import soundfile as sf

_ENCODER = None
SR = 24000


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        from resemblyzer import VoiceEncoder
        _ENCODER = VoiceEncoder("cpu", verbose=False)
    return _ENCODER


def embed_wav_array(wav, sr=SR):
    from resemblyzer import preprocess_wav
    return _encoder().embed_utterance(preprocess_wav(wav, source_sr=sr))


def embed_file(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return embed_wav_array(wav, sr)


def target_embedding(reference_dir):
    """Average embedding of every wav in the reference folder."""
    files = sorted(glob.glob(os.path.join(reference_dir, "*.wav")))
    if not files:
        raise SystemExit(f"no wavs in {reference_dir}")
    embs = np.stack([embed_file(f) for f in files])
    m = embs.mean(axis=0)
    return m / (np.linalg.norm(m) + 1e-9)


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def similarity_to_target(wav, target_emb, sr=SR):
    return cosine(embed_wav_array(wav, sr), target_emb)
