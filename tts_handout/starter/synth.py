"""Offline Kokoro wrapper: text + style tensor -> waveform.

Assets are expected under KOKORO_DIR (default ./kokoro_assets), containing
config.json, the model .pth, and voices/*.pt — prepare_kit.py lays this out.

NOTE: the kokoro pip package API has moved between versions. If your
installed version differs, the ONLY two lines you may need to adjust are
marked [A] and [B] below (model construction and pipeline call).

    python synth.py --text "Hello there" --voice kokoro_assets/voices/af_heart.pt --out test.wav
"""
import argparse
import glob
import os

import numpy as np
import soundfile as sf
import torch

def _default_kokoro_dir():
    # student layout: starter/ scripts run with assets one level up
    for d in ("./kokoro_assets", "../kokoro_assets"):
        if os.path.isdir(d):
            return d
    return "./kokoro_assets"


KOKORO_DIR = os.environ.get("KOKORO_DIR") or _default_kokoro_dir()
SR = 24000
_PIPELINE = None


def _find_model_path():
    cands = glob.glob(os.path.join(KOKORO_DIR, "*.pth"))
    if not cands:
        raise SystemExit(f"no .pth model file under {KOKORO_DIR}")
    return cands[0]


def get_pipeline(lang_code="a"):
    """lang_code: 'a' American English, 'b' British, 'h' Hindi, ..."""
    global _PIPELINE
    if _PIPELINE is None:
        from kokoro import KModel, KPipeline
        model = KModel(config=os.path.join(KOKORO_DIR, "config.json"),
                       model=_find_model_path())                     # [A]
        _PIPELINE = KPipeline(lang_code=lang_code, model=model)
    return _PIPELINE


def load_voice(path_or_tensor):
    if isinstance(path_or_tensor, torch.Tensor):
        t = path_or_tensor
    else:
        t = torch.load(path_or_tensor, map_location="cpu", weights_only=True)
    # kokoro chokes on float64/other dtypes with a cryptic error — normalize
    return t.detach().to(torch.float32).cpu()


def synthesize(text, voice, speed=1.0):
    """Returns float32 waveform at 24 kHz."""
    pipe = get_pipeline()
    v = load_voice(voice)
    chunks = [r.audio for r in pipe(text, voice=v, speed=speed)]        # [B]
    audio = torch.cat([c if isinstance(c, torch.Tensor) else torch.tensor(c)
                       for c in chunks])
    return audio.detach().cpu().numpy().astype(np.float32)


def stock_voices():
    """dict: voice name -> tensor, for all bundled stock voices."""
    out = {}
    for p in sorted(glob.glob(os.path.join(KOKORO_DIR, "voices", "*.pt"))):
        out[os.path.splitext(os.path.basename(p))[0]] = load_voice(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", required=True, help="stock name or .pt path")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default="out.wav")
    args = ap.parse_args()
    voice = args.voice
    if not os.path.exists(voice):
        voice = os.path.join(KOKORO_DIR, "voices", args.voice + ".pt")
    wav = synthesize(args.text, voice, args.speed)
    sf.write(args.out, wav, SR)
    print(f"wrote {args.out} ({len(wav)/SR:.1f}s)")


if __name__ == "__main__":
    main()
