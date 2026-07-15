import argparse
import json
import os
import random
import time

import numpy as np
import soundfile as sf
import torch

import similarity as sim
import synth


SEARCH_TEXTS = [
    "Hi there, thanks for waiting. Could you tell me a little more about the issue you're seeing?",
    "The delivery should arrive between nine thirty and eleven on Saturday morning.",
    "I grew up in a small town, but I've been living in the city for almost eight years now.",
    "Are you sure the cable is plugged in properly? Sometimes the connector comes loose.",
    "Please confirm your order number after the beep.",
    "I will call you back tomorrow at three thirty.",
]

VALID_TEXTS = [
    "We can reschedule your appointment to Tuesday afternoon if that works better for you.",
    "The total for this month comes to one thousand two hundred and forty five rupees.",
    "Please read out the last four digits of the reference number on your screen.",
]


def score_voice(voice, target, texts):
    vals = []
    rms_penalty = 0.0
    clip_penalty = 0.0

    for text in texts:
        wav = synth.synthesize(text, voice)
        vals.append(sim.similarity_to_target(wav, target))

        rms = float(np.sqrt(np.mean(np.square(wav)) + 1e-9))
        peak = float(np.max(np.abs(wav)))
        if rms < 0.015 or rms > 0.18:
            rms_penalty += abs(rms - 0.07)
        if peak > 0.98:
            clip_penalty += peak - 0.98

    vals = np.array(vals, dtype=np.float32)
    return float(vals.mean() - 0.25 * vals.std() - 0.10 * rms_penalty - 0.25 * clip_penalty), vals.tolist()


def blend(voices, names, weights):
    out = None
    for name, w in zip(names, weights):
        term = voices[name] * float(w)
        out = term if out is None else out + term
    return out.detach().to(torch.float32).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--out", default="voice.pt")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--listen_dir", default="listen_candidates")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.listen_dir, exist_ok=True)

    target = sim.target_embedding(args.reference_dir)
    voices = synth.stock_voices()

    print("Ranking stock voices...")
    stock = []
    for name, voice in voices.items():
        f, vals = score_voice(voice, target, SEARCH_TEXTS[:3])
        stock.append((f, name, vals))
        print(f"{name:20s} train={f:.4f}")

    stock.sort(reverse=True)
    names = [name for _, name, _ in stock[:args.top_k]]

    print("\nUsing top voices:")
    for f, name, vals in stock[:args.top_k]:
        print(f"{name:20s} {f:.4f} {vals}")

    best = None
    best_record = None

    # include pure stock voices
    for name in names:
        weights = np.zeros(len(names), dtype=np.float32)
        weights[names.index(name)] = 1.0
        voice = voices[name]
        train_f, train_vals = score_voice(voice, target, SEARCH_TEXTS)
        valid_f, valid_vals = score_voice(voice, target, VALID_TEXTS)
        objective = 0.65 * train_f + 0.35 * valid_f
        rec = {
            "kind": "stock",
            "names": names,
            "weights": weights.tolist(),
            "train": train_f,
            "valid": valid_f,
            "objective": objective,
            "train_vals": train_vals,
            "valid_vals": valid_vals,
        }
        if best_record is None or objective > best_record["objective"]:
            best = voice.clone()
            best_record = rec

    print("\nSearching convex blends...")
    for i in range(1, args.iters + 1):
        # Dirichlet alpha < 1 favors sparse blends; safer than soup blends.
        alpha = np.ones(len(names), dtype=np.float32) * 0.45
        weights = np.random.dirichlet(alpha).astype(np.float32)

        voice = blend(voices, names, weights)
        train_f, train_vals = score_voice(voice, target, SEARCH_TEXTS)
        valid_f, valid_vals = score_voice(voice, target, VALID_TEXTS)
        objective = 0.65 * train_f + 0.35 * valid_f

        if objective > best_record["objective"]:
            best = voice.clone()
            best_record = {
                "kind": "blend",
                "names": names,
                "weights": weights.tolist(),
                "train": train_f,
                "valid": valid_f,
                "objective": objective,
                "train_vals": train_vals,
                "valid_vals": valid_vals,
                "iter": i,
            }
            print(f"iter {i:04d} NEW objective={objective:.4f} train={train_f:.4f} valid={valid_f:.4f}")
            print("  " + ", ".join(f"{n}:{w:.3f}" for n, w in zip(names, weights) if w > 0.03))

    torch.save(best, args.out)

    with open("best_blend.json", "w") as f:
        json.dump(best_record, f, indent=2)

    for idx, text in enumerate(VALID_TEXTS, 1):
        wav = synth.synthesize(text, best)
        sf.write(os.path.join(args.listen_dir, f"best_valid_{idx}.wav"), wav, synth.SR)

    print("\nBEST")
    print(json.dumps(best_record, indent=2))
    print(f"\nsaved {args.out}")
    print(f"wrote listen files under {args.listen_dir}/")


if __name__ == "__main__":
    main()


