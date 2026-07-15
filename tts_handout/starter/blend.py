"""Baseline (mediocre on purpose): score all stock voices against the
target speaker, print the ranking, and save a naive 50/50 blend of the top
two as blend_baseline.pt. Beating this file is the assignment.

    python blend.py --reference_dir ../reference --text "The quick brown fox jumps over the lazy dog."
"""
import argparse

import torch

import synth
import similarity as sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--text",
                    default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--out", default="blend_baseline.pt")
    args = ap.parse_args()

    target = sim.target_embedding(args.reference_dir)
    voices = synth.stock_voices()
    print(f"scoring {len(voices)} stock voices against the target ...")

    scores = []
    for name, v in voices.items():
        wav = synth.synthesize(args.text, v)
        s = sim.similarity_to_target(wav, target)
        scores.append((s, name))
        print(f"  {name:20s} {s:.4f}")
    scores.sort(reverse=True)

    print("\ntop 5:")
    for s, name in scores[:5]:
        print(f"  {name:20s} {s:.4f}")

    (s1, n1), (s2, n2) = scores[0], scores[1]
    blend = 0.5 * voices[n1] + 0.5 * voices[n2]   # naive; weights unsearched
    torch.save(blend, args.out)
    wav = synth.synthesize(args.text, blend)
    b = sim.similarity_to_target(wav, target)
    print(f"\nnaive 50/50 blend of {n1}+{n2}: {b:.4f}  -> saved {args.out}")
    print("BASELINE TO BEAT:", max(b, s1))


if __name__ == "__main__":
    main()
