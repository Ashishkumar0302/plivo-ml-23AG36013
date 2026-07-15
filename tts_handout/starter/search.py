"""Search skeleton: random walk over voice-tensor space. Runs as-is, but
the fitness function and search strategy are deliberately naive — that is
your hour.

    python search.py --reference_dir ../reference --start blend_baseline.pt \
        --iters 150 --out voice.pt

Ideas the skeleton does NOT do for you:
  * fitness beyond raw similarity (naturalness terms, self-similarity
    across sentences, spectral sanity checks) — see the warning in
    similarity.py
  * evaluating on 2-3 DIFFERENT sentences per candidate (one sentence
    overfits)
  * annealing the step size; accepting sideways moves; restarts
  * structured perturbations: are all 256 dimensions doing the same kind
    of work? Perturb halves separately and find out.
  * the tensor is 510 rows of 256 dims — synthesizing a given text uses ONE
    row, picked by the text's phoneme count. Which rows does your fitness
    actually test, and what is randn_like doing to all the others? (This is
    why a local gain can evaporate on sentences you never evaluated.)
  * listening checkpoints: dump audio every N accepted steps and USE YOUR EARS
"""
import argparse

import torch

import synth
import similarity as sim

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please confirm your order number after the beep.",
    "I will call you back tomorrow at three thirty.",
]


def fitness(voice, target_emb, texts):
    """TODO: this is the whole game. Raw similarity WILL get gamed."""
    total = 0.0
    for t in texts:
        wav = synth.synthesize(t, voice)
        total += sim.similarity_to_target(wav, target_emb)
    return total / len(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--start", required=True, help="starting .pt tensor")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--step", type=float, default=0.03)
    ap.add_argument("--out", default="voice.pt")
    ap.add_argument("--listen_every", type=int, default=5)
    args = ap.parse_args()

    target = sim.target_embedding(args.reference_dir)
    best = synth.load_voice(args.start).clone()
    best_f = fitness(best, target, SENTENCES[:1])   # TODO: 1 sentence = overfit
    print(f"start fitness: {best_f:.4f}")

    accepted = 0
    for i in range(1, args.iters + 1):
        cand = best + args.step * torch.randn_like(best)   # TODO: structure?
        f = fitness(cand, target, SENTENCES[:1])
        if f > best_f:                                     # TODO: acceptance rule?
            best, best_f, accepted = cand, f, accepted + 1
            print(f"iter {i:4d}  accepted #{accepted}  fitness {best_f:.4f}")
            if accepted % args.listen_every == 0:
                import soundfile as sf
                sf.write(f"listen_{accepted}.wav",
                         synth.synthesize(SENTENCES[0], best), synth.SR)
                print(f"  -> wrote listen_{accepted}.wav — GO LISTEN")

    torch.save(best, args.out)
    import soundfile as sf
    sf.write("listen_final.wav", synth.synthesize(SENTENCES[0], best), synth.SR)
    print(f"final fitness {best_f:.4f} -> saved {args.out}")
    print("wrote listen_final.wav — LISTEN BEFORE YOU SUBMIT")


if __name__ == "__main__":
    main()
