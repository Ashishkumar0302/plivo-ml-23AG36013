# Starter

Setup (BEFORE the hour — downloads ~1 GB, must print READY):
  python setup_env.py
  # espeak-ng must be installed on your machine first (see setup_env.py header)

Your loop:
  python blend.py  --reference_dir ../reference          # baseline + ranking
  python search.py --reference_dir ../reference --start blend_baseline.pt --out voice.pt

Log every run in RUNLOG.md — including what you HEARD, not just scores.

Before time is up, your submission folder needs: voice.pt, your search
code, RUNLOG.md, NOTES.md, and SUMMARY.html (see the assignment brief,
"Deliverables" section).
