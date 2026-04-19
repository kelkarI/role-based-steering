# Main results — Gemma vs Qwen

Held-out test split, Holm–Bonferroni across all 24 conditions.
Qwen rows are mean ± cross-seed std (3 test seeds).
Gemma rows are single-seed for the 3 residual conditions (added after
the Gemma multi-seed run) and multi-seed for the 21 other conditions.

| Model | Cond | best coef | baseline logit | Δ logit | Δ rate pp | sig seeds | Holm p (seed 42) |
|---|---|---|---|---|---|---|---|
| gemma-2-27b-it | assistant_axis | +2000 | +1.009 | -0.370 ± 0.032 | -4.33 | 3/3 | 3.0747162625746185e-18 |
| gemma-2-27b-it | caa | -2000 | +1.009 | -0.874 ± 0.014 | -9.0 | 3/3 | 3.2612772181140693e-23 |
| gemma-2-27b-it | devils_advocate | +2000 | +1.009 | -0.516 ± 0.028 | -8.33 | 3/3 | 2.6443008559004035e-17 |
| gemma-2-27b-it | contrarian | +2000 | +1.009 | -0.281 ± 0.007 | -3.0 | 3/3 | 1.5644721576841046e-07 |
| gemma-2-27b-it | skeptic | +2000 | +1.009 | -0.706 ± 0.027 | -9.33 | 3/3 | 4.066228533757956e-21 |
| gemma-2-27b-it | judge | +2000 | +1.009 | -0.551 ± 0.013 | -9.0 | 3/3 | 9.147343309770509e-21 |
| gemma-2-27b-it | scientist | +2000 | +1.009 | -0.503 ± 0.005 | -7.0 | 3/3 | 9.948490510969348e-21 |
| gemma-2-27b-it | peacekeeper | +2000 | +1.009 | -0.046 ± 0.018 | 3.67 | 0/3 | 1.0 |
| gemma-2-27b-it | pacifist | +2000 | +1.009 | +0.105 ± 0.013 | 2.33 | 1/3 | 0.07248705531003066 |
| gemma-2-27b-it | collaborator | +500 | +1.009 | +0.050 ± 0.009 | 3.33 | 2/3 | 0.002363166292437025 |
| gemma-2-27b-it | facilitator | -5000 | +1.009 | -0.722 ± 0.019 | -9.0 | 0/3 | 1.0 |
| gemma-2-27b-it | skeptic_residual | +2000 | +1.009 | -0.735 ± 0.026 | -9.33 | 3/3 | 3.603023952696323e-11 |
| gemma-2-27b-it | contrarian_residual | +2000 | +1.009 | -0.323 ± 0.004 | -4.0 | 3/3 | 0.07248705531003066 |
| gemma-2-27b-it | collaborator_residual | +5000 | +1.009 | -0.915 ± 0.028 | -8.33 | 3/3 | 7.142077647409764e-19 |
| qwen3-32b | assistant_axis | +200 | +3.024 | -2.434 ± 0.082 | -21.33 | 3/3 | 1.6430997833803619e-19 |
| qwen3-32b | caa | -200 | +3.024 | -1.989 ± 0.042 | -20.67 | 3/3 | 3.4244421096293767e-15 |
| qwen3-32b | devils_advocate | +200 | +3.024 | -2.296 ± 0.071 | -15.33 | 3/3 | 2.5996968152224312e-18 |
| qwen3-32b | contrarian | -200 | +3.024 | -2.299 ± 0.071 | -15.33 | 3/3 | 2.9693128035875663e-18 |
| qwen3-32b | skeptic | +200 | +3.024 | -1.847 ± 0.134 | -16.33 | 3/3 | 3.652751005799948e-17 |
| qwen3-32b | judge | +200 | +3.024 | -1.723 ± 0.094 | -4.0 | 3/3 | 7.650696045339467e-16 |
| qwen3-32b | scientist | -100 | +3.024 | -1.008 ± 0.161 | -5.0 | 3/3 | 7.585917613787059e-12 |
| qwen3-32b | peacekeeper | -200 | +3.024 | -0.733 ± 0.094 | 0.33 | 0/3 | 1.0 |
| qwen3-32b | pacifist | +500 | +3.024 | -3.003 ± 0.008 | -33.67 | 0/3 | 1.0 |
| qwen3-32b | collaborator | -100 | +3.024 | -0.054 ± 0.153 | -2.33 | 0/3 | 1.0 |
| qwen3-32b | facilitator | -200 | +3.024 | -0.493 ± 0.244 | 0.67 | 0/3 | 1.0 |
| qwen3-32b | skeptic_residual | +200 | +3.024 | -1.823 ± 0.138 | -14.67 | 3/3 | 2.3616435885498267e-16 |
| qwen3-32b | contrarian_residual | -100 | +3.024 | -1.671 ± 0.123 | -11.0 | 3/3 | 6.57068917964229e-19 |
| qwen3-32b | devils_advocate_residual | +200 | +3.024 | -2.217 ± 0.076 | -13.33 | 3/3 | 3.419731814579329e-18 |
