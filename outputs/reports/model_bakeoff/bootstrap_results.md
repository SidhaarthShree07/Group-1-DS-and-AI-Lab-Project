| model | macro-F1 | repair acc | AUROC | SCORE (mean +- std, 95% interval) |
|---|---|---|---|---|
| qwen3_4b | 0.579 +- 0.015 | 0.274 +- 0.013 | 0.767 +- 0.009 | 0.544 +- 0.010  [0.523, 0.564] |
| qwen3_5_9b | 0.643 +- 0.013 | 0.328 +- 0.014 | 0.843 +- 0.009 | 0.609 +- 0.010  [0.588, 0.627] |
| gemma4_e4b | 0.692 +- 0.015 | 0.341 +- 0.014 | 0.860 +- 0.009 | 0.637 +- 0.011  [0.616, 0.659] |

Gemma - qwen3_5_9b: score diff +0.028 (95% interval +0.003 to +0.053); Gemma ahead in 98.5% of 2000 resamples
   macro-F1: +0.048 [+0.013, +0.083], Gemma ahead 99.7%
   repair acc: +0.012 [-0.020, +0.046], Gemma ahead 75.5%
   AUROC: +0.018 [-0.001, +0.036], Gemma ahead 97.0%

Gemma - qwen3_4b: score diff +0.093 (95% interval +0.067 to +0.117); Gemma ahead in 100.0% of 2000 resamples
   macro-F1: +0.112 [+0.072, +0.152], Gemma ahead 100.0%
   repair acc: +0.066 [+0.040, +0.092], Gemma ahead 100.0%
   AUROC: +0.093 [+0.072, +0.115], Gemma ahead 100.0%
