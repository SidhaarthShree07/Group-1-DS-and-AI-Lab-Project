Seeds: [1, 2, 3] (each = a fresh draw of 1,500 cells; greedy decoding, so seeds only change which cells are tested)

| model | json_valid_pct | precision | recall | macro_f1 | auroc | repair_acc | false_repair | score |
|---|---|---|---|---|---|---|---|---|
| qwen3_4b | 99.844 +- 0.102 | 0.510 +- 0.015 | 0.627 +- 0.026 | 0.585 +- 0.017 | 0.776 +- 0.013 | 0.271 +- 0.004 | 0.677 +- 0.010 | 0.548 +- 0.009 |
| qwen3_5_9b | 99.333 +- 0.231 | 0.547 +- 0.007 | 0.753 +- 0.009 | 0.642 +- 0.008 | 0.838 +- 0.001 | 0.317 +- 0.008 | 0.701 +- 0.003 | 0.603 +- 0.005 |
| gemma4_e4b | 99.978 +- 0.038 | 0.663 +- 0.011 | 0.742 +- 0.014 | 0.698 +- 0.011 | 0.864 +- 0.004 | 0.357 +- 0.014 | 0.393 +- 0.022 | 0.645 +- 0.007 |

Score per seed:
| seed | qwen3_4b | qwen3_5_9b | gemma4_e4b | winner |
|---|---|---|---|---|
| 1 | 0.538 | 0.601 | 0.652 | gemma4_e4b |
| 2 | 0.551 | 0.609 | 0.646 | gemma4_e4b |
| 3 | 0.556 | 0.599 | 0.637 | gemma4_e4b |

Per-table F1 (mean +- std over seeds):
  qwen3_4b: beers 0.46+-0.05, flights 0.28+-0.03, hospital 0.91+-0.02, movies 0.62+-0.05, rayyan 0.66+-0.04
  qwen3_5_9b: beers 0.60+-0.02, flights 0.36+-0.01, hospital 0.89+-0.02, movies 0.65+-0.01, rayyan 0.70+-0.03
  gemma4_e4b: beers 0.66+-0.02, flights 0.68+-0.01, hospital 0.86+-0.02, movies 0.67+-0.03, rayyan 0.62+-0.01

Gemma - qwen3_5_9b: score diff per seed [0.051, 0.037, 0.038], mean +0.042 +- 0.007

Gemma - qwen3_4b: score diff per seed [0.114, 0.095, 0.082], mean +0.097 +- 0.016

Seed wins: {'qwen3_4b': 0, 'qwen3_5_9b': 0, 'gemma4_e4b': 3}
WINNER (highest mean score): gemma4_e4b
