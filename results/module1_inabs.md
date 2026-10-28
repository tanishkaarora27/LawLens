IN-Abs test split, n=100 (gold summaries average 932 words). Weights (cue,emb,pos): [0.3, 0.0, 0.7]. Mean F1 [95% bootstrap CI over documents].

| System @ word budget | rouge1 | rouge2 | rougeL | bertscore_f1 | words | quotes verified |
|---|---|---|---|---|---|---|
| lead@450 | 0.442 [0.421, 0.466] | 0.200 [0.181, 0.222] | 0.216 [0.202, 0.233] | 0.862 [0.859, 0.865] | 450 | n/a |
| cues@450 | 0.488 [0.467, 0.509] | 0.240 [0.224, 0.256] | 0.249 [0.236, 0.262] | 0.876 [0.874, 0.879] | 450 | 1369/1369 |
| lead@900 | 0.486 [0.462, 0.508] | 0.247 [0.227, 0.268] | 0.245 [0.229, 0.263] | 0.862 [0.859, 0.865] | 890 | n/a |
| cues@900 | 0.540 [0.517, 0.561] | 0.286 [0.267, 0.305] | 0.271 [0.257, 0.286] | 0.874 [0.871, 0.877] | 794 | 2403/2403 |

Paired difference vs the lead baseline at the same budget (system minus lead, 95% bootstrap CI). `*` marks a CI that excludes 0.

| System | rouge1 | rouge2 | rougeL | bertscore_f1 |
|---|---|---|---|---|
| cues@450 | +0.046 [+0.033, +0.058]* | +0.040 [+0.024, +0.055]* | +0.032 [+0.019, +0.044]* | +0.014 [+0.012, +0.017]* |
| cues@900 | +0.054 [+0.043, +0.066]* | +0.039 [+0.021, +0.056]* | +0.026 [+0.013, +0.039]* | +0.012 [+0.009, +0.014]* |
