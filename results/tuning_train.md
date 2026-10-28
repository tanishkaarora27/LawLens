IN-Abs train split, n=10 (gold summaries average 930 words). Mean F1 [95% bootstrap CI over documents].

| System @ word budget | rouge1 | rouge2 | rougeL | bertscore_f1 | words | quotes verified |
|---|---|---|---|---|---|---|
| lead@450 | 0.483 [0.384, 0.563] | 0.228 [0.155, 0.296] | 0.233 [0.178, 0.294] | 0.870 [0.860, 0.878] | 450 | n/a |
| cues@450 | 0.475 [0.399, 0.535] | 0.207 [0.177, 0.235] | 0.225 [0.195, 0.250] | 0.876 [0.868, 0.883] | 404 | 105/105 |
| full@450 | 0.470 [0.397, 0.524] | 0.201 [0.167, 0.231] | 0.226 [0.191, 0.252] | 0.876 [0.869, 0.882] | 404 | 109/109 |
