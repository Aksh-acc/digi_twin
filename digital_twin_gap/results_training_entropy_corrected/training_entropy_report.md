# Entropy introduced in training -- results

Clean (eps=0) empirical training-label entropy, in bits (max = 2.0 for a uniform 4-way label):

- agent training labels : **1.9943 bits**
- human training labels : **1.9978 bits**

Interpretation: as entropy is injected into the training labels, matched-cell accuracy degrades toward the 25% chance line. A twin whose accuracy collapses quickly is highly reliant on clean, low-entropy supervision; one that degrades slowly has learned more robust structure. Comparing the agent and human curves at *equal training entropy* isolates how much of the human-agent gap is due to target noise versus genuine behavioural structure.

See `training_entropy_curve.csv` for the full grid and `fig_training_entropy.png` for the curves.