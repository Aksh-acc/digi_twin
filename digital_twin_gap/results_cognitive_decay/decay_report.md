# Cognitive-decay model -- findings

`delta` is the fitted decay rate of decision sharpness over a session; higher = faster degradation into noise. `sharpness_start` is the initial inverse-temperature (higher = more decisive).

- **agent**: δ = 0.0083, sharpness 20.12 → 15.47 (23.1% drop)
- **human**: δ = 0.0253, sharpness 1.03 → 0.96 (6.8% drop)

Note: agent decision sharpness starts far higher than human (agents are near-deterministic given the option features, humans are much noisier from the first trial). The decay term captures *within-run* drift on top of that baseline difference. See `decay_by_agent.csv` for per-model rates and `fig_sharpness_curves.png` for the curves.