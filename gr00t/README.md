# GR00T + SDN (placeholder)

This directory will hold the GR00T N1.5 / N1.6 integration (SIMPLER and real-world ALOHA experiments).

It should reuse the backbone-agnostic selection in [`sdn/selection.py`](../sdn/selection.py)
(`sdn_select`, `select_by_grounding`, `select_by_smoothness`) and the negative-observation
generator in [`sdn/negatives`](../sdn/negatives), so that both backbones share one
implementation of the method.
