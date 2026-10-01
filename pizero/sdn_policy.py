import torch

from sdn.selection import sdn_select, select_by_grounding, select_by_smoothness
from simpler_env.policies.pizero.pizero_model import PiZeroInference


class PiZeroSDNInference(PiZeroInference):
    """pi0 with Selected Diffusion Noise (SDN) test-time action selection.

    Every step samples `num_samples` action chunks from independent initial noises and
    executes the first `exec_horizon` actions of the selected chunk, where `exec_horizon`
    is the chunk length pi0 was trained with (`horizon_steps` in the eval config).

    Args:
        num_samples: N, candidates sampled per observation (for both G and B).
        knn_k: k of the k-NN grounding score.
        top_m: M, candidates kept by the grounding filter.
        long_horizon: length of the (extended) chunks scored by the smoothness stage;
            defaults to `exec_horizon`, i.e. no extension.
        lambda_energy, ignore_gripper: see `sdn.selection.smoothness_scores`.
    """

    def __init__(self,
                 num_samples=12,
                 knn_k=6,
                 top_m=3,
                 long_horizon=None,
                 lambda_energy=0.05,
                 ignore_gripper=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_naive:
            raise NotImplementedError("SDN sampling requires use_naive=False")

        self.num_samples = num_samples
        self.knn_k = knn_k
        self.top_m = top_m
        self.exec_horizon = self.base_model.horizon_steps
        self.long_horizon = long_horizon or self.exec_horizon
        if self.long_horizon < self.exec_horizon:
            raise ValueError(f"long_horizon={self.long_horizon} < exec_horizon={self.exec_horizon}")
        self.smoothness_kwargs = dict(lambda_energy=lambda_energy, ignore_gripper=ignore_gripper)

    @torch.no_grad()
    def sdn_step(self, image, neg_image, instruction, proprio):
        """Full SDN: grounding filter against the negative set, then smoothness refinement."""
        actions, neg_actions = self._sample_pos_neg(image, neg_image, instruction, proprio, self.long_horizon)
        idx = sdn_select(actions, neg_actions, self.knn_k, self.top_m,
                         grounding_horizon=self.exec_horizon, **self.smoothness_kwargs)
        return self._postprocess(actions[idx])

    @torch.no_grad()
    def grounding_step(self, image, neg_image, instruction, proprio):
        """Ablation: stage 1 only, execute the most grounded candidate."""
        actions, neg_actions = self._sample_pos_neg(image, neg_image, instruction, proprio, self.exec_horizon)
        idx = select_by_grounding(actions, neg_actions, self.knn_k)
        return self._postprocess(actions[idx])

    @torch.no_grad()
    def smooth_step(self, image, instruction, proprio):
        """Ablation: stage 2 only, execute the smoothest of all candidates (no negative set)."""
        inputs = self._preprocess_with_horizon(image, instruction, proprio, self.long_horizon)
        actions = self._sample(inputs)
        idx = select_by_smoothness(actions, **self.smoothness_kwargs)
        return self._postprocess(actions[idx])

    def _sample_pos_neg(self, image, neg_image, instruction, proprio, horizon):
        """Sample N chunks from the original and N from the negative observation in one batch."""
        inputs = self._preprocess_with_horizon(image, instruction, proprio, horizon)
        neg_inputs = self._preprocess_with_horizon(neg_image, instruction, proprio, horizon)
        batch = {k: torch.cat([inputs[k], neg_inputs[k]], dim=0) for k in inputs}
        actions, neg_actions = torch.chunk(self._sample(batch), 2, dim=0)
        return actions, neg_actions

    def _preprocess_with_horizon(self, image, instruction, proprio, horizon):
        # the attention mask and position ids depend on the number of action tokens
        self.base_model.set_action_horizon(horizon)
        return self.preprocess_inputs(image, instruction, proprio)

    def _sample(self, inputs):
        with torch.inference_mode():
            return self.model.infer_actions(**inputs, num_repeats=self.num_samples)

    def _postprocess(self, chunk):
        raw_actions = chunk[None, :self.exec_horizon]
        return raw_actions, self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
