import torch

from simpler_env.policies.pizero.pizero_model import PiZeroInference
from .kde_contrast_decoding import ContrastDecoding


class PiZeroContrastInference(PiZeroInference):
    def __init__(self, 
                 alpha=0.1,
                 num_repeats=64,
                 bandwidth_factor=1.0,
                 keep_threshold=0.5,
                 ag_weight=0.5,
                 knn_k=5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_repeats = num_repeats
        self.ag_weight = ag_weight
        self.knn_k = knn_k
        self.contrast_decoding = ContrastDecoding(alpha, bandwidth_factor, keep_threshold, 'torch')

        # set to None to disable clipping in infer_action function
        self.clip_value = self.model.final_action_clip_value
        self.model.final_action_clip_value = None
    
    @torch.no_grad()
    def step(self, image, contrast_image, instruction, proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        contrast_inputs = self.preprocess_inputs(contrast_image, instruction, proprio)
       
        # actions = self.forward_actions(inputs)
        # contrast_actions = self.forward_actions(contrast_inputs)
 
        all_inputs = {}
        for k in inputs:
            all_inputs[k] = torch.cat([inputs[k], contrast_inputs[k]], dim=0)
        all_actions = self.forward_actions(all_inputs)
        actions, contrast_actions = torch.chunk(all_actions, 2, dim=0)

        raw_actions = self.contrast_decoding(actions, contrast_actions)
        if self.clip_value is not None:
            raw_actions = torch.clamp(raw_actions, -self.clip_value, self.clip_value)
        
        actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
        return raw_actions, actions, {}

    @torch.no_grad()
    def knn_de_step(self, image, contrast_image, instruction, proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        contrast_inputs = self.preprocess_inputs(contrast_image, instruction, proprio)
       
        # actions = self.forward_actions(inputs)
        # contrast_actions = self.forward_actions(contrast_inputs)
 
        all_inputs = {}
        for k in inputs:
            all_inputs[k] = torch.cat([inputs[k], contrast_inputs[k]], dim=0)
        all_actions = self.forward_actions(all_inputs)
        actions, contrast_actions = torch.chunk(all_actions, 2, dim=0)
        # print("actions.shape:", actions.shape, "contrast_actions.shape:", contrast_actions.shape)

        def cd_with_knn(actions, contrast_actions, eps=1e-8):
            """
            actions: (N, T, D) = (24, 4, 7)
            contrast_actions: same shape

            return:
                best_action: (1, T, D)
            """

            N = actions.shape[0]
            A = actions.reshape(N, -1).float()              # (N, 28)
            B = contrast_actions.reshape(N, -1).float()     # (N, 28)

            # kNN in B
            dist_AB = torch.cdist(A, B, p=2) ** 2   # (N, N)
            print("knn_k:", self.knn_k)
            knn_dist_AB, _ = torch.topk(dist_AB, k=self.knn_k, largest=False, dim=1)
            R_B = knn_dist_AB.sum(dim=1)  # (N,)

            # kNN in A (exclude self)
            dist_AA = torch.cdist(A, A, p=2) ** 2   # (N, N)
            # mask diagonal (self-distance = 0)
            inf_mask = torch.eye(N, device=A.device) * 1e9
            dist_AA = dist_AA + inf_mask
            knn_dist_AA, _ = torch.topk(dist_AA, k=self.knn_k, largest=False, dim=1)
            R_A = knn_dist_AA.sum(dim=1)  # (N,)

            # best of N 
            scores = torch.log(R_B + eps) - torch.log(R_A + eps)  # (N,)
            best_idx = torch.argmax(scores)
            best_action = actions[best_idx:best_idx+1]  # (1, T, D)

            return best_action
        
        raw_actions = cd_with_knn(actions, contrast_actions) # should have shape of (1,4,7)
        if self.clip_value is not None:
            raw_actions = torch.clamp(raw_actions, -self.clip_value, self.clip_value)
        
        actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
        return raw_actions, actions, {}
    
    @torch.no_grad()
    def ag_step(self, image, contrast_image, instruction, proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        contrast_inputs = self.preprocess_inputs(contrast_image, instruction, proprio)
       
        # actions = self.forward_actions(inputs)
        # contrast_actions = self.forward_actions(contrast_inputs)
 
        all_inputs = {}
        for k in inputs:
            all_inputs[k] = torch.cat([inputs[k], contrast_inputs[k]], dim=0)
        all_actions = self.auto_guidance_forward_actions(all_inputs)

        assert len(all_actions) == 2, "AutoGuidance without CD should return 2 actions"

        actions, contrast_actions = torch.chunk(all_actions, 2, dim=0)

        raw_actions = actions
        if self.clip_value is not None:
            raw_actions = torch.clamp(raw_actions, -self.clip_value, self.clip_value)
        
        actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
        return raw_actions, actions, {}

    @torch.no_grad()
    def contrast_in_ag_step(self, image, contrast_image, instruction, proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        contrast_inputs = self.preprocess_inputs(contrast_image, instruction, proprio)
       
        # actions = self.forward_actions(inputs)
        # contrast_actions = self.forward_actions(contrast_inputs)
 
        all_inputs = {}
        for k in inputs:
            all_inputs[k] = torch.cat([inputs[k], contrast_inputs[k]], dim=0)
        all_actions = self.cd_in_ag_forward_actions(all_inputs, cd_function=self.contrast_decoding)

        # assert len(all_actions) == 2, "AutoGuidance without CD should return 2 actions"
        actions, contrast_actions = torch.chunk(all_actions, 2, dim=0)
        raw_actions = self.contrast_decoding(actions, contrast_actions)

        if self.clip_value is not None:
            raw_actions = torch.clamp(raw_actions, -self.clip_value, self.clip_value)
        
        actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
        return raw_actions, actions, {}

    @torch.no_grad()
    def ag_contrast_step(self, image, contrast_image, instruction, proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        contrast_inputs = self.preprocess_inputs(contrast_image, instruction, proprio)
       
        # actions = self.forward_actions(inputs)
        # contrast_actions = self.forward_actions(contrast_inputs)

        '''
        input_ids torch.Size([1, 276])
        pixel_values torch.Size([1, 3, 224, 224])
        vlm_position_ids torch.Size([1, 276])
        proprio_position_ids torch.Size([1, 1])
        action_position_ids torch.Size([1, 4])
        proprios torch.Size([1, 1, 8])
        image_text_proprio_mask torch.Size([1, 1, 277, 277])
        action_mask torch.Size([1, 1, 4, 281])
        '''
 
        all_inputs = {}
        for k in inputs:
            all_inputs[k] = torch.cat([inputs[k], contrast_inputs[k]], dim=0)
        all_actions = self.auto_guidance_forward_actions(all_inputs) # 2*num_repeats, 4, 7
        actions, contrast_actions = torch.chunk(all_actions, 2, dim=0) # num_repeats, 4, 7
        raw_actions = self.contrast_decoding(actions, contrast_actions) # 1, 4, 7

        if self.clip_value is not None:
            raw_actions = torch.clamp(raw_actions, -self.clip_value, self.clip_value)
        
        actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
        return raw_actions, actions, {}

    def forward_actions(self, inputs):
        inputs.update({'num_repeats': self.num_repeats})
        with torch.inference_mode():
            if self.use_naive:
                actions = self.model.infer_actions_naive(**inputs)
            else:
                actions = self.model.infer_actions(**inputs)
        return actions

    def auto_guidance_forward_actions(self, inputs):
        inputs.update({'num_repeats': self.num_repeats})
        inputs.update({'ag_weight': self.ag_weight})
        with torch.inference_mode():
            if self.use_naive:
                actions = self.model.infer_actions_naive(**inputs)
                assert 2!=2
            else:
                actions = self.model.auto_guidance_infer_actions(**inputs)
        return actions

    def cd_in_ag_forward_actions(self, inputs, cd_function):
        inputs.update({'num_repeats': self.num_repeats})
        inputs.update({'ag_weight': self.ag_weight})
        with torch.inference_mode():
            if self.use_naive:
                actions = self.model.infer_actions_naive(**inputs)
                assert 2!=2
            else:
                actions = self.model.cd_in_ag_infer_actions(**inputs, cd_function=cd_function)
        return actions