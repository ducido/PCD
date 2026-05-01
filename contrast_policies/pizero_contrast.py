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
                 knn_k=10,
                 top_k=5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_repeats = num_repeats
        self.ag_weight = ag_weight
        self.knn_k = knn_k
        self.top_k = top_k
        self.contrast_decoding = ContrastDecoding(alpha, bandwidth_factor, keep_threshold, 'torch')

        # set to None to disable clipping in infer_action function
        self.clip_value = self.model.final_action_clip_value
        self.model.final_action_clip_value = "cc"

    @torch.no_grad()
    def baseline_step(self, image, instruction, proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        raw_actions = self.forward_actions(inputs)
        actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
        return raw_actions, actions
    
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
    def base_best_of_N_smooth_step(self, image, instruction, proprio, M_action_horizon):
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

        self.model._orig_mod.horizon_steps = 4
        self.model._orig_mod.num_action_tokens = 4
        self.model._orig_mod.total_num_tokens = (
            self.model._orig_mod.max_image_text_tokens
            + self.model._orig_mod.num_proprio_tokens
            + self.model._orig_mod.num_action_tokens
        )
        self.model.horizon_steps = 4
        self.model.num_action_tokens = 4
        self.model.total_num_tokens = (
            self.model.max_image_text_tokens
            + self.model.num_proprio_tokens
            + self.model.num_action_tokens
        )
        inputs = self.preprocess_inputs(image, instruction, proprio)
        N_actions = self.forward_actions(inputs)

        self.model._orig_mod.horizon_steps = M_action_horizon
        self.model._orig_mod.num_action_tokens = M_action_horizon
        self.model._orig_mod.total_num_tokens = (
            self.model._orig_mod.max_image_text_tokens
            + self.model._orig_mod.num_proprio_tokens
            + self.model._orig_mod.num_action_tokens
        )
        self.model.horizon_steps = M_action_horizon
        self.model.num_action_tokens = M_action_horizon
        self.model.total_num_tokens = (
            self.model.max_image_text_tokens
            + self.model.num_proprio_tokens
            + self.model.num_action_tokens
        )
        M_inputs = self.preprocess_inputs(image, instruction, proprio)
        '''
        input_ids torch.Size([1, 276])
        pixel_values torch.Size([1, 3, 224, 224])
        vlm_position_ids torch.Size([1, 276])
        proprio_position_ids torch.Size([1, 1])
        action_position_ids torch.Size([1, 16])
        proprios torch.Size([1, 1, 8])
        image_text_proprio_mask torch.Size([1, 1, 277, 277])
        action_mask torch.Size([1, 1, 16, 293])
        '''
       
        # actions = self.forward_actions(inputs)
        # contrast_actions = self.forward_actions(contrast_inputs)
        M_actions = self.M_action_horizon_forward_actions(M_inputs, M_action_horizon=M_action_horizon)

        def jerk_smoothest_action(M_actions):
            """
            M_actions: torch.Tensor of shape (C, T, D)

            Returns:
                best_idx: int
                best_action: (T, D)
                jerk_rms: (C,)
            """
            print(M_actions.shape)

            assert M_actions.ndim == 3
            C, T, D = M_actions.shape
            assert T >= 4, "Need at least 4 timesteps to compute jerk"

            # Normalize M_actions
            std = M_actions.std(dim=(0,1), keepdim=True)
            norm_M_actions = M_actions / (std + 1e-6)

            # Compute jerk (C, T-3, D)
            jerk = (
                norm_M_actions[:, 3:, :-1]
                - 3 * norm_M_actions[:, 2:-1, :-1]
                + 3 * norm_M_actions[:, 1:-2, :-1]
                - norm_M_actions[:, :-3, :-1]
            )

            # ||j||^2 over action dim → (C, T-3)
            jerk_sq = (jerk ** 2).sum(dim=-1)

            # mean over time → (C,)
            jerk_mean = jerk_sq.mean(dim=1)

            # RMS → (C,)
            jerk_rms = torch.sqrt(jerk_mean + 1e-8)  # tránh nan

            # best candidate
            best_idx = torch.argmin(jerk_rms)
            best_action = M_actions[best_idx:best_idx+1]

            return best_action

        best_smooth_action_M = jerk_smoothest_action(M_actions)

        def select_best_from_N(N, M_best):
            ref = M_best[:,:4]
            dist = ((N - ref)**2).sum(dim=(-1, -2))
            idx = torch.argmin(dist)
            return N[idx:idx+1]

        raw_actions = select_best_from_N(N_actions, best_smooth_action_M)
        if self.clip_value is not None:
            raw_actions = torch.clamp(raw_actions, -self.clip_value, self.clip_value)
        
        actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
        return raw_actions, actions, {}

    @torch.no_grad()
    def negative_prompt_pcd_step(self, image, negative_instruction, instruction, proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        contrast_inputs = self.preprocess_inputs(image, negative_instruction, proprio)
       
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
    def knn_topK_motion_step(self, image, contrast_image, instruction, proprio):
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

        def cd_with_knn_topK(actions, contrast_actions, eps=1e-8):
            """
            actions: (N, T, D)
            contrast_actions: (C, T, D)

            return:
                best_action: (1, T, D)
            """
            print(f"Best-of-N KNN k = {self.knn_k}, top_k = {self.top_k}")

            N = actions.shape[0]
            C = contrast_actions.shape[0]

            A = actions.reshape(N, -1).float()  # (N, TD)
            B = contrast_actions.reshape(C, -1).float()  # (C, TD)

            # kNN in B
            dist_AB = torch.cdist(A, B, p=2) ** 2   # (N, C)
            knn_dist_AB, _ = torch.topk(dist_AB, k=self.knn_k, largest=False, dim=1)
            R_B = knn_dist_AB.sum(dim=1)  # (N,)

            # kNN in A (exclude self)
            dist_AA = torch.cdist(A, A, p=2) ** 2   # (N, N)
            inf_mask = torch.eye(N, device=A.device) * 1e9
            dist_AA = dist_AA + inf_mask
            knn_dist_AA, _ = torch.topk(dist_AA, k=self.knn_k, largest=False, dim=1)
            R_A = knn_dist_AA.sum(dim=1)  # (N,)

            # score
            scores = torch.log(R_B + eps) - torch.log(R_A + eps)

            # lấy top K
            top_scores, top_indices = torch.topk(scores, k=self.top_k, largest=True)

            top_actions = actions[top_indices]  # (top_k, T, D)

            return top_actions, top_scores
        
        top_K_best_action, _ = cd_with_knn_topK(actions, contrast_actions) # should have shape of (top_k,4,7)
        print(top_K_best_action.shape)
        best_action, jerk_rms = jerk_smoothest_action(top_K_best_action)
        print(best_action.shape)

        raw_actions = best_action
        if self.clip_value is not None:
            raw_actions = torch.clamp(raw_actions, -self.clip_value, self.clip_value)
        
        actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
        return raw_actions, actions, {}


    @torch.no_grad()
    def negative_prompt_knn_de_step(self, image, negative_instruction, instruction, proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        contrast_inputs = self.preprocess_inputs(image, negative_instruction, proprio)
       
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
    def masking_state_knn_step(self, image, instruction, proprio, contrast_proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        contrast_inputs = self.preprocess_inputs(image, instruction, contrast_proprio)
       
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
    def negative_prompt_and_inpainting_knn_step(self, image, contrast_image, negative_instruction, instruction, proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        # contrast_inputs = self.preprocess_inputs(contrast_image, instruction, proprio)
        contrast_inputs = self.preprocess_inputs(contrast_image, negative_instruction, proprio)
       
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
    def negative_prompt_plus_inpainting_knn_step(self, image, contrast_image, negative_instruction, instruction, proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        contrast_inputs_image = self.preprocess_inputs(contrast_image, instruction, proprio)
        contrast_inputs_prompt = self.preprocess_inputs(image, negative_instruction, proprio)
       
        # actions = self.forward_actions(inputs)
        # contrast_actions = self.forward_actions(contrast_inputs)
 
        # all_inputs = {}
        # for k in inputs:
        #     all_inputs[k] = torch.cat([inputs[k], contrast_inputs_image[k], contrast_inputs_prompt[k]], dim=0)
        all_inputs = {}
        for k in inputs:
            all_inputs[k] = torch.cat([inputs[k], contrast_inputs_image[k]], dim=0)
        all_actions = self.forward_actions(all_inputs)
        actions, contrast_actions_image = torch.chunk(all_actions, 2, dim=0)

        all_inputs = {}
        for k in inputs:
            all_inputs[k] = torch.cat([inputs[k], contrast_inputs_prompt[k]], dim=0)
        all_actions = self.forward_actions(all_inputs)
        _ , contrast_actions_prompt = torch.chunk(all_actions, 2, dim=0)

        # actions, contrast_actions_image, contrast_actions_prompt = torch.chunk(all_actions, 3, dim=0)
        # print("actions.shape:", actions.shape, "contrast_actions.shape:", contrast_actions.shape)

        def cd_with_knn(actions, contrast_actions_image, contrast_actions_prompt, eps=1e-8):
            """
            actions: (N, T, D) = (24, 4, 7)
            contrast_actions_image: same shape
            contrast_actions_prompt: same shape

            return:
                best_action: (1, T, D)
            """
            print("knn_k:", self.knn_k)
            N = actions.shape[0]
            A = actions.reshape(N, -1).float()              # (N, 28)
            B = contrast_actions_image.reshape(N, -1).float()     # (N, 28)
            C = contrast_actions_prompt.reshape(N, -1).float()     # (N, 28)

            # kNN in B
            dist_AB = torch.cdist(A, B, p=2) ** 2   # (N, N)
            knn_dist_AB, _ = torch.topk(dist_AB, k=self.knn_k, largest=False, dim=1)
            R_B = knn_dist_AB.sum(dim=1)  # (N,)

            # # kNN in C
            dist_AC = torch.cdist(A, C, p=2) ** 2   # (N, N)
            knn_dist_AC, _ = torch.topk(dist_AC, k=self.knn_k, largest=False, dim=1)
            R_C = knn_dist_AC.sum(dim=1)  # (N,)

            # kNN in A (exclude self)
            dist_AA = torch.cdist(A, A, p=2) ** 2   # (N, N)
            # mask diagonal (self-distance = 0)
            inf_mask = torch.eye(N, device=A.device) * 1e9
            dist_AA = dist_AA + inf_mask
            knn_dist_AA, _ = torch.topk(dist_AA, k=self.knn_k, largest=False, dim=1)
            R_A = knn_dist_AA.sum(dim=1)  # (N,)

            # best of N 
            scores = torch.log(R_B + eps) + torch.log(R_C + eps) - torch.log(R_A + eps)  # (N,)
            # scores = torch.log(R_B + eps)  - torch.log(R_A + eps)  # (N,)
            best_idx = torch.argmax(scores)
            best_action = actions[best_idx:best_idx+1]  # (1, T, D)

            return best_action
        
        raw_actions = cd_with_knn(actions, contrast_actions_image, contrast_actions_prompt) # should have shape of (1,4,7)
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

    def M_action_horizon_forward_actions(self, inputs, M_action_horizon):
        inputs.update({'num_repeats': self.num_repeats})
        inputs.update({'M_action_horizon': M_action_horizon})
        with torch.inference_mode():
            if self.use_naive:
                actions = self.model.infer_actions_naive(**inputs)
                assert 2!=2
            else:
                actions = self.model.M_action_horizon_infer_actions(**inputs)
        return actions




def jerk_smoothest_action(long_action):
    """
    long_action: torch.Tensor of shape (C, T, D)

    Returns:
        best_idx: int
        best_action: (T, D)
        jerk_rms: (C,)
    """
    assert long_action.ndim == 3
    C, T, D = long_action.shape
    assert T >= 4, "Need at least 4 timesteps to compute jerk"

    # Normalize long_action
    std = long_action.std(dim=(0,1), keepdim=True)
    norm_long_action = long_action / (std + 1e-6)

    # Compute jerk (C, T-3, D)
    jerk = (
        norm_long_action[:, 3:, :-1]
        - 3 * norm_long_action[:, 2:-1, :-1]
        + 3 * norm_long_action[:, 1:-2, :-1]
        - norm_long_action[:, :-3, :-1]
    )

    # ||j||^2 over action dim → (C, T-3)
    jerk_sq = (jerk ** 2).sum(dim=-1)
    # mean over time → (C,)
    jerk_mean = jerk_sq.mean(dim=1)
    # RMS → (C,)
    jerk_rms = torch.sqrt(jerk_mean + 1e-8)  # tránh nan
    # best candidate
    best_idx = torch.argmin(jerk_rms)
    best_action = long_action[best_idx:best_idx+1]
    return best_action, jerk_rms