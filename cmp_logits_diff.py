import torch
save_time = 0
for i in range(2):

    original_tensor = torch.load(f'logits_{i}.pt',weights_only=True)

    mod_logit_tensor = torch.load(f'refact_flash_attn_logits{i}.pt',weights_only=True)

    print("Original Tensor:", original_tensor)
    print("Original Tensor Shape:", original_tensor.shape)
    print("Modified Tensor:", mod_logit_tensor)
    print("Modified Tensor Shape:", mod_logit_tensor.shape)