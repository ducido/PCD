import torch

# tạo tensor dễ nhìn
all_actions = torch.arange(48).unsqueeze(1)  # shape (4, 1)
print("all_actions:\n", all_actions)

# chunk thành 2 phần
actions, contrast_actions = torch.chunk(all_actions, 2, dim=0)

print("\nactions:\n", actions)
print("\ncontrast_actions:\n", contrast_actions)