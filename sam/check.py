import torch
import matplotlib.pyplot as plt
import numpy as np

# load .pth file
checkpoint = torch.load('./checkpoints/best_dice_checkpoint.pth')
checkpoint_original = torch.load('./checkpoints/sam2.1_hiera_base_plus.pt')

# print the keys in the checkpoint
print("Keys in the checkpoint:", checkpoint.keys())  # 如果是字典，打印所有键
print("Keys in the checkpoint_original:", checkpoint_original.keys())  # 如果是字典，打印所有键

# state_dict = checkpoint['state_dict']
# # 打印权重层及其维度
# for key, value in state_dict.items():
#     print(f"{key}: {value.shape}")  # 输出层名和对应的权重形状

state_dict_original = checkpoint_original['model']
# 打印权重层及其维度
for key, value in state_dict_original.items():
    print(f"{key}: {value.shape}")  # 输出层名和对应的权重形状


# 假设可视化某个卷积层的权重
# layer_name = 'image_encoder.pos_embed'  # 替换为实际的层名
# weights = state_dict[layer_name].cpu().numpy()

# # 可视化第一个卷积核
# plt.imshow(weights[0, 0], cmap='viridis')
# plt.colorbar()
# plt.title(f"Weights of {layer_name}")
# plt.show()


# # 打印 epoch 值
# print("Epoch:", checkpoint['epoch'])

# # 打印 model
# print("model:", checkpoint['model'])

# # 查看 state_dict 的键
# print("State_dict keys:", checkpoint['state_dict'].keys())

# # 打印 optimizer 的结构
# print("Optimizer keys:", checkpoint['optimizer'].keys())

# # 打印 best_tol 和 path_helper
# print("Best_tol:", checkpoint['best_tol'])
# print("Path_helper:", checkpoint['path_helper'])


