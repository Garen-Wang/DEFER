"""
这部分代码是用来生成 resnet50 早退的 4 个子模型的，同时验证分割正常
"""
import os
from typing import List
import torch
import torch.nn as nn
from resnet50_early_exit import ResNet50WithEarlyExit, Bottleneck


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ResNet50WithEarlyExit(Bottleneck, [3, 4, 6, 3]).to(device)
model.load_state_dict(torch.load('./checkpoints/early_100-2023-10-24_025128.pth')['model_state_dict'])

# 如果我们分成了 self.layers 和 self.exits
# 是不是就可以有统一的划分 model 的方法了呢？
def split_model(model: nn.Module):
    main_models = []
    exit_models = []
    assert len(model.layers) == len(model.exits) + 1

    main_models.append(nn.Sequential(model.layers[0], model.layers[1]))
    exit_models.append(model.exits[0])
    main_models.append(model.layers[2])
    exit_models.append(model.exits[1])
    main_models.append(model.layers[3])
    exit_models.append(model.exits[2])
    main_models.append(model.layers[4])
    exit_models.append(model.exits[3])

    ret = []
    for idx, (main_model, exit_model) in enumerate(zip(main_models, exit_models)):
        # SUB_MODEL_PATH = './sub_models/resnet50'
        # torch.jit.script(main_model).save(os.path.join(SUB_MODEL_PATH, f'main_model_{idx}.pth'))
        # torch.jit.script(exit_model).save(os.path.join(SUB_MODEL_PATH, f'exit_model_{idx}.pth'))
        ret.append((main_model, exit_model))
    return ret
# device = 'cpu'

x = torch.randn(1, 3, 32, 32).to(device)
y = model(x)
print(y[-1])

temp = x
test_ys = []
sub_models = split_model(model)
# sub_models = []
# SUB_MODEL_PATH = './sub_models/resnet50'
# for i in range(4):
#     sub_models.append((
#         torch.jit.load(os.path.join(SUB_MODEL_PATH, f'main_model_{i}.pth')).to(device),
#         torch.jit.load(os.path.join(SUB_MODEL_PATH, f'exit_model_{i}.pth')).to(device),
#     ))

for (main_model, exit_model) in sub_models:
    temp = main_model(temp)
    test_ys.append(exit_model(temp))

print(test_ys[-1])
print(torch.eq(y[-1], test_ys[-1]))