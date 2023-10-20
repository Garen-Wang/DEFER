import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict

class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cl1 = nn.Linear(25, 60)
        self.cl2 = nn.Linear(60, 84)
        self.fc1 = nn.Linear(84, 10)

    def forward(self):
        x = F.relu(self.cl1(x))
        x = F.relu(self.cl2(x))
        return self.fc1(x)

# model = MyModel()
# print([n for n, _ in model.named_children()])

model = torchvision.models.vgg16(pretrained=True)

def nested_children(model: nn.Module):
    children = dict(model.named_children())
    output = {}
    if children == {}:
        return model
    else:
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)

    return output

name_children = nested_children(model)
name_layers = OrderedDict(model.named_modules())

print(name_layers)
# print(name_children['classifier']['0'].state_dict())