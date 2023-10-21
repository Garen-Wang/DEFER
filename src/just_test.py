from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.alexnet import AlexNet
from collections import defaultdict, OrderedDict

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

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class DAG:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

class CustomDAGBuilder:
    def __init__(self, model):
        self.model = model
        self.dag = DAG()
        self.module_by_name = {}
        self.name_of_module = {}
        self._init_modules(model)

    def _init_modules(self, model):
        name_children = nested_children(model)
        self.module_by_name = flatten_dict(name_children)
        # 把所有实际的层拿出来了，同时保有了顺序
        for name, module in self.module_by_name.items():
            self.name_of_module[module] = name

    def register_hooks(self, module):
        def forward_hook(module, input, output):
            # Add edges to the DAG
            # print(f'Now the module is {self.name_of_module[module]}, type is {module._get_name()}')
            for inp in input:
                # print(f'Input shape: {inp.shape}; Output shape: {output.shape}')
                self.dag.add_edge(inp, output)

        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == self.model):
            module.register_forward_hook(forward_hook)

    def build_dag(self):
        self.model.apply(self.register_hooks)

    def print_dag(self):
        for layer, connections in self.dag.graph.items():
            # print(f'{layer} -> {connections}')
            for connection in connections:
                print(f'{layer.shape} -> {connection.shape}')
    

class MyAlexNet(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        # WARNING: 必须保证你写的这些东西，是与forward函数对应的，如果顺序不对应，那就完了
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # 不要用这个，线性可分的话，就意味着forward函数不能生成
        # out = out.view(out.size(0), -1)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = MyAlexNet()

# TODO: 有了 DAG 后，确定切分点是否可以分成串行的两边，如果不可行，返回错误
# 如果可行，在中心切分模型
# 向各个部分，传送模型与权重 state_dict
# 建立 socket
# 传送 data
# 开始推理

dag_builder = CustomDAGBuilder(model)
dag_builder.build_dag()
# 必须实际跑一遍，forward hook 才会被执行
x = torch.randn(1, 3, 227, 227)
model.eval()
original_inference_result = model(x)

def split_model(model: nn.Module, names: List[str]) -> List[nn.Module]:
    part_module_names = [[] for _ in range(len(names) + 1)]
    part_modules = [] # 分界点的那个 module 的名字
    cnt = 0
    for name, m in model.named_children():
        print('[DEBUG] name of named_children: ', name)
        part_module_names[cnt].append(name)
        if cnt != len(names) and name == names[cnt]:
            cnt += 1
            part_modules.append(m)
    
    if cnt != len(names):
        raise TypeError(f'part_module_names list has wrong length')
    
    part_modules_idx = []
    cnt = 0
    for idx, (name, m) in enumerate(model.named_children()):
        if m is part_modules[cnt]:
            cnt += 1
            part_modules_idx.append(idx)
            if cnt == len(part_modules):
                break
    
    if len(part_modules_idx) != len(part_modules):
        raise TypeError(f'Module name found, but module index not found in the model')

    sub_models = []
    sub_models.append(nn.Sequential(*list(model.children())[:part_modules_idx[0]]))
    for i in range(1, len(part_modules_idx)):
        sub_models.append(nn.Sequential(*list(model.children())[part_modules_idx[i-1]:part_modules_idx[i]]))
    sub_models.append(nn.Sequential(*list(model.children())[part_modules_idx[-1]:]))

    # copy weights to sub models
    original_state_dict = model.state_dict()
    sub_model_state_dicts = []
    for i in range(len(sub_models)):
        sub_model_state_dicts.append({key: value for key, value in original_state_dict.items() if any(key.startswith(prefix) for prefix in part_module_names[i])})

    assert len(sub_model_state_dicts) == len(sub_models)
    for i in range(len(sub_models)):
        sub_models[i].load_state_dict(sub_model_state_dicts[i], strict=False)

    # assert
    # for i in range(min(len(sub_model1.state_dict().items()), len(model.state_dict().items()))):
    #     original_model_state_dict = list(model.state_dict().items())[i][1]
    #     sub_model_state_dict = list(sub_model1.state_dict().items())[i][1]
    #     assert original_model_state_dict.equal(sub_model_state_dict)
    
    # for i in range(len(sub_model2.state_dict().items())):
    #     original_model_state_dict = list(model.state_dict().items())[i + len(sub_model1.state_dict().items())][1]
    #     sub_model_state_dict = list(sub_model2.state_dict().items())[i][1]
    #     assert original_model_state_dict.equal(sub_model_state_dict)

    return sub_models

split_module_names = ['layer3', 'flatten', 'fc1']
sub_models = split_model(model, split_module_names)
for sub_model in sub_models:
    sub_model.eval()

def model_parts_inference(x):
    for model_part in sub_models:
        x = model_part(x)
    return x

# 取一个左开右闭

test_inference_result = model_parts_inference(x)
print(original_inference_result)
print(test_inference_result)
print(model(x))