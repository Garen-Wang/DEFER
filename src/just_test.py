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
            print(f'Now the module is {self.name_of_module[module]}, type is {module._get_name()}')
            for inp in input:
                print(f'Input shape: {inp.shape}; Output shape: {output.shape}')
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
    

model = AlexNet()

# TODO: 有了 DAG 后，确定切分点是否可以分成串行的两边，如果不可行，返回错误
# 如果可行，在中心切分模型
# 向各个部分，传送模型与权重 state_dict
# 建立 socket
# 传送 data
# 开始推理

dag_builder = CustomDAGBuilder(model)
dag_builder.build_dag()
# 必须实际跑一遍，forward hook 才会被执行
x = torch.randn(1, 3, 224, 224)
_ = model(x)
# dag_builder.print_dag()