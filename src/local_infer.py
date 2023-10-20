# For benchmarking against DEFER, try this file that uses Single Device Inference

import time
import torch
import torchvision
from torchvision import transforms
from PIL import Image

# pytorch...

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.vgg19(pretrained=True)
model = model.to(device)

def load_image(image_path, shape=None, transform=None):
    img = Image.open(image_path)
    if shape is not None:
        img = img.resize(shape, Image.LANCZOS)
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    img = transform(img).unsqueeze(0)
    return img.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229,0.224, 0.225))
])
img = load_image('../images/dog.png', shape=[224, 224], transform=transform)

time_run = 1
in_sec = time_run * 60
start = time.time()
result_count = 0

model.eval()
while (time.time() - start) < in_sec:
    temp = model(img)
    # print(temp.shape)
    result_count += 1
print(f"In {time_run} min, {result_count} results")
# In 1 min, 263 results