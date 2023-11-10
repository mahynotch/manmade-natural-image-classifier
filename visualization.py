import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
import argparse
from torchvision import models, transforms

# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image')
args = vars(ap.parse_args())

# Load the ResNet-50 Model
net = models.resnet.ResNet(models.resnet.Bottleneck, [3, 4, 6, 3], 2)
net.load_state_dict(torch.load("./parameter/NinthAttempt.pth"))

model_weights = []  # save the conv layer weights
conv_layers = []  # save the 49 conv layers
model_children = list(model.children())
# print(model_childern)

# counter: keep count of the conv layers
counter = 0

# 将所有卷积层以及相应权重加入到两个空list中
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)

print(f'Total convolutional layers: {counter}')
# for weight, conv in zip(model_weights, conv_layers):
#     print(f'CONV: {conv} ====> SHAPE: {weight.shape}')

# 可视化first conv layer filters
plt.figure(figsize=(20, 17))
for i, filter in enumerate(model_weights[0]):
    plt.subplot(8, 8, i+1)  # conv0: 卷积核大小7*7，共有64个
    plt.imshow(filter[0, :, :].detach(), cmap='gray')
    plt.axis('off')
    plt.savefig('../outputs/filter.png')
plt.show()

# 可视化图像
img = cv.imread(f"../input/cat.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

img = np.array(img)  # Image->ndarray
img = transform(img)  # tensor
print(img.size())  # [3, 512, 512]
img = img.unsqueeze(0)    # 增加一个维度，表明在这个batch只有一张图片
print(img.size())  # [1, 3, 512, 512]

# pass the image through all the layers
results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    results.append(conv_layers[i](results[-1]))

outputs = results

# Visualize 64 feature maps from each layer
for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64:
            break
        plt.subplot(8, 8, i+1)
        plt.imshow(filter, cmap='gray')
        plt.axis('off')
    print(f'Saving layer {num_layer} feature maps ... ')
    plt.savefig(f'../outputs/layer_{num_layer}.png')
    plt.close()

