import torch
import torchvision
import torch.nn as nn
from model import ManNatClassifier
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.Resize([512, 512]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


if __name__ == "__main__":
    train_set = torchvision.datasets.ImageFolder(
        root="./train",
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=50,
        shuffle=True,
        num_workers=0
    )
    test_set = torchvision.datasets.ImageFolder(
        root="./test",
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(test_set, 
		batch_size=100,
		shuffle=False,
        num_workers=0)
    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.__next__()
    net = ManNatClassifier()		
    net.to(device)
    loss_function = nn.CrossEntropyLoss() 				# 定义损失函数为交叉熵损失函数 
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器（训练参数，学习率）
    for epoch in range(5):  # 一个epoch即对整个训练集进行一次训练
        running_loss = 0.0
        time_start = time.perf_counter()
        
        for step, data in enumerate(train_loader, start=0):   # 遍历训练集，step从0开始计算
            inputs, labels = data 	# 获取训练集的图像和标签
            optimizer.zero_grad()   # 清除历史梯度
            
            # forward + backward + optimize
            outputs = net(inputs.to(device))  				  # 正向传播
            loss = loss_function(outputs, labels.to(device)) # 计算损失
            loss.backward() 					  # 反向传播
            optimizer.step() 					  # 优化器更新参数

            # 打印耗时、损失、准确率等数据
            running_loss += loss.item()
            if step % 20 == 19:
                with torch.no_grad(): 
                    outputs = net(test_image.to(device)) 
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0)
                    
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
                        (epoch + 1, step + 1, running_loss / 500, accuracy))
                    
                    print('%f s' % (time.perf_counter() - time_start))        # 打印耗时
                    running_loss = 0.0

    print('Finished Training')

    # 保存训练得到的参数
    save_path = './parameter/FirstAttempt.pth'
    torch.save(net.state_dict(), save_path)
    