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
    [transforms.Resize([256, 256]),
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
    test_image, test_label = next(test_data_iter)
    net = ManNatClassifier()		
    net.to(device)
    loss_function = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(net.parameters(), lr=0.001)  
    for epoch in range(5): 
        running_loss = 0.0
        time_start = time.perf_counter()   
        for step, data in enumerate(train_loader, start=0):  
            inputs, labels = data 
            optimizer.zero_grad()  
            print(inputs.shape)
            # forward + backward + optimize
            outputs = net(inputs.to(device))  	
            print(outputs.shape)
            print(labels.shape)
            loss = loss_function(outputs, labels.to(device)) 
            loss.backward()
            optimizer.step() 
            running_loss += loss.item()
            if step % 20 == 19:
                with torch.no_grad(): 
                    outputs = net(test_image.to(device)) 
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0)
                    
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                        (epoch + 1, step + 1, running_loss / 500, accuracy))
                    
                    print('%f s' % (time.perf_counter() - time_start))
                    running_loss = 0.0

    print('Finished Training')

    # 保存训练得到的参数
    save_path = './parameter/FirstAttempt.pth'
    torch.save(net.state_dict(), save_path)
    