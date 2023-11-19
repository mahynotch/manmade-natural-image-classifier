import torch
import torchvision
import torch.nn as nn
from model_test import ManNatClassifier
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

torch.seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.Resize((128, 128)), 
     transforms.ToTensor(),
     transforms.Normalize([0.46629176, 0.46785083, 0.46707144], [0.26165548, 0.2573791, 0.28735372])])


if __name__ == "__main__":
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    epoch_size = 30
    batch_size = 50
    lr = 3e-4
    print(f"Timestamp: {timestamp}, epoch_size: {epoch_size}, batch_size: {batch_size}, lr: {lr}, seed: {torch.seed}")
    print(f"device: {device}, imagesize: 256*256, type: train_test")
    loss_list = []
    accuracy_list = []
    train_set = torchvision.datasets.ImageFolder(
        root="./train",
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
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
    batch_num = len(train_set) // batch_size
    print(f"batch_num: {batch_num}, len(train_set): {len(train_set)}")
    test_data_iter = iter(test_loader)
    test_image, test_label = next(test_data_iter)
    net = ManNatClassifier()		
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    init = time.perf_counter()   
    for epoch in range(epoch_size): 
        running_loss = 0.0
        time_start = time.perf_counter()   
        for step, data in enumerate(train_loader, start=0):  
            inputs, labels = data 
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs.to(device))  	
            loss = loss_function(outputs, labels.to(device)) 
            loss.backward()
            optimizer.step() 
            running_loss += loss.item()
            if step % batch_num == batch_num - 1:
                with torch.no_grad(): 
                    net.eval()
                    outputs = net(test_image.to(device))
                    net.train() 
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                        (epoch + 1, step + 1, running_loss / batch_num, accuracy))
                    loss_list.append(running_loss / batch_num)
                    accuracy_list.append(accuracy)
                    print('%f s' % (time.perf_counter() - time_start))
                    running_loss = 0.0
    print('Finished Training')
    print(f"Training time: {time.perf_counter() - init}")                
    plt.plot(range(1, epoch_size + 1), loss_list, label="train_loss")
    plt.savefig(f"./result/{timestamp}-train-loss(test).png")
    plt.cla()
    plt.plot(range(1, epoch_size + 1), accuracy_list, label="eval_accuracy")
    plt.savefig(f"./result/{timestamp}-eval-acc(test).png")
    pd.DataFrame({"epoch":range(1, epoch_size + 1), "train_loss":loss_list, "accuracy_list":accuracy_list}).to_csv(f"./result/{timestamp}-train-loss(nopre).csv")

    save_path = f'./parameter/train_test-{timestamp}.pth'
    print(f"Parameter save path: {save_path}")
    torch.save(net.state_dict(), save_path)
    