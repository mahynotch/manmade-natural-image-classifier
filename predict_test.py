import torch
import torchvision.transforms as transforms
from PIL import Image
from model_test import ManNatClassifier
import os
import time

DIR = "./test/manmade_test"
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}, DIR: {DIR}, pretrained: {False}")
transform = transforms.Compose(
    [transforms.Resize((256, 256)), 
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
net = ManNatClassifier()
Para = "/root/man_nat_classify/parameter/train-2023-11-11 11:24:04.pth"
net.load_state_dict(torch.load(Para))
print(f"Using: {Para}")
net.eval()
start = time.perf_counter()
classes = ("manmade", "natural")
files = os.listdir(DIR)
man_cnt = 0
nat_cnt = 0
man_set = []
nat_set = []
for i in files:
    im = Image.open(DIR + "/" + i)
    im = transform(im) 
    im = torch.unsqueeze(im, dim=0)  
    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    if  int(predict[0]) == 0:
        man_cnt += 1
        man_set.append(i)
    else:
        nat_cnt += 1
        nat_set.append(i)
print(f"manmade: {man_cnt}, natural: {nat_cnt}")
print(f"Time: {time.perf_counter()-start}")
print(f"anomaly: {man_set if man_cnt < nat_cnt else nat_set}")
print(f"accuracy: {man_cnt/(man_cnt+nat_cnt) if man_cnt > nat_cnt else nat_cnt/(man_cnt+nat_cnt)}")
