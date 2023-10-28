import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ManNatClassifier
import os

DIR = "./crawler_output/Statue of liberty"
transform = transforms.Compose(
    [transforms.Resize((512, 512)), 
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
net = ManNatClassifier()
net.load_state_dict(torch.load("./parameter/ThirdAttempt(ASGD).pth"))
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
print(f"anomaly: {man_set if man_cnt < nat_cnt else nat_set}")
