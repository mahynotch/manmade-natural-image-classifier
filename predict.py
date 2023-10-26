import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ManNatClassifier
import os

DIR = "./crawler_output/My Little Pony/"
transform = transforms.Compose(
    [transforms.Resize((256, 256)), 
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
net = ManNatClassifier()
net.load_state_dict(torch.load("./parameter/FirstAttempt.pth"))
classes = ("manmade", "natural")
files = os.listdir(DIR)
for i in files:
    im = Image.open(DIR + "/" + i)
    im = transform(im) 
    im = torch.unsqueeze(im, dim=0)  
    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    print(classes[int(predict)], end=" ")