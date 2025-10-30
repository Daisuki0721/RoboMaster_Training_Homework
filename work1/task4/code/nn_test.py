import os
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'test_set'

result_dict = {'ImageID':[], 'label':[]}

result = pd.DataFrame(result_dict)

model = models.resnet50(num_classes = 54).to('cuda:0')
model.load_state_dict(torch.load('models_resnet50/resnet_00090.pth', map_location='cuda'))
print(model)

data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.ToTensor()
])

model.eval()

for filename in os.listdir(data_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(data_dir, filename)
        
    org_image = Image.open(img_path).convert('RGB')
    image: torch.Tensor = data_transform(org_image)     # type: ignore
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output: torch.Tensor = model(image)

    predicted_class = pd.DataFrame({'ImageID':filename, 'label':output.argmax(1).item()})

    result = pd.concat([result, predicted_class], ignore_index = True)

result.sort_values(by = 'ImageID', ascending = True).to_csv('result.csv', index = False)
print('Result output completed.')
