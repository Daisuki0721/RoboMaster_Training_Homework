import time
from datetime import datetime
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, StepLR
import models
from data_creater import create_datasets

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# parameters
epoch = 100
total_train_step = 0
total_test_step = 0
learning_rate = 1e-2
momentum = 0.9

# defining the training device
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Training device: {device}')

# preparing dataset
# train_data = torchvision.datasets.CIFAR10(root="./dataset", train = True,
#                                           download = True, transform = dataset_transform)
# test_data = torchvision.datasets.CIFAR10(root="./dataset", train = False,
#                                           download = True, transform = dataset_transform)
train_data, test_data = create_datasets('./armor_8c_new', transform = data_transform)

# displaying length
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f'The length of train data is {train_data_size}')
print(f'The length of test data is {test_data_size}')

# loading dataset
train_dataloader = DataLoader(train_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)

# defining neural network
resnet = models.resnet34(num_classes=8)
resnet.to(device)

# defining loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# defining optimizer
optimizer = torch.optim.SGD(resnet.parameters(), lr = learning_rate, momentum = momentum)

# optimizing learning rate
warmup_epochs = 5
gamma = 0.1
# 1. Warm-up 
warmup_scheduler = LinearLR(
    optimizer,
    start_factor = 1.0 / warmup_epochs,
    end_factor = 1.0,
    total_iters = warmup_epochs
)
# 2. main
main_scheduler = StepLR(
    optimizer,
    step_size = 30,
    gamma=gamma
)
# 3. Combining two schedulers
scheduler = SequentialLR(
    optimizer,
    schedulers = [warmup_scheduler, main_scheduler],
    milestones = [warmup_epochs]  # switch when reaches warmup_epochs
)

# data writing in
writer = SummaryWriter('./logs_train')

start_time = time.time()
for i in range(epoch):
    print(f'---------------Epoch {i+1}---------------')
    print(f'Current Learning Rate: {learning_rate}')

    # training steps
    resnet.train()
    for imgs, targets in train_dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = resnet(imgs)
        loss: torch.Tensor = loss_fn(outputs, targets)

        # optimizing
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 50 == 0:
            end_time = time.time()
            print(f'{datetime.now().replace(microsecond = 0)}: training round:{total_train_step}, Used time = {end_time - start_time:.2f}s, Loss = {loss.item():.6f}')
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # adjusting learning rate
    scheduler.step()

    # testing steps
    resnet.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs: torch.Tensor  = resnet(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f'Loss on the test dataset: {total_test_loss:.6f}')
    print(f'Acc: {total_accuracy/test_data_size*100:.4f}%')
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('Accuracy', total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(resnet.state_dict(), f'models_resnet34/resnet34_{i+1}.pth')
    end_time = time.time()
    print(f'time totally used: {end_time - start_time:.2f}s')
    print('Neural Network saved.')

writer.close()
