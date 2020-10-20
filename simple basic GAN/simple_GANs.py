import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#batch_size
batch_size = 128

#load dataset
dataset = MNIST(root='.',transform= transforms.ToTensor(),download=True)
w,h = dataset.train_data.shape[1], dataset.train_data.shape[2]

#dataloader
dataloader = DataLoader(dataset=dataset, batch_size= batch_size)

#Discriminator Model
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()
        # self.conv1 = nn.Conv2d(3,8,kernel_size=3)
        # self.bn1 = nn.BatchNorm2d(8)
        # self.pool = nn.MaxPool2d(2)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(8,16,kernel_size=3)
        # self.bn2 = nn.BatchNorm2d(16)
        # self.conv3 = nn.Conv2d(16,32,kernel_size=3)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.conv4 = nn.Conv2d(32,64,kernel_size=3)
        # self.linear1 = nn.Linear(-1,128)
        # self.linear2 = nn.Linear(128,1)


        self.linear1 = nn.Linear(w*h,256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128,128)
        self.bn3 = nn.BatchNorm1d(128)
        self.linear4 = nn.Linear(128,1)
        self.relu = nn.ReLU(inplace=True)
    


        
    def forward(self,x):
        # x= self.conv1(x)
        # x= self.bn1(x)
        # x= self.pool(x)
        # x= self.relu(x)
        # x= self.conv2(x)
        # x= self.bn2(x)
        # x= self.pool(x)
        # x= self.relu(x)
        # x= self.conv3(x)
        # x= self.bn3(x)
        # x= self.pool(x)
        # x= self.relu(x)
        # x= self.conv4(x)
        # x= self.pool(x)
        # x= self.relu(x)
        # x= x.view(batch_size,-1)
        # x= self.linear1(x)
        # x= self.relu(x)
        # x= self.linear2(x)
        
        
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.linear4(x)

        
        return x
        


# Genearator Model
class Generator(nn.Module):
    def __init__(self,z_dim,batch_size=batch_size):
        super(Generator,self).__init__()
        
        self.batch_size = batch_size
        
        self.linear1 = nn.Linear(z_dim,128)
        self.bn1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128,128*2)
        self.bn2 = nn.BatchNorm1d(128*2)
        self.linear3 = nn.Linear(128*2,128*4)
        self.bn3 = nn.BatchNorm1d(128*4)
        self.linear4 = nn.Linear(128*4,w*h)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        
        return x
    
# noise vector
def get_noise(z_dim,batch_size= batch_size ,device='cpu'):
    
    noise=torch.randn(batch_size,z_dim,device=device)
    
    return noise

criterion = nn.BCEWithLogitsLoss()
n_epochs = 20
z_dim = 64
display_step = 50
lr = 0.00001
device = 'cpu'
generator = Generator(z_dim).to(device)

discriminator =Discriminator().to(device)

gen_optim = Adam(generator.parameters())

disc_optim = Adam(discriminator.parameters())

def disc_loss(disc,gen,real_data,criterion):

    noise = get_noise(z_dim,device='cpu')
    fake_img = gen(noise).detach()

    fake_result = disc(fake_img)
    zeros = torch.zeros_like(fake_result)
    
    fake_loss = criterion(fake_result,zeros)

    real_result = disc(real_data)
    ones = torch.ones_like(real_result)

    real_loss = criterion(real_result,ones)

    loss = (fake_loss + real_loss) / 2

    
    return loss

def gen_loss(disc,gen,criterion):
    noise = get_noise(z_dim,device='cpu')
    fake_img = gen(noise)
    fake_result = disc(fake_img)
    ones = torch.ones_like(fake_result)

    gen_loss = criterion(fake_result,ones)
    
    return gen_loss

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):

    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

for epoch in range(n_epochs):
    
    for real , _ in tqdm(dataloader,total=len(dataloader)):
        batch_size=len(real)
        
        gen_optim.zero_grad()
        disc_optim.zero_grad()
        
        real_data = real.view(batch_size,-1).to(device)
        
        disc_loss_value = disc_loss(disc=discriminator,gen= generator ,real_data= real_data ,criterion= criterion)
        disc_loss_value.backward(retain_graph=True)
        disc_optim.step()
        
        gen_loss_value= gen_loss(disc = discriminator ,gen= generator ,criterion= criterion)
        gen_loss_value.backward(retain_graph=True)
        gen_optim.step()
        
    print(f"Epoch {epoch},  Generator loss: {gen_loss_value}, discriminator loss: {disc_loss_value}")

    show_tensor_images(real_data)
    noise= get_noise(z_dim)
    fake = generator(noise)
    show_tensor_images(fake)

