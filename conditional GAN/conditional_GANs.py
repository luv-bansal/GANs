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
dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

# DCGAN
# Here are the main features of DCGAN :

# Use convolutions without any pooling layers
# Use batchnorm in both the generator and the discriminator
# Don't use fully connected hidden layers
# Use ReLU activation in the generator for all layers except for the output, which uses a Tanh activation.
# Use LeakyReLU activation in the discriminator for all layers except for the output, which does not use an activation

class Critics(nn.Module):
    
    def __init__(self):
        super(Critics,self).__init__()
        self.block1 = self.disc_block(input_channels = 1 ,output_channels = 8 )
        self.block2 = self.disc_block(input_channels = 8 ,output_channels = 16 )
        self.block3 = self.disc_block(input_channels = 16 ,output_channels = 32 )
        self.block4 = self.disc_block(input_channels = 32 ,output_channels = 64 )
        self.block5 = self.disc_block(input_channels = 64 ,output_channels = 128 )
        self.block6 = self.disc_block(input_channels = 128 ,output_channels = 1, final_layer= True)
        
    def disc_block(self,input_channels ,output_channels,kernel_size= 3, stride=1 , final_layer = False):
        if final_layer == False:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(2)
                )
        
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels,kernel_size=kernel_size,stride=stride),

                )
        
        
    def forward(self, x):
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        return x.view(len(x), -1)
        
# Genearator Model
class Generator(nn.Module):
    def __init__(self, z_dim , batch_size):
        super(Generator,self).__init__()
        
        self.block1 = self.gen_block( z_dim ,output_channels = 256 ,kernel_size= 4, stride=1 )
        self.block2 = self.gen_block( 256 ,output_channels = 128 ,kernel_size= 5, stride=1 )
        self.block3 = self.gen_block(128 ,output_channels = 64 ,kernel_size= 4, stride=1 )
        self.block4 = self.gen_block(64 ,output_channels = 32 ,kernel_size= 4, stride=2 )
        self.block5 = self.gen_block( 32 ,output_channels = 16 ,kernel_size= 3, stride=1 )
        self.block6 = self.gen_block(16 ,output_channels = 1 ,kernel_size= 3, stride=1, final_layer=True)

        
    def gen_block(self, input_channels ,output_channels,kernel_size= 3, stride=1 , final_layer = False):
        if final_layer == False:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
                )
        
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels,kernel_size=kernel_size,stride=stride),
                nn.Tanh()
                )
        

    def forward(self, noise):
        x = self.block1(noise )
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        return x


    
def get_noise(batch_size  , z_dim = 64, device = 'cpu'):
    
    noise = torch.randn(batch_size,z_dim , device= device )
    
    noise = noise.view(batch_size,z_dim ,1,1)
    
    return noise

criterion = nn.BCEWithLogitsLoss()
n_epochs = 2
z_dim = 64
display_step = 50
lr = 0.00001
device = 'cpu'

critics =Critics().to(device)

generator = Generator(z_dim,batch_size).to(device)


gen_optim = Adam(generator.parameters())

critics_optim = Adam(critics.parameters())

def gradient_penalty(fake, real, critics, elips):
    mix_image = fake*elips +(1-elips)*real
    fake_mix = critics(mix_image)
    
    gradient = torch.autograd.grad(fake_mix, mix_image
                                   , grad_outputs= torch.ones_like(fake_mix)
                                   , retain_graph=True
                                   , create_graph=True
                                   , allow_unused=True)[0]
    gradient = gradient.view(len(gradient), -1)
    
    gradient_norm = gradient.norm(2, dim = 1)
    
    gp = ((gradient_norm -1)**2).mean()
    
    return gp
    

def gen_loss(critics, gen):
    noise = get_noise(batch_size , z_dim=z_dim)
    fake = gen(noise)
    fake_score=critics(fake)
    
    gen_loss = - fake_score.mean()
    
    return gen_loss

def critics_loss(critics, gen, real, lamda):
    noise = get_noise(batch_size, z_dim=z_dim)
    fake = gen(noise)
    fake_score=critics(fake)
    real_score = critics(real)
    
    elips = torch.randn(len(fake),1,1,1)
    gp = gradient_penalty(fake, real, critics, elips)
    critics_loss = fake_score.mean() - real_score.mean() +lamda * gp
    
    return critics_loss

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):

    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

for epoch in tqdm(range(n_epochs),total=n_epochs):
    
    for real , _ in dataloader:
        batch_size=len(real)
        
        gen_optim.zero_grad()
        critics_optim.zero_grad()
        
        real_data = real.to(device)

        critics_loss_value = critics_loss(critics=critics, gen= generator ,real= real_data, lamda=10)
        critics_loss_value.backward(retain_graph=True)
        critics_optim.step()
        
        gen_loss_value= gen_loss(critics = critics ,gen= generator )
        gen_loss_value.backward(retain_graph=True)
        gen_optim.step()
        
    print(f"Epoch {epoch},  Generator loss: {gen_loss_value}, discriminator loss: {critics_loss_value}")

    show_tensor_images(real_data)
    noise= get_noise(z_dim)
    fake = generator(noise)
    show_tensor_images(fake)