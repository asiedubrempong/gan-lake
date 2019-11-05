##################################################################################################
#                                  Generative Adversarial Nets                                   #
# Pytorch implementation of the Generative Adversarial Networks paper (Goodfellow et.al., 2014). #
# Model is trained on MNIST dataset.                                                             #
# Link to paper: https://arxiv.org/abs/1406.2661                                                 #
##################################################################################################



import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F


# check if gpu is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.ToTensor()

path = '../../data/' # path to folder that contains the data

train_dataset = datasets.MNIST(, train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# define GAN model
# discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_layer_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim*4)
        self.fc2 = nn.Linear(hidden_layer_dim*4, hidden_layer_dim*2)
        self.fc3 = nn.Linear(hidden_layer_dim*2, hidden_layer_dim)
        self.fc4 = nn.Linear(hidden_layer_dim, 1)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # matrix multiplication -> leaky relu -> dropout
        x = self.dropout(F.leaky_relu(self.fc1(x), 0.2))
        x = self.dropout(F.leaky_relu(self.fc2(x), 0.2))
        x = self.dropout(F.leaky_relu(self.fc3(x), 0.2))
        x = self.fc4(x)
        return x


# generator
class Generator(nn.Module):
    def __init__(self, z_dim, hidden_layer_dim, output_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(z_dim, hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, hidden_layer_dim*2)
        self.fc3 = nn.Linear(hidden_layer_dim*2, hidden_layer_dim*4)
        self.fc4 = nn.Linear(hidden_layer_dim*4, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # matrix multiplication -> leaky relu -> dropout
        x = self.dropout(F.leaky_relu(self.fc1(x), 0.2))
        x = self.dropout(F.leaky_relu(self.fc2(x), 0.2))
        x = self.dropout(F.leaky_relu(self.fc3(x), 0.2))
        # output uses a tanh activation function to ensure that
        # values are in the range -1 to 1
        x = torch.tanh(self.fc4(x))
        return x


d_input_dim = 28 * 28 # size of a flattened MNIST image
# size of the smallest hidden layer, other hidden layers 
# are multiples of this
hidden_layer_dim = 32 

z_dim = 100 # size of the random noise vector used to generate images
g_output_dim = 28*28

# create discriminator and generator
D = Discriminator(d_input_dim, hidden_layer_dim)
G = Generator(z_dim, hidden_layer_dim, g_output_dim)

D.to(device)
G.to(device)

# create optimizers
d_optimizer = optim.Adam(D.parameters(), lr=0.002)
g_optimizer = optim.Adam(G.parameters(), lr=0.002)

# create loss functions
def real_loss(logits, smooth=False):
    criterion = nn.BCEWithLogitsLoss()
    labels = torch.ones(*logits.shape) * 0.9 if smooth else torch.ones(*logits.shape)
    labels = labels.to(device)
    loss = criterion(logits, labels)
    return loss

def fake_loss(logits):
    criterion = nn.BCEWithLogitsLoss()
    labels = torch.zeros(*logits.shape)
    labels = labels.to(device)
    loss = criterion(logits, labels)
    return loss

samples = []
n_epochs = 100
print_every = 800

# training loop
for epoch in range(1, n_epochs+1):
    for batch_i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        # flatten the images
        real_images = real_images.view(-1, 28*28)
        batch_size = real_images.shape[0]
        
        # create a constant z vector to be used to evaluate the generator
        fixed_z = np.random.uniform(-1, 1, (batch_size, z_dim))
        fixed_z = torch.from_numpy(fixed_z).float()
        fixed_z = fixed_z.to(device)
        
        ##########################################################
        ################## Train the Discriminator ###############
        ##########################################################
        d_optimizer.zero_grad()
        
        # scale real images to be between -1 and 1
        real_images = real_images * 2 - 1
        # first pass real images from the training set
        real_logits = D(real_images)
        d_real_loss = real_loss(real_logits, smooth=True)
        
        # pass fake generated images to the discriminator
        z = np.random.uniform(-1, 1, (batch_size, z_dim))
        z = torch.from_numpy(z).float()
        z = z.to(device)
        
        fake_images = G(z)
        fake_logits = D(fake_images)
        d_fake_loss = fake_loss(fake_logits)
        
        # sum up the losses and backpropagate
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        
        ##########################################################
        ################## Train the Generator ###################
        ##########################################################
        g_optimizer.zero_grad()
        
        # generate images and pass to discriminator
        z = np.random.uniform(-1, 1, (batch_size, z_dim))
        z = torch.from_numpy(z).float()
        z = z.to(device)
        
        fake_images = G(z)
        fake_logits = D(fake_images)
        # use real loss function because generator seeks to maximize
        # the probability of the discriminator mistaking a generated 
        # image for a real image
        g_loss = real_loss(fake_logits)
        g_loss.backward()
        g_optimizer.step()
        
        # print statistics
        if batch_i % print_every == 0:
            print(f'Epoch: {epoch}/{n_epochs}  Discriminator loss: {d_loss:.3f}',
                  f'Generator loss: {g_loss:.3f}')
    else:
        losses.append((d_loss.item(), g_loss.item()))
        
        # generate and save a sample of images
        G.eval() # put model in evaluation mode
        fake_images = G(fixed_z)
        samples.append(fake_images)
        G.train() # put model back in training mode


# save generated samples
with open('generated_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)