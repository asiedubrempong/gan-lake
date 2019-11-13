# Unsupervised Representation Learning With Deep Convolutional Adversarial Networks
# 
# Trained on SVHN

# import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

import torch 
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F

train_on_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if train_on_gpu else 'cpu')

# no data augmentation is being done on the dataset
# the only transform applied is to turn the images into pytorch tensors
transform = transforms.ToTensor()

train_dataset = datasets.SVHN('../../data', split='train', transform=transform, download=True)
# batch size is set according to recommendations from the dcgan paper
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)


# Define the DCGAN model
# Generator

def frac_conv_layer(in_channels, out_channels, kernel_size, stride=2, padding=1, add_batch_norm=True):
    """
        A helper function to create a fractionally strided convolutional layer, 
        optionally followed by a batch_norm layer
    """
    layers = []
    frac_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride, padding, bias=False)
    layers.append(frac_conv)
    if add_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_size, conv_dim=32):
        super().__init__()
        
        self.conv_dim = conv_dim
        # fully connected layer to convert z vector into size that can be 
        # used for the first conv_layer
        self.fc = nn.Linear(z_size, conv_dim*6*2*2)
        
        self.frac_conv_1 = frac_conv_layer(conv_dim*6, conv_dim*4, 4, add_batch_norm=True)
        self.frac_conv_2 = frac_conv_layer(conv_dim*4, conv_dim*2, 4, add_batch_norm=True)
        self.frac_conv_3 = frac_conv_layer(conv_dim*2, conv_dim, 4, add_batch_norm=True)
        self.frac_conv_4 = frac_conv_layer(conv_dim, 3, 4, add_batch_norm=False)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.conv_dim*6, 2, 2)
        x = torch.relu(self.frac_conv_1(x))
        x = torch.relu(self.frac_conv_2(x))
        x = torch.relu(self.frac_conv_3(x))
        x = torch.tanh(self.frac_conv_4(x))
        
        return x


# Discriminator
def conv_layer(in_channels, out_channels, kernel_size, stride=2, padding=1, add_batch_norm=True):
    """
        A helper function to create a convolutional layer, 
        optionally followed by a batch_norm layer
    """
    layers = []
    conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, bias=False)
    layers.append(conv)
    if add_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, conv_dim=32):
        super().__init__()
        
        self.conv1 = conv_layer(3, conv_dim, 4, add_batch_norm=False)
        self.conv2 = conv_layer(conv_dim, conv_dim*2, 4, add_batch_norm=True)
        self.conv3 = conv_layer(conv_dim*2, conv_dim*4, 4, add_batch_norm=True)
        self.conv4 = conv_layer(conv_dim*4, conv_dim*6, 4, add_batch_norm=True)
        self.fc = nn.Linear(conv_dim*6*2*2, 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(-1, conv_dim*6*2*2)
        x = self.fc(x)
        
        return x


# Build the complete network
z_size = 100 # size of the z_vector used to generate images
conv_dim = 32

G = Generator(z_size, conv_dim)
D = Discriminator(conv_dim)
# put models on the GPU
G = G.to(device);
D = D.to(device);

# Weight initialization

def init_weights(module):
    """
    Takes in a module and initalizes all the weights with values taken from
    a zero-centered normal distribution with a standard deviation of 0.2
    
    Bias is initialized to zero
    """
    classname = module.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if hasattr(module, 'weight'):
            module.weight.data.normal_(0.0, 0.2)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)
            
G.apply(init_weights)
D.apply(init_weights)

# Define loss functions and optimizers
def real_loss(logits, smooth=True):
    criterion = nn.BCEWithLogitsLoss()
    labels = torch.ones(*logits.shape)*0.9 if smooth else torch.ones(*logits.shape)
    labels = labels.to(device)
    loss = criterion(logits, labels)
    return loss

def fake_loss(logits):
    labels = torch.zeros_like(logits)
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(logits, labels)

# use hyperparameters values recommended by the dcgan paper
lr = 0.0002
betas = (0.5, 0.999)

d_optimizer = optim.Adam(D.parameters(), lr, betas)
g_optimizer = optim.Adam(G.parameters(), lr, betas)


# Training Loop

n_epochs = 50
print_every = 400
generated_samples = []

discriminator_losses = []
generator_losses = []

# generate fixed z for evaluating the generator
fixed_z = np.random.uniform(-1, 1, size=(images.shape[0], z_size))
fixed_z = torch.from_numpy(fixed_z).float()
fixed_z = fixed_z.to(device)

for epoch in range(1, n_epochs+1):
    for idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        # scale images to be between the range -1 to 1
        real_images = real_images * 2 - 1
        
        ##############################################################
        #################   Train the Discriminator   ################
        ##############################################################
        d_optimizer.zero_grad()
        
        # pass real images to the discriminator
        real_logits = D(real_images)
        d_real_loss = real_loss(real_logits,smooth=True)
        
        # pass fake images to the discriminator 
        z = np.random.uniform(-1, 1, size=(images.shape[0], z_size))
        z = torch.from_numpy(z).float()
        z = z.to(device)
        
        fake_images = G(z)
        fake_logits = D(fake_images)
        d_fake_loss = fake_loss(fake_logits)
        
        # combine real and fake loss to get total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        # backpropagate and update the gradients of the discriminator
        d_loss.backward()
        d_optimizer.step()
        
        ###############################################################
        #################   Train the Generator    ####################
        ###############################################################
        g_optimizer.zero_grad()
        
        # generate fake images 
        z = np.random.uniform(-1, 1, size=(images.shape[0], z_size))
        z = torch.from_numpy(z).float()
        z = z.to(device)
        
        fake_images = G(z)
        
        # pass fake images to the discriminator
        fake_logits = D(fake_images)
        g_loss = real_loss(fake_logits, smooth=False)
        
        # backpropagate and update the gradients of the generator
        g_loss.backward()
        g_optimizer.step()
        
        if idx % print_every == 0:
            discriminator_losses.append(d_loss.item())
            generator_losses.append(g_loss.item())
            # print statistics
            print(f'Epoch: {epoch}/{n_epochs}     Discriminator loss: {d_loss: .3f}',
                  f'    Generator loss: {g_loss: .3f}')
    else: 
        # generate and store sample images for this epoch
        G.eval() # put G in eval mode
        fake_images = G(fixed_z)
        generated_samples.append(fake_images)
        G.train() # put G in train mode


# save samples
with open('saved_samples/svhn_samples.pkl', 'wb') as f:
    pkl.dump(generated_samples, f)

rows = 5
cols = 10

fig, ax = plt.subplots(figsize=(20, 15), nrows=rows, ncols=cols, sharex=True, sharey=True)

for epoch_sample, ax_row in zip(generated_samples[::int(len(generated_samples)/rows)], ax):
    for img, ax_col in zip(epoch_sample[::int(len(epoch_sample)/cols)], ax_row):
        img = img.cpu().detach() # detach tensor from its history
        img = img.numpy() # convert to numpy array
        img = img.transpose(1, 2, 0)
        img = ((img + 1) * 255 / (2)).astype(np.uint8) # rescale to 0-255
        ax_col.imshow(img)
        ax_col.xaxis.set_visible(False)
        ax_col.yaxis.set_visible(False)
