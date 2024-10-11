#importing the libraries
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Transform to normalize the data and convert it to a tensor
transform  = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])

#load the dataset
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transform, download = True)
train_loader = DataLoader(train_dataset, batch_size= 64, shuffle = True)

for x , y in train_loader:
    print(x.shape)
    print(y.shape)
    break

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__() # we are generator which will generate random image
        self.model = nn.Sequential(
            nn.Linear(100, 256), 
            nn.LeakyReLU(0.2), # here we are using LeakyReLU  activation functio so that some neagtive values can also pass through it.
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28*28),
            nn.Tanh()  # We are using the tan function to see values are from -1 or 1
        )

    def forward(self, x):
        x = self.model(x) # this step forwards x input in the model
        x = x.view(-1, 1, 28, 28)
        return x
    
# Initialize the models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
lr = 0.0002
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function (Binary Cross Entropy)
criterion = nn.BCELoss()

num_epochs = 50
for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device) # here we are labeling the real images as ones
        fake_labels = torch.zeros(batch_size, 1).to(device) # here we are labeling the fake images as zeros

        ### Train Discriminator ###
        optimizer_D.zero_grad()

        # Real images
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels) # here we are measuring the loss

        # Fake images
        z = torch.randn(batch_size, 100).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        # Total loss and backpropagation
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        ### Train Generator ###
        optimizer_G.zero_grad()

        # Generator wants the discriminator to believe that the fake images are real
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)  # We use real_labels here!

        # Backpropagation
        g_loss.backward()
        optimizer_G.step()

    print(f'Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}')

import numpy as np

# Generate samples
z = torch.randn(16, 100).to(device)
fake_images = generator(z)

# Plot the images
fake_images = fake_images.cpu().detach().numpy()
fake_images = np.squeeze(fake_images)

plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(fake_images[i], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()