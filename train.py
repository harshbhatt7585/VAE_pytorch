import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from model import Encoder, Decoder

# Device configuration
device = torch.device('mps')

# Hyperparameters
num_epochs = 100
learning_rate = 1e-4
beta = 0.00025  # KL divergence weight

# Data loading
transform = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 4
dataset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Add these hyperparameters
accumulation_steps = 4  # Adjust as needed
effective_batch_size = batch_size * accumulation_steps

# Modify the training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)

        # Forward pass
        reconstructed, encoded = model(images)

        # Compute loss
        recon_loss = nn.MSELoss()(reconstructed, images)

        # Extract mean and log_variance from encoded
        mean, log_variance = torch.chunk(encoded, 2, dim=1)

        kl_div = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
        loss = recon_loss + beta * kl_div

        # Normalize the loss to account for accumulation
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulation_steps

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], '
              f'Loss: {loss.item()*accumulation_steps:.4f}, Recon Loss: {recon_loss.item():.4f}, KL Div: {kl_div.item():.4f}')

        with torch.no_grad():
            # Take the first image from the batch
            sample_image = images[0].unsqueeze(0)
            sample_reconstructed = model(sample_image)[0]
            
            sample_image = (sample_image * 0.5) + 0.5
            sample_reconstructed = (sample_reconstructed * 0.5) + 0.5
            
            torchvision.utils.save_image(sample_reconstructed, 'reconstructed.png')

    # Save the model checkpoint
    torch.save(model.state_dict(), f'vae_model_epoch_{epoch+1}.pth')

print('Training finished!')
