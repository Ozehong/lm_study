import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Student and Teacher networks using a Vision Transformer (ViT) architecture
class ViT(nn.Module):
    def __init__(self, output_dim):
        super(ViT, self).__init__()
        # Use a ViT model from timm
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=output_dim)

    def forward(self, x):
        return self.vit(x)

# Data loading and augmentation
def get_dataloader(batch_size):
    transform = Compose([
        Resize(224),  # Resize the image to 224x224
        RandomCrop(224, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CIFAR10(root="./cifar10", train=True, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize student and teacher networks
output_dim = 128  # Example output dimension
gs = ViT(output_dim).to(device)
gt = ViT(output_dim).to(device)
gt.load_state_dict(gs.state_dict())  # Teacher network initially copies the student weights

# Initialize center
C = torch.zeros(output_dim, device=device) # vit输出的特征是128，(128)

# DINO-specific loss function H
def H(t, s, C, tps, tpt):
    t = t.detach()  # stop gradient for teacher
    s = F.softmax(s / tps, dim=1) # (B,128)
    t = F.softmax((t - C) / tpt, dim=1) # (B,128)
    return - (t * torch.log(s)).sum(dim=1).mean()

# Mock function for augmentations (in a real scenario we would use more complex augmentations)
def augment(x):
    return x + 0.1 * torch.randn_like(x)

# Set up hyperparameters
batch_size = 64
tps = 0.1  # Temperature for student softmax
tpt = 0.07  # Temperature for teacher softmax
l = 0.6  # Network momentum rate
m = 0.5  # Center momentum rate

# Optimizer for the student network
optimizer = optim.SGD(gs.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)

# Training loop
loader = get_dataloader(batch_size)
for x, _ in loader:  # We don't need labels for self-supervised learning
    x = x.to(device) # (B,3,224,224)
    x1, x2 = augment(x), augment(x)  # random views
    s1, s2 = gs(x1), gs(x2)  # student output, (B,128)
    t1, t2 = gt(x1), gt(x2)  # teacher output, (B,128)
    loss = H(t1, s2, C, tps, tpt)/2 + H(t2, s1, C, tps, tpt)/2 # divide by 2 for combined loss
    loss.backward()  # back-propagate
    optimizer.step()  # SGD update for student
    optimizer.zero_grad()  # Clear gradients

    # Teacher and center updates
    with torch.no_grad():
        for teacher_param, student_param in zip(gt.parameters(), gs.parameters()):
            teacher_param.data = l * teacher_param.data + (1 - l) * student_param.data
        C = m * C + (1 - m) * torch.cat([t1, t2]).mean(dim=0)

print("Training complete!")