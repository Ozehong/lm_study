import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

# 数据加载
transform = ToTensor()
dataset = CIFAR10(root="./cifar10", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# 使用ResNet50
def get_resnet50(output_dim):
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, output_dim)
    return model


# InfoNCE Loss
def info_nce_loss(q, k, queue, temperature=0.07):
    q = nn.functional.normalize(q, dim=1, p=2)
    k = nn.functional.normalize(k, dim=1, p=2)
    queue = nn.functional.normalize(queue, dim=0, p=2)

    positive_similarity = torch.bmm(q.view(N,1,C), k.view(N,C,1))

    negative_similarity = torch.mm(q, queue)

    logits = torch.cat([positive_similarity.squeeze(-1), negative_similarity], dim=-1)

    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

    loss = nn.CrossEntropyLoss()(logits / temperature, labels)
    return loss



# 参数设定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C = 1024
N = loader.batch_size
K = 4096
f_q = get_resnet50(C).to(device)
f_k = get_resnet50(C).to(device)
f_k.load_state_dict(f_q.state_dict())

m = 0.99
queue = torch.randn(C, K).to(device)
queue_ptr = 0

optimizer = optim.Adam(f_q.parameters(), lr=0.001)



# 模拟数据增强函数
def aug(x):
    return x + 0.1 * torch.randn_like(x)


# 主循环
for x, _ in loader:
    x = x.to(device)
    x_q = aug(x)
    x_k = aug(x)

    q = f_q(x_q)
    k = f_k(x_k)

    k = k.detach()

    loss = info_nce_loss(q, k, queue)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        for param_q, param_k in zip(f_q.parameters(), f_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

        # Update the keys queue
        batch_size = k.size(0)
        queue[:, queue_ptr:queue_ptr + batch_size] = k.T
        queue_ptr = (queue_ptr + batch_size) % K

print("Training complete!")
