import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


class AdaptMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdaptMLP, self).__init__()
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        down = self.down_proj(x)
        relu = self.relu(down)
        up = self.up_proj(relu)
        return up

class Adapter(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        super(Adapter, self).__init__()
        # Load a pre-trained ViT model
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)

        # Freeze the rest of the parameters in the ViT model
        for param in self.vit.parameters():
            param.requires_grad = False

        # add AdaptMLP layers in each block
        self.adapt_mlps = []
        for i, block in enumerate(self.vit.blocks): #每个Block包括self attention layernorm以及这个mlp
            # Freeze all parameters except for those in AdaptMLP
            for param in block.parameters():
                param.requires_grad = False

            self.adapt_mlps.append(AdaptMLP(input_dim=self.vit.blocks[i].mlp.fc1.in_features, hidden_dim=hidden_dim)) # hidden_dim=64

        # Replace the classifier head with a new trainable layer
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        x = self.vit.patch_embed(x)  # Apply patch embedding, (B,196,384)
        if self.vit.cls_token is not None:
            cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # (1,1,384) -> (B,1,384)
            x = torch.cat((cls_token, x), dim=1)  # (B,197,384)
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed.expand(x.shape[0], -1, -1)  # (1,197,184)  -> (B,197,384)
        x = self.vit.pos_drop(x)  # Apply dropout if present

        # Pass the input through the modified ViT model
        for i, block in enumerate(self.vit.blocks):
            # Apply the original block's layer normalization and attention
            block_input = x

            x = block.norm1(x) # (B,197,384)
            x = block.attn(x)
            x = block_input + x

            # Save the output of the original MLP
            original_mlp_output = block.norm2(x)
            original_mlp_output = block.mlp(original_mlp_output) # (B,197,384)

            # Pass the same input through the AdaptMLP
            adapt_mlp_output = self.adapt_mlps[i](x)  # (B,197,384)

            # Combine the outputs of the original MLP and AdaptMLP
            x = original_mlp_output + adapt_mlp_output

            # Finally, apply the head to get the classification output

        # Apply the final normalization and classification head
        x = self.vit.norm(x)
        if self.vit.cls_token is not None:
            x = x[:, 0] # (B,197,384) -> (B,384)
        return self.vit.head(x)



def load_cifar10_dataset():
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    return loader


def main():
    # 加载数据集
    dataset = load_cifar10_dataset()
    model = Adapter(num_classes=10, hidden_dim=64)

    for images, labels in dataset:
        logits = model(images)

        # classification loss
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        # 输出损失
        print(loss)

if __name__ == "__main__":
    main()