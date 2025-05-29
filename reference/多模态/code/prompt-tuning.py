import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


class VisualPromptTuning(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_learnable_tokens=1, mode='shallow'):
        super(VisualPromptTuning, self).__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
        self.num_learnable_tokens = num_learnable_tokens
        self.mode = mode

        # Replace classifier head (just to ensure it is trainable)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

        # Create learnable tokens for shallow mode
        self.learnable_tokens = nn.Parameter(torch.randn(1, num_learnable_tokens, self.vit.embed_dim))  # (1,10,384)

        if mode == 'deep':
            # For deep mode, create learnable tokens for each block
            self.deep_learnable_tokens = nn.ParameterList([
                nn.Parameter(torch.randn(1, num_learnable_tokens, self.vit.embed_dim)) # (1,10,384)
                for _ in self.vit.blocks
            ])

        # Freeze all but the head and prompt tokens
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.vit.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.vit.patch_embed(x)

        if self.vit.cls_token is not None:
            cls_tokens = self.vit.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1) # (B,197,384)
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        if self.mode == 'shallow':
            # concatenate the learnable tokens at the beginning, deep在后面会做
            x = torch.cat((self.learnable_tokens.expand(x.size(0), -1, -1), x), dim=1) # expand: (1,10,384)->(B,10,384), cat:(B,197,384)->(B,207,384)

        for i, block in enumerate(self.vit.blocks):
            if self.mode == 'deep':
                # For deep mode, replace the learnable tokens at each block
                x[:, 1:self.num_learnable_tokens + 1] = self.deep_learnable_tokens[i] #(1,10,384)
            x = block(x)

        x = self.vit.norm(x) # (B,197,384)
        if self.vit.cls_token is not None:
            x = x[:, 0]

        return self.vit.head(x)


def load_cifar10_dataset():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    return loader


def main():
    # Load the CIFAR-10 dataset
    dataset = load_cifar10_dataset()

    # Initialize the model in 'deep' mode with a single learnable token
    model = VisualPromptTuning(num_classes=10, hidden_dim=64, num_learnable_tokens=10, mode='deep')

    for images, labels in dataset:
        logits = model(images)

        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        # Output the losses
        print(f'Loss: {loss.item()}')


if __name__ == "__main__":
    main()
