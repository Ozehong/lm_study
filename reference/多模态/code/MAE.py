import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed


def mse_loss(reconstructed_patches, patches, mask_indices):
    """
        Parameters:
        - reconstructed_patches: Tensor of shape (B, 196, 768), reconstructed patch embeddings.
        - patches: Tensor of shape (B, 196, 768), original patch embeddings.
        - mask_indices: LongTensor of shape (196,), indices of the masked patches.
    """
    B = reconstructed_patches.size(0)

    # Only consider the masked patches, gather函数用来从patches中选择那些被mask的patches出来
    # 虽然有多个batch，但每个batch上都是那些位置的元素被mask，我们先把每个batch的mask_indices都扩展成(B,196,1)的形状，然后再用gather函数
    # 得到（B，147,768）
    masked_original = torch.gather(patches, 1,
                                   mask_indices[None, :, None].expand(B, -1, patches.size(2)))
    masked_reconstructed = torch.gather(reconstructed_patches, 1,
                                   mask_indices[None, :, None].expand(B, -1, reconstructed_patches.size(2)))

    # Calculate the squared differences
    loss = (masked_original - masked_reconstructed) ** 2 # (B,147,768)

    # Calculate the mean over all dimensions
    loss = loss.mean()

    return loss


class MAE(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=768, mask_ratio=0.75):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        # patch embbding，用于切patch
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Encoder (sequence of blocks)
        self.encoder = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=12, mlp_ratio=4.0) for _ in range(12)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Mask token is learned during training
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size ** 2 * 3)  # To predict the RGB values of the masked patches
        )

    def forward(self, imgs):
        B, C, H, W = imgs.shape # (B, 3, 224, 224)

        # Extract patches
        patches = self.patch_embed(imgs) # (B,196,768)

        # Add positional encoding to patch embeddings
        x = patches + self.pos_embed.repeat(B, 1, 1) # self.pos_embed的batch是1，要变成batch

        # Masking
        N = x.shape[1] # patch的数量，196
        num_masked = int(self.mask_ratio * N)  # 147
        all_indices = torch.randperm(N, device=x.device) # 把196个patch的数组随机打乱，返回一个list，每个元素都是一个打乱后的下标
        mask_indices = all_indices[:num_masked]  # 选取前147个patch的下标
        mask = torch.ones((N,), device=x.device) # (196)的全1向量
        mask[mask_indices] = 0

        # input unmasked tokens to the encoder
        unmasked_tokens = x.clone() # (B,196,768)
        unmasked_tokens[:, mask_indices] = self.mask_token # 147个patch的位置替换为mask_token
        for blk in self.encoder:
            unmasked_tokens = blk(unmasked_tokens)
        unmasked_tokens = self.norm(unmasked_tokens) # (B,196,768)

        # Now we combine the masked and unmasked tokens for the decoder
        encoded_patches = unmasked_tokens

        # Add positional encodings for decoding
        encoded_patches += self.pos_embed.repeat(B, 1, 1) # (B,196,768)

        # Decode each token to reconstruct the patches
        reconstructed_patches = self.decoder(encoded_patches) # (B,196,768)

        return reconstructed_patches, patches, mask_indices # 重建patch，原始patch，mask的下标


# Example usage:
if __name__ == '__main__':
    img = torch.rand(2, 3, 224, 224)  # Example image batch
    mae_model = MAE()
    reconstructed_patches, patches, mask_indices = mae_model(img)
    loss = mse_loss(reconstructed_patches, patches, mask_indices)
    print(loss)