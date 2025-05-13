import math
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

# Helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature=10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# LoRA Layer
class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Module, r: int, alpha: int):
        """
        Args:
            original_layer (nn.Module): 原始的线性层 (e.g., nn.Linear)
            r (int): 低秩的维度
            alpha (int): LoRA 的缩放因子
        """
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        if isinstance(original_layer, nn.Linear):
            self.W_A = nn.Linear(original_layer.in_features, r, bias=False)
            self.W_B = nn.Linear(r, original_layer.out_features, bias=False)
        else:
            raise ValueError("Unsupported layer type for LoRA")

        # 初始化 W_A 和 W_B
        nn.init.kaiming_uniform_(self.W_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_B.weight)

    def forward(self, x):
        return self.original_layer(x) + self.scaling * self.W_B(self.W_A(x))
    
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class AttentionWithLoRA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, r=4, alpha=16):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        # 使用 LoRA 替换 to_qkv 和 to_out
        self.to_qkv = LoRALayer(nn.Linear(dim, inner_dim * 3, bias=False), r=r, alpha=alpha)
        self.to_out = LoRALayer(nn.Linear(inner_dim, dim, bias=False), r=r, alpha=alpha)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# FeedForward Class
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

# Transformer with LoRA
class TransformerWithLoRA(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, r=4, alpha=16, lora_layers=None):
        """
        Args:
            dim (int): 输入特征维度
            depth (int): Transformer 层数
            heads (int): Attention 的头数
            dim_head (int): 每个头的特征维度
            mlp_dim (int): FeedForward 的隐藏层维度
            r (int): LoRA 的秩
            alpha (int): LoRA 的缩放因子
            lora_layers (list): 指定哪些层使用 LoRA（基于索引，0 开始）
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        lora_layers = lora_layers or []  # 如果为空，默认所有层都不用 LoRA

        for i in range(depth):
            if i in lora_layers:
                # 使用带 LoRA 的 Attention
                attn_layer = AttentionWithLoRA(dim, heads=heads, dim_head=dim_head, r=r, alpha=alpha)
            else:
                # 使用普通的 Attention
                attn_layer = Attention(dim, heads=heads, dim_head=dim_head)

            self.layers.append(nn.ModuleList([
                attn_layer,
                FeedForward(dim, mlp_dim)
            ]))
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.final_norm(x)
        return x

class TransformerWithSelectiveLoRA(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, r=4, alpha=16, lora_layers=None):
        """
        Args:
            dim (int): 输入特征维度
            depth (int): Transformer 层数
            heads (int): Attention 的头数
            dim_head (int): 每个头的特征维度
            mlp_dim (int): FeedForward 的隐藏层维度
            r (int): LoRA 的秩
            alpha (int): LoRA 的缩放因子
            lora_layers (list): 指定哪些层使用 LoRA（基于索引，0 开始）
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        lora_layers = lora_layers or []  # 如果为空，默认所有层都不用 LoRA

        for i in range(depth):
            if i in lora_layers:
                # 使用带 LoRA 的 Attention
                attn_layer = AttentionWithLoRA(dim, heads=heads, dim_head=dim_head, r=r, alpha=alpha)
            else:
                # 使用普通的 Attention
                attn_layer = Attention(dim, heads=heads, dim_head=dim_head)

            self.layers.append(nn.ModuleList([
                attn_layer,
                FeedForward(dim, mlp_dim)
            ]))
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.final_norm(x)
        return x


# Original Transformer Class (for reference)
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# SimpleViT Class
class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)

# LoRA_SimpleViT Class
class LoRA_SimpleViT(SimpleViT):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, r=4, alpha=16, lora_layers=None):
        """
        Args:
            lora_layers (list): 指定哪些 Transformer 层使用 LoRA（基于索引，0 开始）
        """
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dim_head=dim_head
        )

        # 使用修改后的 TransformerWithSelectiveLoRA
        self.transformer = TransformerWithSelectiveLoRA(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            r=r,
            alpha=alpha,
            lora_layers=lora_layers  # 指定哪些层使用 LoRA
        )

        # 冻结所有参数，除 LoRA 的 W_A 和 W_B
        self.freeze_parameters(lora_layers)

    def freeze_parameters(self, lora_layers):
        """
        冻结使用 LoRA 的 Transformer 层的原始参数，仅训练 LoRA 的 W_A 和 W_B 参数。
        未使用 LoRA 的 Transformer 层保持所有参数可训练。

        Args:
            lora_layers (list): 指定哪些层使用 LoRA（基于索引，0 开始）
        """
        for name, param in self.named_parameters():
            if 'transformer.layers' in name:
                # 提取当前层的索引
                layer_idx = int(name.split('.')[2])
                if layer_idx in lora_layers:
                    # 使用 LoRA 的层
                    if 'W_A' in name or 'W_B' in name:
                        param.requires_grad = True  # 保持 LoRA 参数可训练
                    else:
                        param.requires_grad = False  # 冻结原始参数
                else:
                    # 未使用 LoRA 的层，保持所有参数可训练
                    param.requires_grad = True
            else:
                # 其他参数保持可训练
                param.requires_grad = True

# Functions to Save and Load LoRA Parameters
def save_lora_parameters(model, filepath):
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'W_A' in name or 'W_B' in name:
            lora_state_dict[name] = param.cpu()
    torch.save(lora_state_dict, filepath)

def load_lora_parameters(model, filepath, device='cpu'):
    lora_state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(lora_state_dict, strict=False)
