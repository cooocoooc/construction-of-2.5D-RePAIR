
import math
import torch
import torch.nn as nn

# position embedding
class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb
    
class PatchEncoder(nn.Module):

    def __init__(self, in_channel = 4, embed_dim = 128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding = 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride = 2, padding = 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride = 2, padding =1), nn.ReLU(),
            nn.Conv2d(256, embed_dim, 3, stride = 2, padding =1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x):
        # x: [B*N, C, H, W] or [B, N, C, H, W]
        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
        out_put = self.convs(x).squeeze(-1).squeeze(-1) # [B*N, embed_dim]
        return out_put

class FragmentAggregator(nn.Module):

    def __init__(self, embed_dim = 128, num_heads = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first = True)

    def forward(self, patch_features, patch_frag_idx, mask_patch, num_frag):
        # patch_features:[B. N, D], patch_frag_idx:[B, N]
        B, N, D = patch_features.shape
        aggregator_list = []
        for batch in range(B):
            frag_features = []
            for frag_id in range(num_frag):
                mask = (patch_frag_idx[batch] == frag_id) & mask_patch[batch]
                features = patch_features[batch][mask]
                if features.shape[0] == 0:
                    features = torch.zeros(1, D, device = features.device)
                else:
                    attn_out, _ = self.attn(features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0))
                    features = attn_out.mean(dim = 1) #[D]
                frag_features.append(features.squeeze(0))
                
            aggregator_list.append(torch.stack(frag_features, dim = 0)) #[num_frag, D]
        
        return torch.stack(aggregator_list, dim = 0)      

class CoarseDenoiser(nn.Module):

    def __init__(self, feature_dim = 128, dim_model = 256, num_heads = 4, num_layers = 2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, dim_model)
        )
        self.proj_feature = nn.Linear(feature_dim, dim_model)
        self.proj_coord = nn.Linear(2, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model = dim_model, nhead = num_heads, batch_first = True)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
        self.head = nn.Linear(dim_model, 2 + 4) # [B, M, 2] position noise + [B, M, 4] angle logits

    def forward(self, noisy_coords, frag_features, t):
        # noisy_pos:[B, M, 2], frag_features:[B, M, D]
        B, M, _ = noisy_coords.shape
        t_emb = self.time_mlp(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, M, -1)
        h = self.proj_feature(frag_features) + self.proj_coord(noisy_coords) + t_emb
        h = self.trans(h)
        out_put = self.head(h)  # [B, M, 6]
        coord_noise = out_put[:, :, :2]  # [B, M, 2]
        angle_logits = out_put[:, :, 2:]  # [B, M, 4]
        return coord_noise, angle_logits

class FineDenoiser(nn.Module):

    def __init__(self, feature_dim = 128, dim_model = 128, num_heads = 4, num_layers = 2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, dim_model)
        )
        self.proj_feature = nn.Linear(feature_dim, dim_model)
        self.proj_coord = nn.Linear(2, dim_model)
        self.proj_offset = nn.Linear(2, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model = dim_model, nhead = num_heads, batch_first = True)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
        self.head = nn.Linear(dim_model, 2)
        
    def forward(self, noisy_offsets, patch_features, coarse_centers, t):
        # noisy_offset:[B, N, 2], patch_features:[B, N, D], coarse_centers:[B. N, 2]
        B, N, _ = noisy_offsets.shape
        t_emb = self.time_mlp(t) # [B, d_model]
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)
        h = self.proj_feature(patch_features) + self.proj_coord(coarse_centers) + self.proj_offset(noisy_offsets) + t_emb
        h = self.trans(h)
        offset_noise = self.head(h)
        return offset_noise

class DiffusionScheduler:

    def __init__(self, T = 100, beta_start = 1e-4, beta_end = 0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0)

    def sqrt_sample(self, x0, t, noise = None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

    def sample_posterior(self, x_t, pred_noise, t):
        # DDPM sample
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        alpha = self.alphas[t].view(-1, 1, 1)
        beta = self.betas[t].view(-1, 1, 1)
        if t[0] == 0:
            return (x_t - beta * pred_noise / torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha)
        else:
            mean = (x_t - beta * pred_noise / torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(beta) * noise
            