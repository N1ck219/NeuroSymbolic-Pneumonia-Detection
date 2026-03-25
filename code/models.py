# --- IMPORTS ---
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchxrayvision as xrv
except ImportError:
    pass

# --- UTILS & ATTENTION MODULES ---
class AddCoords(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1).float() / (x_dim - 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2).float() / (y_dim - 1)
        xx_channel = (xx_channel * 2 - 1).repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = (yy_channel * 2 - 1).repeat(batch_size, 1, 1, 1).transpose(2, 3)
        return torch.cat([input_tensor, xx_channel.to(input_tensor.device), yy_channel.to(input_tensor.device)], dim=1)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv1(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# --- CENN FRONTEND ---
class CeNNFrontEnd2DG(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3, K=16, alpha_init=0.1):
        super().__init__()
        self.K = K
        self.out_channels = out_channels
        padding = kernel_size // 2
        
        self.template_B = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.template_A = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
        val = max(0.01, min(0.99, 1.0 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(val) - math.log(1.0 - val)))

    def forward(self, u):
        control = self.template_B(u)
        x = control.clone()
        alpha = torch.sigmoid(self.alpha_logit)
        beta = 1.0 - alpha
        
        c_idx = self.template_A.kernel_size[0] // 2
        mask = torch.zeros_like(self.template_A.weight, dtype=torch.bool)
        mask[range(self.out_channels), range(self.out_channels), c_idx, c_idx] = True
        weight_A = torch.where(mask, torch.clamp(self.template_A.weight, min=1.0), self.template_A.weight)

        for _ in range(self.K):
            x_act = torch.tanh(x)
            feedback = F.conv2d(x_act, weight_A, padding=self.template_A.padding)
            x = alpha * x + beta * (feedback + control + self.bias)
            
        return x

# --- MAIN ARCHITECTURE ---
class HighResPneumoniaDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cenn = CeNNFrontEnd2DG(in_channels=1, out_channels=3, K=8)
        
        try:
            self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all").features
            old_conv = self.backbone.conv0
            self.backbone.conv0 = nn.Conv2d(3, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None))
        except:
            import torchvision.models as models
            dn = models.densenet121(weights='DEFAULT')
            old_conv = dn.features.conv0
            dn.features.conv0 = nn.Conv2d(3, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None))
            self.backbone = dn.features

        self.common_cbam = CBAM(1024)
        
        self.cls_neck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.cls_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
        
        self.reg_neck = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.add_coords = AddCoords()
        self.reg_head = nn.Sequential(
            nn.Conv2d(130, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 24 * 24, 256), nn.ReLU(), nn.Linear(256, 4) 
        )
        self.pool = nn.AdaptiveAvgPool2d((24, 24))

    def forward(self, x):
        x = self.cenn(x)
        feat = self.backbone(x)
        feat = self.common_cbam(feat)
        
        c_feat = self.cls_neck(feat)
        c_avg = self.avg_pool(c_feat).flatten(1)
        c_max = self.max_pool(c_feat).flatten(1)
        cls_out = self.cls_head(torch.cat([c_avg, c_max], dim=1))
        
        r_feat = self.reg_neck(feat)
        r_feat = self.add_coords(r_feat)
        r_feat = self.pool(r_feat)   
        reg_out = torch.sigmoid(self.reg_head(r_feat))
        
        return cls_out, reg_out

class NoCeNNPneumoniaDetector(nn.Module):
    """
    Ablation Model: Architettura identica alla proposta originale ma SENZA
    il frontend CeNN. L'immagine in scala di grigi entra direttamente nel backbone.
    """
    def __init__(self):
        super().__init__()
        # NESSUN FRONTEND CeNN!
        
        try:
            self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all").features
            old_conv = self.backbone.conv0
            # Adattiamo conv0 per ricevere 1 canale anziché 3 (visto che manca la CeNN)
            self.backbone.conv0 = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None))
        except:
            import torchvision.models as models
            dn = models.densenet121(weights='DEFAULT')
            old_conv = dn.features.conv0
            dn.features.conv0 = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None))
            self.backbone = dn.features

        self.common_cbam = CBAM(1024)
        
        self.cls_neck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.cls_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
        
        self.reg_neck = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.add_coords = AddCoords()
        self.reg_head = nn.Sequential(
            nn.Conv2d(130, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 24 * 24, 256), nn.ReLU(), nn.Linear(256, 4) 
        )
        self.pool = nn.AdaptiveAvgPool2d((24, 24))

    def forward(self, x):
        # L'input x entra DIRETTAMENTE nel backbone (Nessun x = self.cenn(x))
        feat = self.backbone(x)
        feat = self.common_cbam(feat)
        
        c_feat = self.cls_neck(feat)
        c_avg = self.avg_pool(c_feat).flatten(1)
        c_max = self.max_pool(c_feat).flatten(1)
        cls_out = self.cls_head(torch.cat([c_avg, c_max], dim=1))
        
        r_feat = self.reg_neck(feat)
        r_feat = self.add_coords(r_feat)
        r_feat = self.pool(r_feat)   
        reg_out = torch.sigmoid(self.reg_head(r_feat))
        
        return cls_out, reg_out