# CELL 1 — Environment Setup
!pip install -q ultralytics>=8.4.0 pycocotools scipy opencv-python-headless

import torch, sys, os
print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
assert torch.cuda.is_available(), "GPU not detected – switch to a GPU runtime."
DEVICE = torch.device("cuda:0")
print(f"Using {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

# CELL 2 — Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2, math, time, random, glob, yaml, copy, json, warnings
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# CELL 3 — Model Configuration
class CFG:
    """Central config — tweak channels / depths to hit ≈100 M params."""
    nc            = 80
    img_size      = 640
    channels      = [128, 256, 512, 768, 1024]
    backbone_depths = [3, 6, 6, 3]
    neck_depths   = [3, 3, 3, 3]
    strides       = [8, 16, 32]
    max_det       = 300
    # Training
    epochs        = 100
    batch_size    = 8
    lr0           = 0.01
    lrf           = 0.01
    momentum      = 0.937
    weight_decay  = 5e-4
    warmup_epochs = 3
    warmup_momentum = 0.8
    warmup_bias_lr = 0.1
    # Loss
    box_weight    = 7.5
    cls_weight    = 0.5
    # ProgLoss / STAL
    progloss      = True
    stal           = True
    stal_area_thr = 32**2
    # TAL
    tal_topk      = 13
    tal_alpha     = 1.0
    tal_beta      = 6.0

# CELL 4 — Core Building Blocks
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p
class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU (standard YOLO conv block)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              dilation=d, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU(inplace=True) if act is True else (
                    act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
class DWConv(Conv):
    """Depth-wise convolution."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
class Bottleneck(nn.Module):
    """Standard residual bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (fast C3 variant)."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k2(C2f):
    """C3k2 — the core block used in YOLO11/YOLO26 backbone & neck.
    When `c3k=True`, each slot is a full C3 block (heavier);
    when `c3k=False`, it is a plain Bottleneck (lighter)."""
    def __init__(self, c1, c2, n=1, c3k=False, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        if c3k:
            self.m = nn.ModuleList(
                C3(self.c, self.c, n=2, shortcut=shortcut, g=g, k=3)
                for _ in range(n))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling — Fast."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_ * 4, c2, 1)
        self.m = nn.MaxPool2d(k, stride=1, padding=k // 2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x); y2 = self.m(y1); y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class Attention(nn.Module):
    """Multi-head self-attention on 2-D feature maps (YOLO-style PSA)."""
    def __init__(self, dim, num_heads=4, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.key_dim   = max(int(self.head_dim * attn_ratio), 16)
        self.scale     = self.key_dim ** -0.5
        h = dim + self.key_dim * num_heads * 2       # Q + K + V sizes
        self.qkv  = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe   = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).view(B, self.num_heads,
                                self.key_dim * 2 + self.head_dim, N)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(B, C, H, W)
        out = out + self.pe(v.reshape(B, C, H, W))
        return self.proj(out)

class PSABlock(nn.Module):
    """Position-Sensitive Attention block (attention + FFN)."""
    def __init__(self, c, num_heads=4, attn_ratio=0.5):
        super().__init__()
        self.attn = Attention(c, num_heads, attn_ratio)
        self.ffn  = nn.Sequential(Conv(c, c * 2, 1),
                                  Conv(c * 2, c, 1, act=False))
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class C2PSA(nn.Module):
    """CSP with PSA (applied only at P5 to keep cost low)."""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        nh = max(self.c // 64, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, num_heads=nh) for _ in range(n)))
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat([a, b], 1))

# CELL 5 — Backbone

class Backbone(nn.Module):
    """CSP-Darknet-style backbone with C3k2, SPPF, C2PSA.
    Returns multi-scale features at P3, P4, P5."""
    def __init__(self, ch=CFG.channels, depths=CFG.backbone_depths):
        super().__init__()
        # Stem
        self.stem0 = Conv(3, ch[0], 3, 2)             # /2  → P1
        self.stem1 = Conv(ch[0], ch[1], 3, 2)         # /4  → P2
        # Stage 2
        self.stage2 = C3k2(ch[1], ch[1], n=depths[0], c3k=False, shortcut=True)
        # Stage 3
        self.down3  = Conv(ch[1], ch[2], 3, 2)        # /8  → P3
        self.stage3 = C3k2(ch[2], ch[2], n=depths[1], c3k=False, shortcut=True)
        # Stage 4
        self.down4  = Conv(ch[2], ch[3], 3, 2)        # /16 → P4
        self.stage4 = C3k2(ch[3], ch[3], n=depths[2], c3k=True, shortcut=True)
        # Stage 5
        self.down5  = Conv(ch[3], ch[4], 3, 2)        # /32 → P5
        self.stage5 = C3k2(ch[4], ch[4], n=depths[3], c3k=True, shortcut=True)
        self.sppf   = SPPF(ch[4], ch[4], k=5)
        self.psa    = C2PSA(ch[4], ch[4], n=2)
    def forward(self, x):
        x = self.stem1(self.stem0(x))
        x = self.stage2(x)
        p3 = self.stage3(self.down3(x))   # stride 8
        p4 = self.stage4(self.down4(p3))  # stride 16
        p5 = self.psa(self.sppf(self.stage5(self.down5(p4))))  # stride 32
        return p3, p4, p5

# CELL 6 — Neck (PANet FPN)

class Neck(nn.Module):
    """PANet feature-pyramid neck: top-down + bottom-up paths."""
    def __init__(self, ch=CFG.channels, depths=CFG.neck_depths):
        super().__init__()
        c3, c4, c5 = ch[2], ch[3], ch[4]   # 512, 768, 1024
        # --- top-down ---
        self.up5   = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_c4 = C3k2(c5 + c4, c4, n=depths[0], c3k=False)
        self.up4   = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_c3 = C3k2(c4 + c3, c3, n=depths[1], c3k=False)
        # --- bottom-up ---
        self.down3 = Conv(c3, c3, 3, 2)
        self.bu_c4 = C3k2(c3 + c4, c4, n=depths[2], c3k=False)
        self.down4 = Conv(c4, c4, 3, 2)
        self.bu_c5 = C3k2(c4 + c5, c5, n=depths[3], c3k=True)
    def forward(self, p3, p4, p5):
        # top-down
        td4 = self.td_c4(torch.cat([self.up5(p5), p4], 1))
        td3 = self.td_c3(torch.cat([self.up4(td4), p3], 1))
        # bottom-up
        bu4 = self.bu_c4(torch.cat([self.down3(td3), td4], 1))
        bu5 = self.bu_c5(torch.cat([self.down4(bu4), p5], 1))
        return td3, bu4, bu5  # channels: c3, c4, c5

# CELL 7 — Dual Detection Head (NMS-free, no DFL)
class DualDetectHead(nn.Module):
    """YOLO26-style dual head — one-to-many (o2m) + one-to-one (o2o).
    No Distribution Focal Loss — direct 4-value box regression."""
    def __init__(self, nc=CFG.nc, ch=(512, 768, 1024)):
        super().__init__()
        self.nc, self.nl = nc, len(ch)
        self.no = nc + 4
        def _make_branch(ch_list):
            cv2, cv3 = nn.ModuleList(), nn.ModuleList()
            for c in ch_list:
                c2 = max(c // 4, 64)
                c3 = max(c // 4, min(nc * 2, 256))
                cv2.append(nn.Sequential(
                    Conv(c, c2, 3), Conv(c2, c2, 3),
                    nn.Conv2d(c2, 4, 1)))
                cv3.append(nn.Sequential(
                    Conv(c, c3, 3), Conv(c3, c3, 3),
                    nn.Conv2d(c3, nc, 1)))
            return cv2, cv3
        self.o2m_cv2, self.o2m_cv3 = _make_branch(ch)
        self.o2o_cv2, self.o2o_cv3 = _make_branch(ch)
        self.strides = torch.tensor(CFG.strides, dtype=torch.float32)
        self._init_bias()
    def _init_bias(self):
        """Initialize biases (prior for class probability ~0.01)."""
        for cv3_list in (self.o2m_cv3, self.o2o_cv3):
            for m in cv3_list:
                b = m[-1].bias.view(-1)
                b.data.fill_(-math.log((1 - 0.01) / 0.01))
    @staticmethod
    def _split(feats, cv2, cv3):
        boxes, clss = [], []
        for i, f in enumerate(feats):
            boxes.append(cv2[i](f))
            clss.append(cv3[i](f))
        return boxes, clss
    def forward(self, feats):
        """feats: list of [P3, P4, P5] feature maps.
        Returns dict with 'o2m' and 'o2o', each containing
        (box_raw, cls_raw) as flat tensors (B, ?, N)."""
        o2m_box, o2m_cls = self._split(feats, self.o2m_cv2, self.o2m_cv3)
        o2o_box, o2o_cls = self._split(feats, self.o2o_cv2, self.o2o_cv3)
        def _flatten(box_list, cls_list):
            b_flat = torch.cat([b.flatten(2) for b in box_list], 2)  # (B,4,N)
            c_flat = torch.cat([c.flatten(2) for c in cls_list], 2)  # (B,nc,N)
            return b_flat, c_flat
        return {
            "o2m": _flatten(o2m_box, o2m_cls),
            "o2o": _flatten(o2o_box, o2o_cls),
            "shapes": [(f.shape[2], f.shape[3]) for f in feats],
        }

# CELL 8 — Full Model Assembly + Param Count
class MyYOLO26(nn.Module):
    """Full YOLO26-inspired object detector."""
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg  = cfg
        self.backbone = Backbone(cfg.channels, cfg.backbone_depths)
        self.neck     = Neck(cfg.channels, cfg.neck_depths)
        head_ch = (cfg.channels[2], cfg.channels[3], cfg.channels[4])
        self.head     = DualDetectHead(cfg.nc, head_ch)
    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        f3, f4, f5 = self.neck(p3, p4, p5)
        return self.head([f3, f4, f5])
    @torch.no_grad()
    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params : {total/1e6:,.1f} M")
        print(f"Trainable   : {train/1e6:,.1f} M")
        return total
# — quick sanity check —
_m = MyYOLO26().to(DEVICE)
_m.count_params()
_x = torch.randn(1, 3, 640, 640, device=DEVICE)
_o = _m(_x)
print(f"   o2m box : {_o['o2m'][0].shape}  cls : {_o['o2m'][1].shape}")
print(f"   o2o box : {_o['o2o'][0].shape}  cls : {_o['o2o'][1].shape}")
del _m, _x, _o; torch.cuda.empty_cache()

# CELL 9 — Box Utilities
def make_anchors(shapes, strides, offset=0.5):
    """Generate anchor centres & stride tensors for all feature levels."""
    anchors, st = [], []
    for (h, w), s in zip(shapes, strides):
        sy, sx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32) + offset,
            torch.arange(w, dtype=torch.float32) + offset,
            indexing="ij")
        anchors.append(torch.stack([sx, sy], -1).view(-1, 2) * s)  # pixel coords
        st.append(torch.full((h * w, 1), s, dtype=torch.float32))
    return torch.cat(anchors, 0), torch.cat(st, 0)
def dist2bbox(dist, anchors):
    """distance (l,t,r,b) + anchor → xyxy."""
    lt, rb = dist.split(2, dim=-1)
    x1y1 = anchors - lt
    x2y2 = anchors + rb
    return torch.cat([x1y1, x2y2], -1)
def bbox2dist(anchors, bbox):
    """xyxy box → distances from anchor."""
    x1y1, x2y2 = bbox.split(2, dim=-1)
    lt = anchors - x1y1
    rb = x2y2 - anchors
    return torch.cat([lt, rb], -1)
def bbox_iou(box1, box2, xywh=False, CIoU=True, eps=1e-7):
    """Compute IoU / CIoU between two sets of boxes (N,4)."""
    if xywh:
        b1x1, b1y1 = box1[..., :2] - box1[..., 2:] / 2, None
        # convert to xyxy …
        pass  # not used — all our boxes are xyxy
    b1x1, b1y1, b1x2, b1y2 = box1.unbind(-1)
    b2x1, b2y1, b2x2, b2y2 = box2.unbind(-1)
    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * \
            (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    a1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    a2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    union = a1 + a2 - inter + eps
    iou = inter / union
    if CIoU:
        cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
        ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
        c2 = cw ** 2 + ch ** 2 + eps
        rho2 = ((b1x1 + b1x2 - b2x1 - b2x2) ** 2 +
                (b1y1 + b1y2 - b2y1 - b2y2) ** 2) / 4
        w1, h1 = b1x2 - b1x1, b1y2 - b1y1
        w2, h2 = b2x2 - b2x1, b2y2 - b2y1
        v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) -
                                   torch.atan(w1 / (h1 + eps))) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        return iou - (rho2 / c2 + v * alpha)
    return iou

# CELL 10 — Task-Aligned Assigner (TAL) + Hungarian Matcher
class HungarianMatcher:
    """One-to-one assignment using the Hungarian algorithm (for o2o head)."""

    def __init__(self, cost_cls=1.0, cost_box=5.0, cost_iou=2.0, eps=1e-8):
        self.cost_cls = cost_cls
        self.cost_box = cost_box
        self.cost_iou = cost_iou
        self.eps = eps
    @torch.no_grad()
    def __call__(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_gt, tasks=None):
        """
        Args:
            pred_scores : (B, N, nc) raw logits or probabilities
            pred_bboxes : (B, N, 4) predicted boxes (x1,y1,x2,y2)
            gt_labels   : (B, M)
            gt_bboxes   : (B, M, 4)
            mask_gt     : (B, M)
            tasks       : optional, ignored (for API compatibility with loop)
        Returns:
            target_labels : (B, N)
            target_bboxes : (B, N, 4)
            target_scores : (B, N, nc)
            fg_mask       : (B, N)
        """
        B, N, nc = pred_scores.shape
        M = gt_labels.shape[1]
        device = pred_scores.device
        dtype = pred_scores.dtype
        target_labels = torch.zeros(B, N, dtype=torch.long, device=device)
        target_bboxes = torch.zeros(B, N, 4, device=device, dtype=dtype)
        target_scores = torch.zeros(B, N, nc, device=device, dtype=dtype)
        fg_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        if M == 0:
            return target_labels, target_bboxes, target_scores, fg_mask
        for b in range(B):
            n_gt = int(mask_gt[b].sum().item())
            if n_gt == 0:
                continue
            gt_lb = gt_labels[b, :n_gt]
            gt_bb = gt_bboxes[b, :n_gt]
            ps_raw = pred_scores[b]
            ps = ps_raw.sigmoid() if (ps_raw.min() < 0 or ps_raw.max() > 1.0) else ps_raw
            ps = ps.clamp(min=self.eps, max=1.0 - self.eps)
            cost_cls = -ps[:, gt_lb].T.log()
            cost_box = torch.cdist(gt_bb.float(), pred_bboxes[b].float(), p=1)
            ious = bbox_iou(gt_bb, pred_bboxes[b])
            cost_iou = 1.0 - ious
            C = self.cost_cls*cost_cls + self.cost_box*cost_box + self.cost_iou*cost_iou
            row_idx, col_idx = linear_sum_assignment(C.cpu().numpy())
            for r, c in zip(row_idx, col_idx):
                fg_mask[b, c] = True
                target_labels[b, c] = gt_lb[r]
                target_bboxes[b, c] = gt_bb[r]
                target_scores[b, c, gt_lb[r]] = 1.0

        return target_labels, target_bboxes, target_scores, fg_mask

# CELL 11 — Loss Function (CIoU, ProgLoss, STAL)
class YOLO26Loss(nn.Module):
    """Combined loss with ProgLoss scheduling + STAL weighting."""
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.tal = TaskAlignedAssigner(cfg.tal_topk, cfg.tal_alpha, cfg.tal_beta)
        self.hungarian = HungarianMatcher()
    def _head_loss(self, box_raw, cls_raw, anchors, strides,
                   gt_labels, gt_bboxes, mask_gt, assigner, stal_weight=None):
        """Compute loss for one head (o2m or o2o).
        box_raw : (B,4,N)  cls_raw : (B,nc,N)"""
        B, _, N = box_raw.shape
        nc = cls_raw.shape[1]
        device = box_raw.device
        # Decode predictions
        box_pred = box_raw.permute(0, 2, 1)               # (B,N,4)
        cls_pred = cls_raw.permute(0, 2, 1)               # (B,N,nc)
        # Decode boxes: distances → xyxy
        anchors_dev = anchors.to(device)                   # (N,2)
        strides_dev = strides.to(device)                   # (N,1)
        pred_dist   = box_pred * strides_dev.squeeze(-1).unsqueeze(0).unsqueeze(-1).expand(B, N, 1)
        # Actually simpler: multiply raw by stride
        pred_boxes  = dist2bbox(box_pred * strides_dev.T.unsqueeze(0).expand(B, -1, -1).permute(0,2,1),
                                anchors_dev.unsqueeze(0).expand(B, -1, -1))
        # Simpler approach:
        pred_boxes = dist2bbox(
            box_pred * strides_dev.squeeze(-1)[None, :, None],
            anchors_dev[None].expand(B, -1, -1))           # (B,N,4)
        # Assignment
        with torch.no_grad():
            pred_scores_detach = cls_pred.detach().sigmoid()
            pred_bboxes_detach = pred_boxes.detach()
            tgt_labels, tgt_bboxes, tgt_scores, fg = assigner(
                pred_scores_detach, pred_bboxes_detach, anchors_dev,
                gt_labels, gt_bboxes, mask_gt)
        n_pos = fg.sum().clamp(min=1).float()
        # --- Box loss (CIoU) ---
        if fg.sum() > 0:
            iou = bbox_iou(pred_boxes[fg], tgt_bboxes[fg], CIoU=True)
            # STAL: boost loss for small targets
            if stal_weight is not None and self.cfg.stal:
                w = tgt_bboxes[fg][:, 2] - tgt_bboxes[fg][:, 0]
                h = tgt_bboxes[fg][:, 3] - tgt_bboxes[fg][:, 1]
                area = w * h
                sw = torch.where(area < self.cfg.stal_area_thr,
                                 torch.tensor(2.0, device=device),
                                 torch.tensor(1.0, device=device))
                box_loss = ((1.0 - iou) * sw).sum() / n_pos
            else:
                box_loss = (1.0 - iou).sum() / n_pos
        else:
            box_loss = torch.tensor(0.0, device=device)
        # --- Cls loss (BCE) ---
        cls_loss = self.bce(cls_pred, tgt_scores).sum() / n_pos
        return box_loss, cls_loss, n_pos
    def forward(self, outputs, gt_labels, gt_bboxes, mask_gt, epoch=0, total_epochs=1):
        """
        outputs   : dict from model forward
        gt_labels : (B, M) int
        gt_bboxes : (B, M, 4) xyxy pixel
        mask_gt   : (B, M) bool
        """
        shapes  = outputs["shapes"]
        anchors, strides = make_anchors(shapes, CFG.strides)
        anchors = anchors.to(gt_bboxes.device)
        strides = strides.to(gt_bboxes.device)

        # One-to-many loss
        o2m_box, o2m_cls = outputs["o2m"]
        box_l_m, cls_l_m, _ = self._head_loss(
            o2m_box, o2m_cls, anchors, strides,
            gt_labels, gt_bboxes, mask_gt, self.tal, stal_weight=True)
        # One-to-one loss
        o2o_box, o2o_cls = outputs["o2o"]
        box_l_o, cls_l_o, _ = self._head_loss(
            o2o_box, o2o_cls, anchors, strides,
            gt_labels, gt_bboxes, mask_gt, self.hungarian, stal_weight=True)
        # ProgLoss scheduling
        if self.cfg.progloss and total_epochs > 1:
            progress = min(epoch / total_epochs, 1.0)
            lam_m = 1.0 - 0.5 * progress   # 1.0 → 0.5
            lam_o = 0.5 * progress + 0.5    # 0.5 → 1.0
        else:
            lam_m, lam_o = 1.0, 1.0
        loss_box = lam_m * box_l_m + lam_o * box_l_o
        loss_cls = lam_m * cls_l_m + lam_o * cls_l_o
        total = self.cfg.box_weight * loss_box + self.cfg.cls_weight * loss_cls
        return total, loss_box.detach(), loss_cls.detach()

# CELL 12 — MuSGD Optimizer
class MuSGD(torch.optim.Optimizer):
    """Hybrid SGD + Muon-style gradient normalization.
    Blends classical momentum SGD with sign-normalised updates
    inspired by the Muon optimizer from LLM training."""
    def __init__(self, params, lr=0.01, momentum=0.937,
                 weight_decay=5e-4, muon_strength=0.05, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        muon_strength=muon_strength, nesterov=nesterov)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr  = group["lr"]
            mom = group["momentum"]
            wd  = group["weight_decay"]
            mu  = group["muon_strength"]
            nes = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd:
                    g = g.add(p, alpha=wd)
                state = self.state[p]
                if len(state) == 0:
                    state["buf"] = torch.zeros_like(p)
                    state["v"]   = torch.zeros_like(p)
                buf = state["buf"]
                v   = state["v"]
                # EMA of squared grad (Muon-style normalisation)
                v.mul_(0.999).addcmul_(g, g, value=1 - 0.999)
                muon_g = g / (v.sqrt() + 1e-8)
                # Blend SGD gradient with Muon-normalised gradient
                blended = (1 - mu) * g + mu * muon_g
                buf.mul_(mom).add_(blended)
                if nes:
                    p.add_(blended + mom * buf, alpha=-lr)
                else:
                    p.add_(buf, alpha=-lr)
        return loss

# CELL 13 — Dataset & DataLoader
def xywhn2xyxy(x, w, h):
    """Convert normalised xywh → pixel xyxy."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] - x[..., 2] / 2) * w
    y[..., 1] = (x[..., 1] - x[..., 3] / 2) * h
    y[..., 2] = (x[..., 0] + x[..., 2] / 2) * w
    y[..., 3] = (x[..., 1] + x[..., 3] / 2) * h
    return y
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize + pad image to square."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(w * r), int(h * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape - nw, new_shape - nh
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)
class YOLODataset(Dataset):
    """Reads YOLO-format datasets (images + labels/*.txt)."""
    def __init__(self, img_dir, lbl_dir, img_size=640, augment=True):
        self.img_size = img_size
        self.augment  = augment
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        self.img_files = sorted(
            [f for ext in exts for f in glob.glob(os.path.join(img_dir, ext))])
        self.lbl_dir = lbl_dir
        print(f"{len(self.img_files)} images  |  labels: {lbl_dir}")
    def __len__(self):
        return len(self.img_files)
    def _load_label(self, idx):
        stem = Path(self.img_files[idx]).stem
        lbl_path = os.path.join(self.lbl_dir, stem + ".txt")
        if os.path.isfile(lbl_path):
            with open(lbl_path) as f:
                labels = np.array([x.split() for x in f.read().strip().splitlines()],
                                  dtype=np.float32)
            if len(labels) == 0:
                return np.zeros((0, 5), dtype=np.float32)
            return labels  # (M, 5): cls cx cy w h (normalised)
        return np.zeros((0, 5), dtype=np.float32)
    def __getitem__(self, idx):
        img = cv2.imread(self.img_files[idx])
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        # Handle grayscale / multispectral
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        oh, ow = img.shape[:2]
        labels = self._load_label(idx)
        # Augmentations
        if self.augment:
            if random.random() < 0.5:
                img = img[:, ::-1, :]
                if len(labels):
                    labels[:, 1] = 1.0 - labels[:, 1]
            # HSV augment
            if random.random() < 0.5:
                h_gain, s_gain, v_gain = 0.015, 0.7, 0.4
                r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
                hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
                hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        img, ratio, (pad_w, pad_h) = letterbox(img, self.img_size)
        # Convert labels from normalised xywh → pixel xyxy (post-letterbox)
        nl = len(labels)
        targets = np.zeros((nl, 5), dtype=np.float32)  # cls, x1, y1, x2, y2
        if nl:
            targets[:, 0] = labels[:, 0]
            bboxes = labels[:, 1:5].copy()
            # Scale to original pixels then apply letterbox transform
            bboxes[:, 0] *= ow; bboxes[:, 2] *= ow
            bboxes[:, 1] *= oh; bboxes[:, 3] *= oh
            # xywh → xyxy
            x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            targets[:, 1] = (x - w / 2) * ratio + pad_w
            targets[:, 2] = (y - h / 2) * ratio + pad_h
            targets[:, 3] = (x + w / 2) * ratio + pad_w
            targets[:, 4] = (y + h / 2) * ratio + pad_h
            # Clip
            targets[:, 1:5] = np.clip(targets[:, 1:5], 0, self.img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(img.copy()), torch.from_numpy(targets)
def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    max_t = max(t.shape[0] for t in targets)
    B = len(targets)
    gt_labels = torch.zeros(B, max_t, dtype=torch.long)
    gt_bboxes = torch.zeros(B, max_t, 4)
    mask_gt   = torch.zeros(B, max_t, dtype=torch.bool)
    for i, t in enumerate(targets):
        n = t.shape[0]
        if n > 0:
            gt_labels[i, :n] = t[:, 0].long()
            gt_bboxes[i, :n] = t[:, 1:5]
            mask_gt[i, :n]   = True
    return imgs, gt_labels, gt_bboxes, mask_gt

# CELL 14 — Download Datasets via Ultralytics
def get_dataset_paths(data_yaml_name):
    """Use ultralytics to resolve / download a dataset and return paths."""
    from ultralytics.data.utils import check_det_dataset
    info = check_det_dataset(data_yaml_name)
    train_imgs = info.get("train", "")
    val_imgs   = info.get("val", "")
    nc         = info.get("nc", 80)
    names      = info.get("names", {})
    # Derive label dirs
    def _lbl(img_path):
        return str(Path(str(img_path)).parent).replace("images", "labels")
    return {
        "train_imgs": str(train_imgs), "train_lbls": _lbl(train_imgs),
        "val_imgs":   str(val_imgs),   "val_lbls":   _lbl(val_imgs),
        "nc": nc, "names": names,}
# Download all requested datasets
DATASETS = {}
for ds_name in ["coco8.yaml", "coco128.yaml", "african-wildlife.yaml",
                "coco8-grayscale.yaml"]:
    try:
        print(f"\n{'='*50}\nDownloading {ds_name} ...")
        DATASETS[ds_name] = get_dataset_paths(ds_name)
        print(f"nc={DATASETS[ds_name]['nc']}  "
              f"train={DATASETS[ds_name]['train_imgs']}")
    except Exception as e:
        print(f"Skipped {ds_name}: {e}")

DATASETS["coco.yaml"] = get_dataset_paths("coco.yaml")
DATASETS["coco8-multispectral.yaml"] = get_dataset_paths("coco8-multispectral.yaml")

# CELL 15 — Training Engine
class Trainer:
    def __init__(self, model, cfg=CFG, device=DEVICE):
        self.model  = model.to(device)
        self.cfg    = cfg
        self.device = device
        self.loss_fn = YOLO26Loss(cfg).to(device)
        self.scaler  = torch.amp.GradScaler('cuda')
        # Build optimizer — MuSGD
        pg_bn, pg_weight, pg_other = [], [], []
        for k, v in model.named_parameters():
            if not v.requires_grad:
                continue
            if ".bn" in k or "bn." in k:
                pg_bn.append(v)
            elif ".weight" in k and v.dim() >= 2:
                pg_weight.append(v)
            else:
                pg_other.append(v)
        self.optimizer = MuSGD([
            {"params": pg_bn,     "weight_decay": 0.0},
            {"params": pg_weight, "weight_decay": cfg.weight_decay},
            {"params": pg_other,  "weight_decay": 0.0},
        ], lr=cfg.lr0, momentum=cfg.momentum, muon_strength=0.05)
        self.best_loss = float("inf")
        self.history = defaultdict(list)
    def _lr_schedule(self, epoch):
        """Cosine annealing from lr0 to lr0 * lrf."""
        if epoch < self.cfg.warmup_epochs:
            return self.cfg.lr0 * (epoch + 1) / self.cfg.warmup_epochs
        progress = (epoch - self.cfg.warmup_epochs) / max(
            1, self.cfg.epochs - self.cfg.warmup_epochs)
        return self.cfg.lr0 * (
            (1 + math.cos(math.pi * progress)) / 2 *
            (1 - self.cfg.lrf) + self.cfg.lrf)
    def train_one_epoch(self, loader, epoch):
        self.model.train()
        lr = self._lr_schedule(epoch)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        total_loss, total_box, total_cls, n_batches = 0, 0, 0, 0
        for imgs, gt_labels, gt_bboxes, mask_gt in loader:
            imgs      = imgs.to(self.device, non_blocking=True)
            gt_labels = gt_labels.to(self.device)
            gt_bboxes = gt_bboxes.to(self.device)
            mask_gt   = mask_gt.to(self.device)
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = self.model(imgs)
                loss, box_l, cls_l = self.loss_fn(
                    outputs, gt_labels, gt_bboxes, mask_gt,
                    epoch, self.cfg.epochs)
            if torch.isfinite(loss):
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            total_loss += loss.item()
            total_box  += box_l.item()
            total_cls  += cls_l.item()
            n_batches  += 1
        avg = lambda x: x / max(n_batches, 1)
        return avg(total_loss), avg(total_box), avg(total_cls), lr
    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss, n = 0, 0
        for imgs, gt_labels, gt_bboxes, mask_gt in loader:
            imgs      = imgs.to(self.device)
            gt_labels = gt_labels.to(self.device)
            gt_bboxes = gt_bboxes.to(self.device)
            mask_gt   = mask_gt.to(self.device)
            with torch.amp.autocast('cuda'):
                outputs = self.model(imgs)
                loss, _, _ = self.loss_fn(outputs, gt_labels, gt_bboxes, mask_gt)
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)
    def fit(self, train_loader, val_loader=None, epochs=None):
        epochs = epochs or self.cfg.epochs
        print(f"\nTraining for {epochs} epochs …\n")
        for epoch in range(epochs):
            t0 = time.time()
            train_loss, box_l, cls_l, lr = self.train_one_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader) if val_loader else 0.0
            dt = time.time() - t0
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)
            print(f"  Epoch {epoch+1:3d}/{epochs} │ "
                  f"loss {train_loss:.4f} (box {box_l:.4f} cls {cls_l:.4f}) │ "
                  f"val {val_loss:.4f} │ lr {lr:.6f} │ {dt:.1f}s")
            if val_loss < self.best_loss and val_loader:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), "best_myyolo26.pt")
                print(f"Saved best model (val_loss={val_loss:.4f})")
            elif epoch % 10 == 0:
                torch.save(self.model.state_dict(), "last_myyolo26.pt")
        torch.save(self.model.state_dict(), "last_myyolo26.pt")
        print("\nTraining complete. Saved → last_myyolo26.pt")
    def plot_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(self.history["train_loss"], label="train")
        axes[0].plot(self.history["val_loss"], label="val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(self.history["lr"])
        axes[1].set_title("Learning Rate"); axes[1].grid(True)
        plt.tight_layout(); plt.show()

# CELL 16 — Multi-Dataset Training Pipeline
def train_on_dataset(ds_key, model=None, epochs=50, batch_size=None):
    """Train (or continue training) on a specific dataset."""
    info = DATASETS[ds_key]
    nc   = info["nc"]
    bs   = batch_size or CFG.batch_size
    print(f"\n{'═'*60}")
    print(f"Training on: {ds_key}  (nc={nc})")
    print(f"{'═'*60}")
    # Adjust model nc if needed
    if model is None:
        cfg = copy.deepcopy(CFG)
        cfg.nc = nc
        cfg.epochs = epochs
        model = MyYOLO26(cfg)
        model.count_params()
    else:
        # Rebuild head if nc changed
        if model.cfg.nc != nc:
            print(f"Rebuilding head: {model.cfg.nc} → {nc} classes")
            model.cfg.nc = nc
            head_ch = (model.cfg.channels[2], model.cfg.channels[3], model.cfg.channels[4])
            model.head = DualDetectHead(nc, head_ch)
    # DataLoaders
    train_ds = YOLODataset(info["train_imgs"], info["train_lbls"],
                           CFG.img_size, augment=True)
    val_ds   = YOLODataset(info["val_imgs"], info["val_lbls"],
                           CFG.img_size, augment=False)
    train_ld = DataLoader(train_ds, bs, shuffle=True,  num_workers=2,
                          collate_fn=collate_fn, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   bs, shuffle=False, num_workers=2,
                          collate_fn=collate_fn, pin_memory=True)
    trainer = Trainer(model, model.cfg, DEVICE)
    trainer.fit(train_ld, val_ld, epochs=epochs)
    trainer.plot_history()
    return model

model = None
if "coco8.yaml" in DATASETS:
    model = train_on_dataset("coco8.yaml", model, epochs=30, batch_size=8)
if "coco128.yaml" in DATASETS:
    model = train_on_dataset("coco128.yaml", model, epochs=50, batch_size=8)
if "african-wildlife.yaml" in DATASETS:
    aw_model = train_on_dataset("african-wildlife.yaml", None, epochs=50, batch_size=8)
if "coco8-grayscale.yaml" in DATASETS:
    model = train_on_dataset("coco8-grayscale.yaml", model, epochs=20, batch_size=8)
if "coco.yaml" in DATASETS:
    model = train_on_dataset("coco.yaml", model, epochs=100, batch_size=8)

# CELL 17 — Inference & Visualization
class Predictor:
    """Run end-to-end inference with the one-to-one head (NMS-free)."""
    def __init__(self, model, cfg=CFG, device=DEVICE, conf_thr=0.25):
        self.model = model.to(device).eval()
        self.cfg   = cfg
        self.device = device
        self.conf_thr = conf_thr
    @torch.no_grad()
    def predict(self, img_bgr):
        """img_bgr: raw BGR image (H,W,3).
        Returns: list of (x1,y1,x2,y2,conf,cls_id)."""
        oh, ow = img_bgr.shape[:2]
        img_lb, ratio, (pw, ph) = letterbox(img_bgr, self.cfg.img_size)
        x = torch.from_numpy(
            img_lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        ).unsqueeze(0).to(self.device)
        with torch.amp.autocast('cuda'):
            out = self.model(x)
        shapes  = out["shapes"]
        anchors, strides = make_anchors(shapes, self.cfg.strides)
        anchors = anchors.to(self.device)
        strides = strides.to(self.device)
        box_raw, cls_raw = out["o2o"]                          # use o2o (NMS-free)
        box_pred = box_raw.permute(0, 2, 1)                   # (1,N,4)
        cls_pred = cls_raw.permute(0, 2, 1).sigmoid()         # (1,N,nc)
        pred_boxes = dist2bbox(
            box_pred * strides.squeeze(-1)[None, :, None],
            anchors[None])                                     # (1,N,4)
        # Per-class confidence
        max_conf, cls_id = cls_pred[0].max(dim=-1)            # (N,)
        # Filter by confidence
        keep = max_conf > self.conf_thr
        boxes = pred_boxes[0, keep]
        confs = max_conf[keep]
        clsids = cls_id[keep]
        # Top-k
        if len(confs) > self.cfg.max_det:
            topk = confs.topk(self.cfg.max_det).indices
            boxes, confs, clsids = boxes[topk], confs[topk], clsids[topk]
        # Scale back to original image
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pw) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - ph) / ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, oh)
        dets = torch.cat([boxes, confs.unsqueeze(-1),
                          clsids.float().unsqueeze(-1)], -1)
        return dets.cpu().numpy()
    def visualize(self, img_bgr, dets, class_names=None):
        vis = img_bgr.copy()
        colors = plt.cm.tab20(np.linspace(0, 1, 80))[:, :3] * 255
        for d in dets:
            x1, y1, x2, y2, conf, cid = d
            cid = int(cid)
            c = tuple(int(v) for v in colors[cid % len(colors)])
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            label = f"{class_names[cid] if class_names else cid} {conf:.2f}"
            cv2.putText(vis, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        return vis
# Quick demo
if model is not None:
    predictor = Predictor(model, conf_thr=0.15)
    # Use a sample image from the dataset
    ds_info = DATASETS.get("coco8.yaml") or DATASETS.get("coco128.yaml")
    if ds_info:
        sample_imgs = glob.glob(os.path.join(ds_info["val_imgs"], "*.*"))
        if sample_imgs:
            img = cv2.imread(sample_imgs[0])
            dets = predictor.predict(img)
            print(f"\nDetected {len(dets)} objects in sample image")
            vis = predictor.visualize(img, dets,
                                      ds_info.get("names", None))
            plt.figure(figsize=(10, 8))
            plt.imshow(vis[:, :, ::-1])
            plt.axis("off"); plt.title("MyYOLO26x Inference")
            plt.show()

# CELL 18 — Export to ONNX
def export_onnx(model, path="myyolo26x.onnx", img_size=640):
    """Export model to ONNX for production deployment."""
    model.eval().to("cpu")
    dummy = torch.randn(1, 3, img_size, img_size)
    # We need to wrap the forward to only return o2o outputs
    class InferenceWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            # Return o2o raw box + cls
            return out["o2o"][0], out["o2o"][1]
    wrapper = InferenceWrapper(model)
    torch.onnx.export(
        wrapper, dummy, path,
        input_names=["images"],
        output_names=["box_raw", "cls_raw"],
        dynamic_axes={"images": {0: "batch"}, "box_raw": {0: "batch"},
                      "cls_raw": {0: "batch"}},
        opset_version=17)
    size_mb = os.path.getsize(path) / 1e6
    print(f"ONNX exported → {path} ({size_mb:.1f} MB)")
    model.to(DEVICE)  # move back
if model is not None:
    export_onnx(model)
print("Downloading full COCO dataset (this takes a while)...")
DATASETS["coco.yaml"] = get_dataset_paths("coco.yaml")
model = train_on_dataset("coco.yaml", model, epochs=100, batch_size=16)
export_onnx(model, "myyolo26x_coco.onnx"))
