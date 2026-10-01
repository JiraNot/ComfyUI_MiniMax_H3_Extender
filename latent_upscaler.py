from __future__ import annotations

import gc
import hashlib
import os
import re
import urllib.request
from pathlib import Path

import folder_paths
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import comfy.model_management as mm
except Exception:
    mm = None

FOLDER = "latent_upscale_models"
MODEL_NAME = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
MODEL_URL = (
    "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/"
    "minimax_h3_latent_upscaler_3d_conv_v1/"
    "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors?download=true"
)
MODEL_SHA256 = "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6"
EXPECTED_BYTES = 690_592_992
VAE_DOWNSAMPLE = 16

if FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))

LATENTS_MEAN = [
    0.858090341091156,-0.9606591463088989,1.0661640167236328,-0.5090325474739075,
    -0.2727581858634949,-1.3675414323806763,-0.2553254961967468,-0.26907554268836975,
    -0.5376840829849243,-0.0464097298681736,0.6657370328903198,0.19690127670764923,
    -0.5460608005523682,-0.4035342037677765,-0.23683024942874908,0.25928452610969543,
    -0.30133944749832153,0.211341992020607,-1.1206848621368408,0.3581933379173279,
    -0.04225143790245056,0.2604829967021942,0.22864092886447906,0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407,1.2767263650894165,1.6831774711608887,1.7549455165863037,
    1.5636216402053833,2.194143533706665,0.9653137922286987,1.0569885969161987,
    0.841948926448822,0.7729952931404114,1.8955937623977661,0.946841835975647,
    0.7996809482574463,0.44988900423049927,0.7197399735450745,0.6936293244361877,
    2.961095094680786,2.7694199085235596,3.0496184825897217,2.1088054180145264,
    3.276226282119751,3.1627357006073,2.2816812992095947,2.6127843856811523,
]

_CACHE = {}


def model_path() -> Path:
    root = Path(folder_paths.models_dir) / FOLDER
    root.mkdir(parents=True, exist_ok=True)
    return root / MODEL_NAME


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_upscale_model(progress=None) -> str:
    """Return the fixed BF16 upscaler path, downloading it once if absent."""
    path = model_path()
    verified = path.with_suffix(path.suffix + ".sha256.ok")
    if path.exists():
        if path.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError(f"MiniMax H3 Extender: latent upscaler exists but has an invalid size: {path}")
        marker = verified.read_text(encoding="utf-8").strip() if verified.exists() else ""
        if marker != MODEL_SHA256:
            if _sha256(path) != MODEL_SHA256:
                raise RuntimeError(f"MiniMax H3 Extender: latent upscaler exists but failed SHA256 verification: {path}")
            verified.write_text(MODEL_SHA256, encoding="utf-8")
        return str(path)

    part = path.with_suffix(path.suffix + ".part")
    part.unlink(missing_ok=True)
    if progress:
        progress("Downloading H3 latent upscaler (~691 MB)...")
    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "ComfyUI-MiniMax-H3-Extender/3.0"})
        with urllib.request.urlopen(req, timeout=60) as src, part.open("wb") as dst:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        if part.stat().st_size != EXPECTED_BYTES or _sha256(part) != MODEL_SHA256:
            raise RuntimeError("downloaded latent upscaler failed size/SHA256 verification")
        os.replace(part, path)
        verified.write_text(MODEL_SHA256, encoding="utf-8")
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            "MiniMax H3 Extender: failed to download the latent upscaler. "
            f"You can manually place {MODEL_NAME} in models/{FOLDER}/. {exc}"
        ) from exc
    return str(path)


def _norm(channels):
    return nn.GroupNorm(32, channels)


def _zero(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0.1, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(_norm(channels), nn.SiLU(), nn.Conv3d(channels, self.out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels))
        self.out_norm = _norm(self.out_channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), _zero(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)))
        self.skip = nn.Conv3d(channels, self.out_channels, 1) if self.out_channels != channels else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(h)


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.norm = _norm(channels)
        self.dwconv = nn.Conv3d(channels, channels, (kernel_size, 1, 1), padding=(pad, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12, channels=512, dropout=0.1, temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        emb_dim = 64
        self.embed = nn.Sequential(nn.Linear(1, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.in_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        for i in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, emb_dim, dropout))
            if temporal_every > 0 and i % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))
        for i in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, emb_dim, dropout))
            if temporal_every > 0 and i % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))
        self.norm_out = _norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def _segment(self, x, scale, size):
        emb = self.embed(torch.tensor([[float(scale) - 1.0]], dtype=x.dtype, device=x.device))
        x = self.conv_in(x)
        for b in self.in_blocks:
            x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for b in self.out_blocks:
            x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        return self.conv_out(F.silu(self.norm_out(x)))

    def forward(self, x, scale, target_size, enable_chunking=True):
        t = int(x.shape[2])
        chunk = 32
        temporal_kernel = 0
        for block in self.in_blocks:
            if isinstance(block, TemporalConv):
                temporal_kernel = int(block.dwconv.weight.shape[2])
                break
        overlap = temporal_kernel
        if not enable_chunking or t <= chunk:
            return self._segment(x, scale, target_size)
        padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        out = torch.zeros(
            x.shape[0], x.shape[1], t, target_size[-2], target_size[-1],
            device=x.device, dtype=x.dtype,
        )
        weights = torch.zeros(1, 1, t, 1, 1, device=x.device, dtype=x.dtype)
        pos = 0
        while pos < t:
            seg_start = pos
            seg_end = min(t, pos + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(t, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(t + 2 * overlap, out_end + overlap)
            seg = padded[:, :, lo:hi].contiguous()
            seg_out = self._segment(seg, scale, (hi - lo, target_size[-2], target_size[-1]))
            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            valid = seg_out[:, :, s0:s1]
            n = out_end - out_start
            weight = torch.ones(n, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend = seg_start - out_start
                weight[:blend] = torch.arange(1, blend + 1, device=x.device, dtype=x.dtype) / (blend + 1)
            if out_end > seg_end:
                blend = out_end - seg_end
                weight[-blend:] = torch.arange(blend, 0, -1, device=x.device, dtype=x.dtype) / (blend + 1)
            out[:, :, out_start:out_end] += valid * weight.view(1, 1, n, 1, 1)
            weights[:, :, out_start:out_end] += weight.view(1, 1, n, 1, 1)
            pos += chunk
        return out / weights.clamp(min=1e-8)


def _load_state(path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd


def _detect(sd):
    cfg = dict(in_channels=24, in_blocks=12, out_blocks=12, channels=512, dropout=0.1, temporal_every=2, temporal_kernel=5)
    if "conv_in.weight" in sd:
        cfg["in_channels"] = int(sd["conv_in.weight"].shape[1]); cfg["channels"] = int(sd["conv_in.weight"].shape[0])
    for side in ("in", "out"):
        ids = set()
        for k in sd:
            m = re.match(rf"{side}_blocks\.(\d+)\.in_layers\.", k)
            if m: ids.add(int(m.group(1)))
        if ids: cfg[f"{side}_blocks"] = len(ids)
    temporal = [k for k in sd if k.endswith("dwconv.weight")]
    if temporal:
        cfg["temporal_kernel"] = int(sd[temporal[0]].shape[2])
    else:
        cfg["temporal_every"] = 0
    return cfg


def _load_model(path, device):
    key = str(path)
    model = _CACHE.get(key)
    if model is None:
        sd = _load_state(path)
        cfg = _detect(sd)
        if cfg["in_channels"] != 24:
            raise RuntimeError("MiniMax H3 Extender: latent upscaler is not a 24-channel H3 model")
        model = LatentResizer3D(**cfg).to(torch.bfloat16)
        model.load_state_dict(sd, strict=True)
        model = model.eval().requires_grad_(False)
        _CACHE[key] = model.cpu()
    return model.to(device)


def target_dimensions(width: int, height: int, scale: float):
    scale = max(1.0, float(scale))
    w = max(32, int(round((int(width) * scale) / 32.0)) * 32)
    h = max(32, int(round((int(height) * scale) / 32.0)) * 32)
    return w, h


@torch.inference_mode()
def upscale_video_latent(video: torch.Tensor, scale: float, progress=None):
    if not torch.is_tensor(video) or video.ndim != 5 or int(video.shape[1]) != 24:
        raise ValueError("MiniMax H3 Extender: expected H3 video latent [B,24,T,H,W]")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_device, orig_dtype = video.device, video.dtype
    _, _, t, h, w = video.shape
    target_w_px, target_h_px = target_dimensions(w * VAE_DOWNSAMPLE, h * VAE_DOWNSAMPLE, scale)
    h2, w2 = target_h_px // VAE_DOWNSAMPLE, target_w_px // VAE_DOWNSAMPLE
    if (h2, w2) == (h, w):
        return video, target_w_px, target_h_px
    path = ensure_upscale_model(progress=progress)
    model = _load_model(path, dev)
    x = video.to(dev, dtype=torch.bfloat16, copy=True)
    mean = torch.tensor(LATENTS_MEAN, device=dev, dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, device=dev, dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
    x = (x - mean) / std
    out = model(x, float(scale), (t, h2, w2), enable_chunking=True)
    out = (out * std + mean).to(device=orig_device, dtype=orig_dtype)
    model.cpu(); del x, mean, std
    if mm is not None: mm.soft_empty_cache()
    elif torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return out, target_w_px, target_h_px
