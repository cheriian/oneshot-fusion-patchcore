"""One-shot PatchCore on MVTec AD: WRN-50 (paper) vs DINOv2 vs WRN-50 + DINOv2 feature fusion.
Same images, resolution, coreset (10%), scoring and seeds for every backbone."""
import argparse, csv, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Subset

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
from patchcore.datasets.mvtec import DatasetSplit, MVTecDataset

CLASSES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
           "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
FIELDS = ["classname", "shots", "seed", "method", "n_train", "img_auroc", "pix_auroc", "sec_per_img"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_wrn(imagesize, coreset, device):
    backbone = patchcore.backbones.load("wideresnet50")
    backbone.name, backbone.seed = "wideresnet50", None
    model = patchcore.patchcore.PatchCore(device)
    model.load(backbone=backbone, layers_to_extract_from=["layer2", "layer3"], device=device,
               input_shape=(3, imagesize, imagesize), pretrain_embed_dimension=1024,
               target_embed_dimension=1024, patchsize=3,
               featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(coreset, device),
               nn_method=patchcore.common.FaissNN(False, 4))
    return model


# ---------- Baseline: official PatchCore, WRN-50 layers 2+3 (unchanged from pilot) ----------
def run_wrn(train_ds, test_loader, imagesize, coreset, device):
    model = build_wrn(imagesize, coreset, device)
    model.fit(DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=2))
    t0 = time.time()
    scores, segs, labels, masks = model.predict(test_loader)
    sec = (time.time() - t0) / len(test_loader.dataset)
    return np.array(scores), np.array(segs), np.array(labels), np.array(masks), sec


# ---------- DINOv2 patch tokens (unchanged from pilot) ----------
class DinoFeatures:
    def __init__(self, arch, layers, device):
        self.model = torch.hub.load("facebookresearch/dinov2", arch, trust_repo=True).to(device).eval()
        self.layers, self.device = layers, device

    @torch.no_grad()
    def __call__(self, x):
        feats = self.model.get_intermediate_layers(x.to(self.device), n=self.layers, reshape=True)
        feats = [F.avg_pool2d(f, 3, stride=1, padding=1) for f in feats]  # PatchCore p=3 neighbourhood
        return torch.cat(feats, dim=1)  # B x C x h x w


# ---------- Extension: WRN-50 + DINOv2 fused per patch ----------
class FusionFeatures:
    """WRN-50 patch features exactly as PatchCore builds them (layer2+3, p=3, 1024-d, 28x28 at 224px)
    + DINOv2 tokens upsampled to the same grid. Each branch is L2-normalised and weighted so that
    ||f-f'||^2 = alpha*||w-w'||^2 + (1-alpha)*||d-d'||^2."""
    def __init__(self, arch, layers, imagesize, coreset, alpha, device):
        self.dino = DinoFeatures(arch, layers, device)
        self.wrn = build_wrn(imagesize, coreset, device)  # only its feature extractor is used
        self.a_w, self.a_d = alpha ** 0.5, (1.0 - alpha) ** 0.5
        self.device = device

    @torch.no_grad()
    def __call__(self, x):
        x = x.to(self.device, dtype=torch.float)
        w, shapes = self.wrn._embed(x, detach=False, provide_patch_shapes=True)  # official PatchCore features
        if isinstance(w, (list, tuple)):
            w = torch.stack([torch.as_tensor(v) for v in w])
        w = torch.as_tensor(w, device=self.device, dtype=torch.float)
        (hh, ww), B = shapes[0], x.shape[0]
        assert w.shape[0] == B * hh * ww, f"unexpected WRN feature shape {tuple(w.shape)}"
        w = w.reshape(B, hh, ww, -1).permute(0, 3, 1, 2)                     # B x 1024 x 28 x 28
        d = F.interpolate(self.dino(x), size=(hh, ww), mode="bilinear", align_corners=False)
        return torch.cat([self.a_w * F.normalize(w, dim=1), self.a_d * F.normalize(d, dim=1)], dim=1)


def to_patches(f):  # B x C x h x w -> (B*h*w) x C
    return f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])


@torch.no_grad()
def nn_dist(q, bank, chunk=2048):  # distance of each test patch to nearest memory-bank patch
    return torch.cat([torch.cdist(c, bank).min(dim=1).values for c in q.split(chunk)])


@torch.no_grad()
def run_bank(extractor, train_ds, test_loader, imagesize, coreset, device):
    """Memory-bank PatchCore for any extractor returning B x C x h x w (DINOv2 or Fusion)."""
    loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=2)
    feats = torch.cat([to_patches(extractor(b["image"])).cpu() for b in loader]).numpy()
    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(coreset, device)  # official greedy coreset
    bank = torch.as_tensor(np.asarray(sampler.run(feats)), device=device)
    t0 = time.time()
    scores, maps, labels, masks = [], [], [], []
    for b in test_loader:
        f = extractor(b["image"])
        B, C, h, w = f.shape
        d = nn_dist(to_patches(f), bank).reshape(B, 1, h, w)
        scores += d.flatten(1).max(dim=1).values.cpu().tolist()          # image score = worst patch
        up = F.interpolate(d, size=(imagesize, imagesize), mode="bilinear", align_corners=False)
        maps += [gaussian_filter(m, sigma=4) for m in up[:, 0].cpu().numpy()]  # same smoothing as paper
        labels += b["is_anomaly"].tolist()
        masks.append(b["mask"].numpy())
    sec = (time.time() - t0) / len(test_loader.dataset)
    return np.array(scores), np.stack(maps), np.array(labels), np.concatenate(masks), sec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results.csv")
    p.add_argument("--backbone", choices=["wrn50", "dinov2", "fusion"], required=True)
    p.add_argument("--arch", default="dinov2_vits14")
    p.add_argument("--layers", nargs="+", type=int, default=[7, 11])
    p.add_argument("--alpha", type=float, default=0.5)                # fusion weight on WRN-50
    p.add_argument("--imagesize", type=int, default=224)
    p.add_argument("--classes", nargs="+", default=CLASSES)
    p.add_argument("--shots", nargs="+", type=int, default=[1])       # 0 = full training set
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--coreset", type=float, default=0.1)              # PatchCore-10
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = a.imagesize
    tag = f"{a.arch}_L{'-'.join(map(str, a.layers))}"
    if a.backbone == "wrn50":
        method, extractor = f"wrn50_L2-3_{size}", None
    else:
        assert size % 14 == 0, "image size must be a multiple of 14 for DINOv2 (e.g. 224)"
        if a.backbone == "dinov2":
            method, extractor = f"{tag}_{size}", DinoFeatures(a.arch, a.layers, device)
        else:
            method = f"fusion_wrn50L2-3_{tag}_a{a.alpha:g}_{size}"
            extractor = FusionFeatures(a.arch, a.layers, size, a.coreset, a.alpha, device)
    resize = int(round(size * 256 / 224))  # paper: resize 256, centre-crop 224

    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {(r["classname"], r["shots"], r["seed"], r["method"]) for r in csv.DictReader(f)}
    new_file = not os.path.exists(a.out)
    out = open(a.out, "a", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    for cls in a.classes:
        train_full = MVTecDataset(a.data, classname=cls, resize=resize, imagesize=size, split=DatasetSplit.TRAIN)
        test_ds = MVTecDataset(a.data, classname=cls, resize=resize, imagesize=size, split=DatasetSplit.TEST)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)
        for shots in a.shots:
            for seed in (a.seeds if shots > 0 else [0]):
                if (cls, str(shots), str(seed), method) in done:
                    continue
                set_seed(seed)
                if shots > 0:  # identical training image for every backbone with the same seed
                    idx = np.random.RandomState(seed).choice(len(train_full), size=min(shots, len(train_full)),
                                                             replace=False).tolist()
                    train_ds = Subset(train_full, idx)
                else:
                    train_ds = train_full
                if a.backbone == "wrn50":
                    scores, maps, labels, masks, sec = run_wrn(train_ds, test_loader, size, a.coreset, device)
                else:
                    scores, maps, labels, masks, sec = run_bank(extractor, train_ds, test_loader, size,
                                                                a.coreset, device)
                img = patchcore.metrics.compute_imagewise_retrieval_metrics(scores, labels)["auroc"]
                pix = patchcore.metrics.compute_pixelwise_retrieval_metrics(maps, masks)["auroc"]
                w.writerow(dict(classname=cls, shots=shots, seed=seed, method=method, n_train=len(train_ds),
                                img_auroc=round(img, 5), pix_auroc=round(pix, 5), sec_per_img=round(sec, 4)))
                out.flush()
                print(f"{cls:11s} shots={shots} seed={seed} {method:40s} "
                      f"img={img*100:6.2f} pix={pix*100:6.2f} {sec:.3f}s/img", flush=True)
                torch.cuda.empty_cache()
    out.close()


if __name__ == "__main__":
    main()
