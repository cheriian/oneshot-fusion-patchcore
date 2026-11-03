"""One-shot PatchCore study (MVTec AD, 10 seeds, PatchCore-10 setting):
four-way comparison, paired tests, ablation, report figures."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.environ.get("RESULTS_DIR",
                      "/content/drive/MyDrive/DL_Project")
OUT = os.path.join(ROOT, "report_assets")
os.makedirs(OUT, exist_ok=True)

# Roth et al. 2022, Table S5, 1-shot
PAPER = {"img_auroc": (83.4, 0.6), "pix_auroc": (92.0, 0.2)}
PAPER25 = {"img_auroc": (84.1, 0.7), "pix_auroc": (92.4, 0.3)}

MAIN = {
    "wrn50_L2-3_224": "WRN-50 (repro.)",
    "dinov2_vits14_L7-11_224": "DINOv2",
    "fusion_wrn50L2-3_dinov2_vits14_L7-11_a0.5_224": "Fusion",
}
ABL = {
    "wrn50_L2-3_224": "A. WRN-50 (official)",
    "fusion_wrn50L2-3_dinov2_vits14_L7-11_a1_224":
        "B. WRN-50, L2-norm (alpha=1)",
    "dinov2_vits14_L7-11_224": "C. DINOv2, 16x16 grid",
    "fusion_wrn50L2-3_dinov2_vits14_L7-11_a0_224":
        "D. DINOv2, 28x28 grid (alpha=0)",
    "fusion_wrn50L2-3_dinov2_vits14_L7-11_a0.5_224":
        "E. Fusion (alpha=0.5)",
}
METRICS = ["img_auroc", "pix_auroc"]
KEYS = ["classname", "shots", "seed", "method"]


def load(names):
    path = os.path.join(ROOT, "results.csv")
    df = pd.read_csv(path)
    df = df.drop_duplicates(KEYS)
    keep = (df.shots == 1) & df.seed.between(0, 9)
    keep = keep & df.method.isin(names)
    df = df[keep].copy()
    df["model"] = df.method.map(names)
    counts = df.groupby("model").size()
    ok = len(counts) == len(names) and (counts == 150).all()
    assert ok, f"Incomplete runs:\n{counts}"
    return df


def seed_means(df):
    """Mean over 15 categories per seed (Table S5 protocol), %."""
    g = df.groupby(["model", "seed"])
    ps = g[METRICS].mean() * 100
    ps["sec"] = g.sec_per_img.mean()
    return ps


def paired(wide, met, x, y):
    a = wide[met][x]
    b = wide[met][y]
    if (a != b).any():
        p = stats.wilcoxon(a, b).pvalue
    else:
        p = 1.0
    return np.mean(a - b), int((a > b).sum()), p


def fmt(mean, std, nd=1):
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def vs_paper(ps, models):
    refs = [("PatchCore-10", PAPER),
            ("PatchCore-25 (reference)", PAPER25)]
    for name, ref in refs:
        print(f"Gain over paper {name}, share of error removed:")
        for m in models:
            for met in METRICS:
                ours = ps.loc[m][met].mean()
                r = ref[met][0]
                removed = 100 * (ours - r) / (100 - r)
                print(f"  {m:34s} {met}: {ours - r:+.2f} pts, "
                      f"error removed {removed:5.1f}%")
        print()


def per_category(df, order, tag):
    for met in METRICS:
        t = df.pivot_table(index="classname", columns="model",
                           values=met, aggfunc="mean")
        t = (t * 100)[order].round(2)
        t.loc["MEAN"] = t.mean().round(2)
        print(f"{met} per category:\n{t}\n")
        t.to_csv(os.path.join(OUT, f"{tag}_per_class_{met}.csv"))


def main_comparison():
    df = load(MAIN)
    order = list(MAIN.values())
    ps = seed_means(df)
    rows = []
    for name, ref in [("Paper, PatchCore-10 (Table S5)", PAPER),
                      ("Paper, PatchCore-25 (reference)", PAPER25)]:
        rows.append({"Method": name,
                     "Image AUROC": fmt(*ref["img_auroc"]),
                     "Pixel AUROC": fmt(*ref["pix_auroc"]),
                     "s/img": "-"})
    for m in order:
        s = ps.loc[m]
        rows.append({"Method": m,
                     "Image AUROC": fmt(s.img_auroc.mean(),
                                        s.img_auroc.std(), 2),
                     "Pixel AUROC": fmt(s.pix_auroc.mean(),
                                        s.pix_auroc.std(), 2),
                     "s/img": f"{s.sec.mean():.3f}"})
    table = pd.DataFrame(rows)
    print("FOUR-WAY COMPARISON (1-shot, mean ± std, 10 seeds)")
    print(table.to_string(index=False), "\n")
    table.to_csv(os.path.join(OUT, "table_main.csv"), index=False)
    vs_paper(ps, order)

    wide = ps.reset_index().pivot(index="seed", columns="model")
    pairs = [("Fusion", "WRN-50 (repro.)"),
             ("Fusion", "DINOv2"),
             ("DINOv2", "WRN-50 (repro.)")]
    for met in METRICS:
        print(f"--- {met}: paired tests over 10 seeds ---")
        for x, y in pairs:
            d, w, p = paired(wide, met, x, y)
            print(f"  {x} vs {y}: diff {d:+.2f}, "
                  f"seeds won {w}/10, Wilcoxon p={p:.4f}")
        for m in order:
            ref = PAPER[met][0]
            p = stats.ttest_1samp(wide[met][m], ref).pvalue
            print(f"  {m} vs paper PatchCore-10 {ref}: "
                  f"one-sample t p={p:.4f}")
        print()
    per_category(df, order, "main")
    print("Per-seed image AUROC:")
    print(wide["img_auroc"][order].round(2), "\n")

    plt.rcParams.update({"font.size": 7})
    labels = [("img_auroc", "Image AUROC (%)"),
              ("pix_auroc", "Pixel AUROC (%)")]
    for met, lab in labels:
        g = df.groupby(["classname", "model"])[met]
        mu = g.mean().unstack()[order] * 100
        sd = g.std().unstack()[order] * 100
        ax = mu.plot(kind="bar", yerr=sd, capsize=1.5, width=0.8,
                     figsize=(7.2, 2.4), error_kw={"lw": 0.6})
        ax.set_ylabel(lab)
        ax.set_xlabel("")
        ax.set_ylim(40, 101)
        ax.legend(ncol=3, loc="lower left")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"per_class_{met}.pdf"))
        plt.close()

    ax = wide["img_auroc"][order].plot(marker="o", figsize=(3.5, 2.3))
    ax.axhline(PAPER["img_auroc"][0], ls="--", c="gray", lw=0.8,
               label="Paper PatchCore-10 (83.4)")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mean image AUROC (%)")
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "per_seed_img.pdf"))
    plt.close()


def ablation():
    df = load(ABL)
    order = list(ABL.values())
    ps = seed_means(df)
    abl = ps[METRICS].groupby("model").agg(["mean", "std"]).round(2)
    print("ABLATION (mean ± std over 10 seeds)")
    print(abl, "\n")
    abl.to_csv(os.path.join(OUT, "table_ablation.csv"))
    vs_paper(ps, order)
    key = {v[0]: v for v in order}
    wide = ps.reset_index().pivot(index="seed", columns="model")
    steps = [("B", "A", "L2-norm on WRN-50"),
             ("D", "C", "28x28 grid for DINOv2"),
             ("E", "D", "adding WRN-50 to grid-aligned DINOv2"),
             ("E", "B", "adding DINOv2 to normalised WRN-50")]
    for met in METRICS:
        print(f"--- {met}: ablation, paired over 10 seeds ---")
        for x, y, what in steps:
            d, w, p = paired(wide, met, key[x], key[y])
            print(f"  {x} vs {y} ({what}): diff {d:+.2f}, "
                  f"seeds won {w}/10, Wilcoxon p={p:.4f}")
        print()
    per_category(df, order, "ablation")


if __name__ == "__main__":
    main_comparison()
    ablation()
    print(f"All tables and figures saved to {OUT}")
