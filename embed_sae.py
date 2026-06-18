"""
Pass pre-computed SPECTER2 embeddings through the trained SAE and save
the sparse feature activations.

Inputs  (saes/data/):
  embeddings.npy     — float32 [N, 768] SPECTER2 embeddings
  embedding_ids.npy  — int64   [N]      paper IDs

Outputs (saes/data/):
  sae_acts.npy       — float32 [N, 2048] SAE activations
  sae_act_ids.npy    — int64   [N]       paper IDs (same order)
"""

import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "saes/model")
from saelens import TopKSAE, DEVICE

EMBEDDINGS_NPY = "saes/data/embeddings.npy"
IDS_NPY        = "saes/data/embedding_ids.npy"
CHECKPOINT     = "saes/data/saelens_checkpoint.pt"
OUTPUT_ACTS    = "saes/data/sae_acts.npy"
OUTPUT_IDS     = "saes/data/sae_act_ids.npy"

BATCH_SIZE = 256


def normalize(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    x = x / x.norm(dim=1, keepdim=True).mean()
    return x


def main():
    print(f"Device: {DEVICE}")

    print(f"Loading embeddings from {EMBEDDINGS_NPY}...")
    raw = np.load(EMBEDDINGS_NPY).astype(np.float32)
    ids = np.load(IDS_NPY)
    print(f"  {raw.shape[0]} embeddings, dim={raw.shape[1]}")

    print("Normalizing embeddings (match training preprocessing)...")
    embeddings = normalize(torch.from_numpy(raw)).to(DEVICE)

    print(f"Loading SAE from {CHECKPOINT}...")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    cfg = ckpt["cfg"]
    sae = TopKSAE(cfg["d_in"], cfg["d_sae"], cfg["k"]).to(DEVICE)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    print(f"  SAE config: {cfg}")

    print("Running SAE encode pass...")
    all_acts = []
    with torch.no_grad():
        for i in tqdm(range(0, len(embeddings), BATCH_SIZE), desc="SAE encoding"):
            batch = embeddings[i : i + BATCH_SIZE]
            acts, _ = sae.encode(batch)
            all_acts.append(acts.cpu().float())

    acts_np = torch.cat(all_acts, dim=0).numpy()

    np.save(OUTPUT_ACTS, acts_np)
    np.save(OUTPUT_IDS, ids)
    print(f"Saved SAE activations {acts_np.shape} → {OUTPUT_ACTS}")
    print(f"Saved IDs             {ids.shape}     → {OUTPUT_IDS}")


if __name__ == "__main__":
    main()
