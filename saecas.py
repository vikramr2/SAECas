"""
SAECas: cascade trace of a phrase through the citation network.

Algorithm:
  1. Embed the query phrase with SPECTER2, run through the SAE encoder to get
     a sparse feature vector.
  2. Normalize that feature vector to sum to 1 → feature weights w_f.
  3. Score every node as  score(v) = sum_f  w_f * acts[v, f]
     (linear combination of feature weights times the node's SAE activations).
  4. Find the path through the DAG (citation graph) that maximises the sum of
     node scores — i.e. the "heaviest path" — using dynamic programming.
  5. Print the cascade trace.

Usage:
    python saecas.py "CRISPR gene editing"
    python saecas.py "protein folding neural networks" --top-k 50
"""

import sys
import argparse
import numpy as np
import torch
import pandas as pd
import networkx as nx
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, "saes/model")
from saelens import TopKSAE, DEVICE

# ── paths ─────────────────────────────────────────────────────────────────────

DATA_DIR       = "saes/data"
EMBEDDINGS_NPY = f"{DATA_DIR}/embeddings.npy"
IDS_NPY        = f"{DATA_DIR}/embedding_ids.npy"
SAE_ACTS_NPY   = f"{DATA_DIR}/sae_acts.npy"
SAE_ACT_IDS    = f"{DATA_DIR}/sae_act_ids.npy"
CHECKPOINT     = f"{DATA_DIR}/saelens_checkpoint.pt"
METADATA_CSV   = f"{DATA_DIR}/oc_mini_node_metadata.csv"
EDGELIST_CSV   = f"{DATA_DIR}/oc_mini_edgelist.csv"
SPECTER_MODEL  = "allenai/specter2_base"


# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    print("Loading SAE activations...")
    acts     = np.load(SAE_ACTS_NPY).astype(np.float32)   # [N, dict_size]
    ids      = np.load(SAE_ACT_IDS).astype(np.int64)      # [N]
    id_to_row = {int(pid): i for i, pid in enumerate(ids)}

    print("Loading graph and metadata...")
    meta  = pd.read_csv(METADATA_CSV).set_index("id")
    edges = pd.read_csv(EDGELIST_CSV)
    G     = nx.from_pandas_edgelist(
        edges, source="source", target="target", create_using=nx.DiGraph()
    )
    return acts, ids, id_to_row, G, meta


def load_sae():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    cfg  = ckpt["cfg"]
    sae  = TopKSAE(cfg["d_in"], cfg["d_sae"], cfg["k"]).to(DEVICE)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae


def normalize_embedding(x: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(x.astype(np.float32))
    t = t - t.mean()
    t = t / (t.norm() + 1e-8)
    return t


# ── query → feature weights ───────────────────────────────────────────────────

def query_feature_weights(phrase: str, sae, corpus_acts: np.ndarray) -> np.ndarray:
    """
    Embed phrase with SPECTER2, encode with SAE, then apply IDF weighting so
    that features firing on most nodes are downweighted.

    weight_f = query_act_f * log(N / df_f)   for active features
    Then normalize to sum to 1.
    Returns weight vector of shape [dict_size].
    """
    print("Embedding query with SPECTER2...")
    tokenizer = AutoTokenizer.from_pretrained(SPECTER_MODEL)
    encoder   = AutoModel.from_pretrained(SPECTER_MODEL).to(DEVICE).eval()

    inputs = tokenizer(
        phrase, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        out = encoder(**inputs)
    q_vec = out.last_hidden_state[:, 0, :].cpu().float()  # [1, 768]
    del encoder

    q_norm = q_vec / (q_vec.norm() + 1e-8)

    print("Encoding query through SAE...")
    with torch.no_grad():
        acts, _ = sae.encode(q_norm.to(DEVICE))
    acts = acts.cpu().numpy()[0]  # [dict_size]

    active_mask = acts > 0
    n_active = int(active_mask.sum())
    print(f"  {n_active} active features")

    if n_active == 0:
        raise ValueError("Query produced no active SAE features.")

    # IDF: log(N / document_frequency_f) for each active feature
    N = corpus_acts.shape[0]
    df = (corpus_acts[:, active_mask] > 0).sum(axis=0)  # [n_active]
    idf = np.log(N / (df + 1.0))                        # +1 avoids divide-by-zero

    weights = np.zeros_like(acts)
    weights[active_mask] = acts[active_mask] * idf
    weights = np.clip(weights, 0, None)

    total = weights.sum()
    if total == 0:
        raise ValueError("All IDF-weighted feature scores are zero.")
    weights /= total

    n_idf_active = int((weights > 0).sum())
    print(f"  {n_idf_active} features after IDF weighting")
    return weights


# ── node scoring ──────────────────────────────────────────────────────────────

def score_nodes(weights: np.ndarray, acts: np.ndarray, ids: np.ndarray,
                top_percentile: float = 95.0) -> dict:
    """
    score(v) = sum_f  w_f * acts[v, f]
    Only nodes in the top (100 - top_percentile) percent are kept; the rest
    are zeroed out so the heaviest-path DP doesn't degenerate into a
    near-Hamiltonian path. Percentile-based cutoff is robust to score skew.
    Returns {paper_id: score}.
    """
    active_features = np.nonzero(weights)[0]
    scores = acts[:, active_features] @ weights[active_features]  # [N]

    threshold = np.percentile(scores, top_percentile)
    scores[scores < threshold] = 0.0

    n_above = int((scores > 0).sum())
    print(f"  {n_above} nodes in top {100 - top_percentile:.1f}% (threshold={threshold:.5f})")
    return {int(ids[i]): float(scores[i]) for i in range(len(ids))}


# ── heaviest path via DP ──────────────────────────────────────────────────────

def heaviest_path(G: nx.DiGraph, node_scores: dict) -> list:
    """
    Find the path in G that maximises the sum of node_scores along it.

    The citation graph contains cycles (mutual citations), so we cannot do
    topological sort directly. We condense SCCs into super-nodes first,
    making the condensation a proper DAG, run DP there, then expand the
    winning super-node sequence back to individual nodes.
    """
    scored_nodes = {n for n in G.nodes() if node_scores.get(n, 0.0) > 0}
    if not scored_nodes:
        raise ValueError("No scored nodes found in the graph.")

    # Restrict to scored nodes + their graph neighbors to keep DP tractable.
    reachable = set(scored_nodes)
    for n in scored_nodes:
        reachable.update(G.predecessors(n))
        reachable.update(G.successors(n))
    sub = nx.DiGraph(G.subgraph(reachable))

    # Condense SCCs → DAG of super-nodes.
    # condensation() returns a DAG where each node has a 'members' attribute.
    cond = nx.condensation(sub)

    # Score each super-node as the sum of its members' scores.
    scc_score = {}
    for sn, data in cond.nodes(data=True):
        scc_score[sn] = sum(node_scores.get(m, 0.0) for m in data["members"])

    # DP on the condensation DAG (guaranteed acyclic).
    topo = list(nx.topological_sort(cond))
    best: dict[int, float] = {sn: scc_score[sn] for sn in cond.nodes()}
    pred: dict[int, int | None] = {sn: None for sn in cond.nodes()}

    for sn in topo:
        for u in cond.predecessors(sn):
            candidate = best[u] + scc_score[sn]
            if candidate > best[sn]:
                best[sn] = candidate
                pred[sn] = u

    # Trace back the best super-node path.
    end_sn = max(best, key=best.__getitem__)
    sn_path = []
    cur = end_sn
    while cur is not None:
        sn_path.append(cur)
        cur = pred[cur]
    sn_path.reverse()

    # Expand each super-node to its best-scored member.
    path = []
    for sn in sn_path:
        members = cond.nodes[sn]["members"]
        best_member = max(members, key=lambda m: node_scores.get(m, 0.0))
        path.append(best_member)

    return path


# ── display ───────────────────────────────────────────────────────────────────

def print_trace(path: list, node_scores: dict, meta: pd.DataFrame, weights: np.ndarray):
    active_features = np.nonzero(weights)[0]
    print("\n" + "=" * 70)
    print(f"CASCADE TRACE  ({len(path)} nodes)")
    print("=" * 70)
    for rank, node in enumerate(path):
        score = node_scores.get(node, 0.0)
        if node in meta.index:
            title = meta.loc[node, "title"]
            year  = meta.loc[node].get("year", "")
            title_str = (title[:65] + "…") if len(title) > 65 else title
            label = f"{title_str}  [{year}]"
        else:
            label = str(node)
        arrow = "→ " if rank > 0 else "  "
        print(f"{arrow}[{rank+1:2d}] id={node}  score={score:.4f}")
        print(f"       {label}")
    print("=" * 70)
    print(f"\nTop active features driving this trace:")
    top_feat_idx = np.argsort(weights)[::-1][:10]
    for fi in top_feat_idx:
        if weights[fi] > 0:
            print(f"  feature {fi:4d}  weight={weights[fi]:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAECas cascade trace")
    parser.add_argument("phrase", help="Input phrase to trace through the network")
    parser.add_argument("--top-percentile", type=float, default=95.0,
                        help="Keep only nodes scoring above this percentile (default: 95 → top 5%%)")
    args = parser.parse_args()

    acts, ids, id_to_row, G, meta = load_data()
    sae = load_sae()

    weights     = query_feature_weights(args.phrase, sae, acts)
    node_scores = score_nodes(weights, acts, ids, args.top_percentile)

    n_nonzero = sum(1 for s in node_scores.values() if s > 0)
    print(f"  {n_nonzero} nodes have nonzero score")

    print("Running DP to find heaviest path...")
    path = heaviest_path(G, node_scores)

    print_trace(path, node_scores, meta, weights)


if __name__ == "__main__":
    main()
