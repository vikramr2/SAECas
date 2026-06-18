"""
SAECas: cascade trace of a phrase through the citation network.

Usage:
    python saecas.py "CRISPR gene editing"
    python saecas.py "protein folding" --tree
    python saecas.py "protein folding" --top-percentile 97
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

DATA_DIR      = "saes/data"
SAE_ACTS_NPY  = f"{DATA_DIR}/sae_acts.npy"
SAE_ACT_IDS   = f"{DATA_DIR}/sae_act_ids.npy"
CHECKPOINT    = f"{DATA_DIR}/saelens_checkpoint.pt"
METADATA_CSV  = f"{DATA_DIR}/oc_mini_node_metadata.csv"
EDGELIST_CSV  = f"{DATA_DIR}/oc_mini_edgelist.csv"
SPECTER_MODEL = "allenai/specter2_base"


# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    print("Loading SAE activations...")
    acts = np.load(SAE_ACTS_NPY).astype(np.float32)
    ids  = np.load(SAE_ACT_IDS).astype(np.int64)

    print("Loading graph and metadata...")
    meta  = pd.read_csv(METADATA_CSV).set_index("id")
    edges = pd.read_csv(EDGELIST_CSV)
    G     = nx.from_pandas_edgelist(
        edges, source="source", target="target", create_using=nx.DiGraph()
    )
    return acts, ids, G, meta


def load_sae():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    cfg  = ckpt["cfg"]
    sae  = TopKSAE(cfg["d_in"], cfg["d_sae"], cfg["k"]).to(DEVICE)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae


# ── query → feature weights ───────────────────────────────────────────────────

def query_feature_weights(phrase: str, sae, corpus_acts: np.ndarray,
                          tokenizer=None, encoder=None) -> np.ndarray:
    """
    Returns IDF-weighted, L1-normalized feature weight vector [dict_size].
    Optionally accepts pre-loaded tokenizer/encoder to avoid reloading.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(SPECTER_MODEL)
    if encoder is None:
        encoder = AutoModel.from_pretrained(SPECTER_MODEL).to(DEVICE).eval()

    inputs = tokenizer(
        phrase, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        out = encoder(**inputs)
    q_vec = out.last_hidden_state[:, 0, :].cpu().float()
    q_norm = q_vec / (q_vec.norm() + 1e-8)

    with torch.no_grad():
        q_acts, _ = sae.encode(q_norm.to(DEVICE))
    q_acts = q_acts.cpu().numpy()[0]

    active_mask = q_acts > 0
    if not active_mask.any():
        raise ValueError("Query produced no active SAE features.")

    N  = corpus_acts.shape[0]
    df = (corpus_acts[:, active_mask] > 0).sum(axis=0)
    idf = np.log(N / (df + 1.0))

    weights = np.zeros_like(q_acts)
    weights[active_mask] = q_acts[active_mask] * idf
    weights = np.clip(weights, 0, None)

    total = weights.sum()
    if total == 0:
        raise ValueError("All IDF-weighted feature scores are zero.")
    weights /= total
    return weights


# ── node scoring ──────────────────────────────────────────────────────────────

def score_nodes(weights: np.ndarray, acts: np.ndarray, ids: np.ndarray,
                top_percentile: float = 95.0) -> dict:
    active_features = np.nonzero(weights)[0]
    scores = acts[:, active_features] @ weights[active_features]

    threshold = np.percentile(scores, top_percentile)
    scores[scores < threshold] = 0.0

    n_above = int((scores > 0).sum())
    print(f"  {n_above} nodes in top {100 - top_percentile:.1f}% (threshold={threshold:.5f})")
    return {int(ids[i]): float(scores[i]) for i in range(len(ids))}


# ── condensation helpers ──────────────────────────────────────────────────────

def _build_condensation(G: nx.DiGraph, node_scores: dict):
    """
    Returns (cond, scc_score) where cond is the SCC-condensation DAG and
    scc_score[sn] is the sum of member scores for super-node sn.
    """
    scored_nodes = {n for n in G.nodes() if node_scores.get(n, 0.0) > 0}
    reachable = set(scored_nodes)
    for n in scored_nodes:
        reachable.update(G.predecessors(n))
        reachable.update(G.successors(n))
    sub = nx.DiGraph(G.subgraph(reachable))
    cond = nx.condensation(sub)
    scc_score = {
        sn: sum(node_scores.get(m, 0.0) for m in data["members"])
        for sn, data in cond.nodes(data=True)
    }
    return cond, scc_score


def _best_member(cond, sn: int, node_scores: dict) -> int:
    members = cond.nodes[sn]["members"]
    return max(members, key=lambda m: node_scores.get(m, 0.0))


# ── heaviest path ─────────────────────────────────────────────────────────────

def heaviest_path(G: nx.DiGraph, node_scores: dict) -> list[int]:
    """Heaviest-weight path via DP on the SCC condensation."""
    if not any(s > 0 for s in node_scores.values()):
        raise ValueError("No scored nodes found in the graph.")

    cond, scc_score = _build_condensation(G, node_scores)
    topo = list(nx.topological_sort(cond))

    best: dict[int, float] = {sn: scc_score[sn] for sn in cond.nodes()}
    pred: dict[int, int | None] = {sn: None for sn in cond.nodes()}

    for sn in topo:
        for u in cond.predecessors(sn):
            candidate = best[u] + scc_score[sn]
            if candidate > best[sn]:
                best[sn] = candidate
                pred[sn] = u

    end_sn = max(best, key=best.__getitem__)
    sn_path: list[int] = []
    cur: int | None = end_sn
    while cur is not None:
        sn_path.append(cur)
        cur = pred[cur]
    sn_path.reverse()

    return [_best_member(cond, sn, node_scores) for sn in sn_path]


# ── heaviest tree ─────────────────────────────────────────────────────────────

def heaviest_tree(G: nx.DiGraph, node_scores: dict,
                  max_nodes: int = 30) -> nx.DiGraph:
    """
    Grow a directed tree rooted at the highest-scoring node by repeatedly
    expanding the frontier: at each step add the unvisited neighbor with the
    highest score that is reachable from (or can reach) any node already in
    the tree.  Stops when max_nodes is reached or no scored neighbor remains.

    Returns a DiGraph that is a directed subtree of G.
    """
    scored = {n: s for n, s in node_scores.items() if s > 0 and G.has_node(n)}
    if not scored:
        raise ValueError("No scored nodes found in the graph.")

    root = max(scored, key=scored.__getitem__)
    tree = nx.DiGraph()
    tree.add_node(root)
    in_tree = {root}

    for _ in range(max_nodes - 1):
        best_candidate, best_score, best_parent, edge_dir = None, -1.0, None, None

        for n in list(in_tree):
            for nb in G.successors(n):
                s = scored.get(nb, 0.0)
                if nb not in in_tree and s > best_score:
                    best_candidate, best_score, best_parent, edge_dir = nb, s, n, "out"
            for nb in G.predecessors(n):
                s = scored.get(nb, 0.0)
                if nb not in in_tree and s > best_score:
                    best_candidate, best_score, best_parent, edge_dir = nb, s, n, "in"

        if best_candidate is None:
            break

        in_tree.add(best_candidate)
        if edge_dir == "out":
            tree.add_edge(best_parent, best_candidate)
        else:
            tree.add_edge(best_candidate, best_parent)

    return tree


# ── display ───────────────────────────────────────────────────────────────────

def _node_label(node: int, node_scores: dict, meta: pd.DataFrame,
                width: int = 65) -> str:
    score = node_scores.get(node, 0.0)
    if node in meta.index:
        title = str(meta.loc[node, "title"])
        title = (title[:width] + "…") if len(title) > width else title
    else:
        title = str(node)
    return f"[score={score:.4f}] {title}"


def print_path_trace(path: list[int], node_scores: dict, meta: pd.DataFrame,
                     weights: np.ndarray):
    print("\n" + "=" * 72)
    print(f"CASCADE PATH  ({len(path)} nodes)")
    print("=" * 72)
    for rank, node in enumerate(path):
        arrow = "→ " if rank > 0 else "  "
        print(f"{arrow}[{rank+1:2d}] id={node}")
        print(f"       {_node_label(node, node_scores, meta)}")
    _print_features(weights)


def print_tree_trace(tree: nx.DiGraph, node_scores: dict, meta: pd.DataFrame,
                     weights: np.ndarray):
    # Find root: node with in-degree 0 in the tree, or highest-scored
    roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
    root  = max(roots, key=lambda n: node_scores.get(n, 0.0)) if roots else \
            max(tree.nodes(), key=lambda n: node_scores.get(n, 0.0))

    print("\n" + "=" * 72)
    print(f"CASCADE TREE  ({tree.number_of_nodes()} nodes)")
    print("=" * 72)

    undirected = tree.to_undirected()

    def _print_subtree(node, parent, prefix="", is_last=True):
        connector = "└─ " if is_last else "├─ "
        print(f"{prefix}{connector}id={node}")
        child_prefix = prefix + ("   " if is_last else "│  ")
        print(f"{child_prefix}{_node_label(node, node_scores, meta)}")
        children = [nb for nb in undirected.neighbors(node) if nb != parent]
        for i, child in enumerate(children):
            _print_subtree(child, node, child_prefix, i == len(children) - 1)

    _print_subtree(root, None)
    _print_features(weights)


def _print_features(weights: np.ndarray):
    print("\nTop features driving this result:")
    for fi in np.argsort(weights)[::-1][:10]:
        if weights[fi] > 0:
            print(f"  feature {fi:4d}  weight={weights[fi]:.4f}")
    print("=" * 72)


# ── core pipeline (importable by the visualizer) ──────────────────────────────

def run_cascade(phrase: str, acts: np.ndarray, ids: np.ndarray,
                G: nx.DiGraph, meta: pd.DataFrame, sae,
                top_percentile: float = 95.0, tree_mode: bool = False,
                max_tree_nodes: int = 30,
                tokenizer=None, encoder=None):
    """
    Full pipeline. Returns (result, node_scores, weights) where result is
    either a list[int] (path) or nx.DiGraph (tree).
    """
    weights     = query_feature_weights(phrase, sae, acts, tokenizer, encoder)
    node_scores = score_nodes(weights, acts, ids, top_percentile)

    print("Running DP...")
    if tree_mode:
        result = heaviest_tree(G, node_scores, max_nodes=max_tree_nodes)
    else:
        result = heaviest_path(G, node_scores)

    return result, node_scores, weights


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAECas cascade trace")
    parser.add_argument("phrase", help="Phrase to trace")
    parser.add_argument("--top-percentile", type=float, default=95.0,
                        help="Score percentile cutoff (default 95 → top 5%%)")
    parser.add_argument("--tree", action="store_true",
                        help="Show a branching tree instead of a linear path")
    parser.add_argument("--max-tree-nodes", type=int, default=30,
                        help="Max nodes in tree mode (default 30)")
    args = parser.parse_args()

    acts, ids, G, meta = load_data()
    sae = load_sae()

    result, node_scores, weights = run_cascade(
        args.phrase, acts, ids, G, meta, sae,
        top_percentile=args.top_percentile,
        tree_mode=args.tree,
        max_tree_nodes=args.max_tree_nodes,
    )

    if args.tree:
        assert isinstance(result, nx.DiGraph)
        print_tree_trace(result, node_scores, meta, weights)
    else:
        assert isinstance(result, list)
        print_path_trace(result, node_scores, meta, weights)


if __name__ == "__main__":
    main()