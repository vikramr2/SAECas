"""
SAECas interactive visualizer.

Run:
    python app.py
Then open http://localhost:8050
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, "saes/model")
from saelens import DEVICE
from saecas import (
    load_data, load_sae, run_cascade,
    SPECTER_MODEL,
)

# ── startup ───────────────────────────────────────────────────────────────────

print("Loading data...")
ACTS, IDS, G, META = load_data()
print("Loading SAE...")
SAE = load_sae()
print("Loading SPECTER2 (shared across queries)...")
TOKENIZER = AutoTokenizer.from_pretrained(SPECTER_MODEL)
ENCODER   = AutoModel.from_pretrained(SPECTER_MODEL).to(DEVICE).eval()
print("Ready.")

# ── graph figure builders ─────────────────────────────────────────────────────

def _node_info(node: int, node_scores: dict) -> tuple[str, float]:
    score = node_scores.get(node, 0.0)
    if node in META.index:
        title = str(META.loc[node, "title"])
        label = (title[:80] + "…") if len(title) > 80 else title
    else:
        label = str(node)
    return label, score


def path_figure(path: list[int], node_scores: dict) -> go.Figure:
    if not path:
        return go.Figure()

    # Lay out as a horizontal chain
    pos = {node: (i, 0) for i, node in enumerate(path)}

    edge_x, edge_y, edge_ax, edge_ay = [], [], [], []
    for u, v in zip(path, path[1:]):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_ax.append(x0); edge_ay.append(y0)
        edge_x.append(x1);  edge_y.append(y1)

    scores  = [node_scores.get(n, 0.0) for n in path]
    labels, hovers = [], []
    for n in path:
        label, score = _node_info(n, node_scores)
        labels.append(label)
        hovers.append(f"<b>id {n}</b><br>{label}<br>score: {score:.4f}")

    fig = go.Figure()
    for ax, ay, x, y in zip(edge_ax, edge_ay, edge_x, edge_y):
        fig.add_annotation(
            x=x, y=y, ax=ax, ay=ay, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=1.5,
            arrowcolor="#888",
        )

    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in path],
        y=[pos[n][1] for n in path],
        mode="markers+text",
        text=[f"{i+1}" for i in range(len(path))],
        textposition="top center",
        hovertext=hovers,
        hoverinfo="text",
        marker=dict(
            size=22,
            color=scores,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Score", thickness=12),
            line=dict(width=1.5, color="#333"),
        ),
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x"),
        hovermode="closest",
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
        height=320,
    )
    return fig


def tree_figure(tree: nx.DiGraph, node_scores: dict) -> go.Figure:
    if tree.number_of_nodes() == 0:
        return go.Figure()

    # BFS layout on the underlying undirected tree so every node gets a position
    # regardless of edge direction (heaviest_tree adds both "in" and "out" edges).
    undirected = tree.to_undirected()
    roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
    root  = max(roots, key=lambda n: node_scores.get(n, 0.0)) if roots else \
            max(tree.nodes(), key=lambda n: node_scores.get(n, 0.0))

    level_buckets: dict[int, list] = {0: [root]}
    levels: dict[int, int] = {root: 0}
    visited = {root}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nb in undirected.neighbors(node):
            if nb not in visited:
                visited.add(nb)
                lv = levels[node] + 1
                levels[nb] = lv
                level_buckets.setdefault(lv, []).append(nb)
                queue.append(nb)

    # Any node not reached (disconnected component) gets placed below the tree
    max_lv = max(level_buckets) if level_buckets else 0
    orphans = [n for n in tree.nodes() if n not in visited]
    if orphans:
        level_buckets[max_lv + 1] = orphans

    pos: dict[int, tuple[float, float]] = {}
    for lv, bucket in level_buckets.items():
        for i, n in enumerate(bucket):
            pos[n] = (i - (len(bucket) - 1) / 2.0, -lv)

    nodes = list(tree.nodes())
    scores  = [node_scores.get(n, 0.0) for n in nodes]
    hovers  = []
    for n in nodes:
        label, score = _node_info(n, node_scores)
        hovers.append(f"<b>id {n}</b><br>{label}<br>score: {score:.4f}")

    edge_x, edge_y = [], []
    for u, v in tree.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.2, color="#aaa"), hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers",
        hovertext=hovers,
        hoverinfo="text",
        marker=dict(
            size=18,
            color=scores,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Score", thickness=12),
            line=dict(width=1.5, color="#333"),
        ),
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
        height=480,
        showlegend=False,
    )
    return fig


def detail_table(path_or_tree, node_scores: dict, tree_mode: bool) -> list:
    nodes = list(path_or_tree.nodes()) if tree_mode else path_or_tree
    rows  = [html.Tr([
        html.Th("#",     style=TH),
        html.Th("Score", style=TH),
        html.Th("Paper", style=TH),
    ])]
    for i, node in enumerate(nodes):
        label, score = _node_info(node, node_scores)
        rows.append(html.Tr([
            html.Td(str(i + 1), style=TD),
            html.Td(f"{score:.4f}", style=TD),
            html.Td(label, style={**TD, "fontSize": "12px"}),
        ], style={"background": "#fff" if i % 2 == 0 else "#f5f5f5"}))
    return rows


TH = {"padding": "6px 10px", "background": "#e8e8e8",
      "borderBottom": "1px solid #ccc", "textAlign": "left", "fontSize": "13px"}
TD = {"padding": "5px 10px", "verticalAlign": "top"}

# ── Dash app ──────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="SAECas")

SIDEBAR = {"width": "320px", "minWidth": "260px", "display": "flex",
           "flexDirection": "column", "gap": "12px", "padding": "16px",
           "background": "#1e1e2e", "color": "#cdd6f4", "height": "100vh",
           "overflowY": "auto", "boxSizing": "border-box"}

MAIN = {"flex": "1", "display": "flex", "flexDirection": "column",
        "padding": "16px", "gap": "12px", "overflowY": "auto",
        "background": "#f0f0f7"}

LABEL = {"fontSize": "12px", "color": "#a6adc8", "marginBottom": "2px"}

INPUT_STYLE = {"width": "100%", "padding": "8px", "borderRadius": "6px",
               "border": "1px solid #45475a", "background": "#313244",
               "color": "#cdd6f4", "fontSize": "14px", "boxSizing": "border-box"}

BTN = {"width": "100%", "padding": "9px", "borderRadius": "6px",
       "background": "#cba6f7", "color": "#1e1e2e", "border": "none",
       "fontWeight": "bold", "fontSize": "14px", "cursor": "pointer"}

app.layout = html.Div(
    style={"display": "flex", "fontFamily": "Inter, sans-serif",
           "height": "100vh", "overflow": "hidden"},
    children=[

        # ── sidebar ──────────────────────────────────────────────────────────
        html.Div(style=SIDEBAR, children=[
            html.H2("SAECas", style={"margin": "0 0 4px 0", "fontSize": "22px",
                                     "color": "#cba6f7"}),
            html.P("Citation cascade explorer", style={"margin": "0 0 12px 0",
                   "fontSize": "12px", "color": "#6c7086"}),

            html.Div([
                html.Div("Query phrase", style=LABEL),
                dcc.Textarea(
                    id="query-input",
                    placeholder="e.g. protein folding neural networks",
                    style={**INPUT_STYLE, "height": "72px", "resize": "vertical"},
                ),
            ]),

            html.Div([
                html.Div("Mode", style=LABEL),
                dcc.RadioItems(
                    id="mode-toggle",
                    options=[
                        {"label": " Path  (linear chain)", "value": "path"},
                        {"label": " Tree  (branching)",    "value": "tree"},
                    ],
                    value="path",
                    labelStyle={"display": "block", "marginBottom": "4px",
                                "fontSize": "13px"},
                    inputStyle={"marginRight": "6px"},
                ),
            ]),

            html.Div([
                html.Div(id="percentile-label", style=LABEL),
                dcc.Slider(
                    id="percentile-slider",
                    min=80, max=99, step=1, value=95,
                    marks={80: "80", 90: "90", 95: "95", 99: "99"},
                    tooltip={"always_visible": False},
                ),
            ]),

            html.Div(id="tree-nodes-div", children=[
                html.Div(id="tree-nodes-label", style=LABEL),
                dcc.Slider(
                    id="tree-nodes-slider",
                    min=5, max=60, step=5, value=20,
                    marks={5: "5", 20: "20", 40: "40", 60: "60"},
                    tooltip={"always_visible": False},
                ),
            ]),

            html.Button("Run cascade", id="run-btn", n_clicks=0, style=BTN),

            html.Div(id="status-msg",
                     style={"fontSize": "12px", "color": "#a6e3a1",
                            "minHeight": "18px"}),

            html.Hr(style={"borderColor": "#45475a", "margin": "4px 0"}),
            html.Div("Top features", style={**LABEL, "marginBottom": "4px"}),
            html.Div(id="feature-list",
                     style={"fontSize": "12px", "color": "#89b4fa",
                            "lineHeight": "1.7"}),
        ]),

        # ── main panel ───────────────────────────────────────────────────────
        html.Div(style=MAIN, children=[
            dcc.Graph(id="cascade-graph",
                      config={"displayModeBar": False},
                      style={"background": "#fafafa", "borderRadius": "10px",
                             "boxShadow": "0 1px 4px rgba(0,0,0,.12)"}),

            html.Div(
                style={"overflowY": "auto", "maxHeight": "340px",
                       "borderRadius": "8px", "boxShadow": "0 1px 4px rgba(0,0,0,.1)",
                       "background": "#fff"},
                children=[
                    html.Table(
                        id="detail-table",
                        style={"width": "100%", "borderCollapse": "collapse"},
                    )
                ]
            ),
        ]),
    ]
)


# ── callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("percentile-label", "children"),
    Input("percentile-slider", "value"),
)
def update_percentile_label(val):
    return f"Score percentile cutoff — top {100 - val}%"


@app.callback(
    Output("tree-nodes-label", "children"),
    Input("tree-nodes-slider", "value"),
)
def update_tree_nodes_label(val):
    return f"Max tree nodes — {val}"


@app.callback(
    Output("tree-nodes-div", "style"),
    Input("mode-toggle", "value"),
)
def toggle_tree_slider(mode):
    base = {"marginTop": "4px"}
    if mode == "tree":
        return base
    return {**base, "opacity": "0.35", "pointerEvents": "none"}


@app.callback(
    Output("cascade-graph",  "figure"),
    Output("detail-table",   "children"),
    Output("feature-list",   "children"),
    Output("status-msg",     "children"),
    Input("run-btn", "n_clicks"),
    State("query-input",      "value"),
    State("mode-toggle",      "value"),
    State("percentile-slider","value"),
    State("tree-nodes-slider","value"),
    prevent_initial_call=True,
)
def run_query(_, phrase, mode, percentile, max_tree_nodes):
    if not phrase or not phrase.strip():
        return dash.no_update, dash.no_update, dash.no_update, "Enter a query."

    tree_mode = (mode == "tree")
    try:
        result, node_scores, weights = run_cascade(
            phrase.strip(), ACTS, IDS, G, META, SAE,
            top_percentile=float(percentile),
            tree_mode=tree_mode,
            max_tree_nodes=int(max_tree_nodes),
            tokenizer=TOKENIZER,
            encoder=ENCODER,
        )
    except Exception as e:
        return go.Figure(), [], [], f"Error: {e}"

    if tree_mode:
        assert isinstance(result, nx.DiGraph)
        fig    = tree_figure(result, node_scores)
        n_shown = result.number_of_nodes()
    else:
        assert isinstance(result, list)
        fig    = path_figure(result, node_scores)
        n_shown = len(result)

    table_rows = detail_table(result, node_scores, tree_mode)

    top_feats = np.argsort(weights)[::-1][:8]
    feat_items = [
        html.Div(f"feat {fi}  {weights[fi]:.4f}",
                 style={"fontFamily": "monospace"})
        for fi in top_feats if weights[fi] > 0
    ]

    status = f"{'Tree' if tree_mode else 'Path'} — {n_shown} nodes shown"
    return fig, table_rows, feat_items, status


if __name__ == "__main__":
    app.run(debug=False, port=8050)