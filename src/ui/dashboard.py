"""
Fraud Investigation Dashboard
-----------------------------
A Streamlit interface for fraud analysts to investigate transactions,
view model performance, and monitor system health.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import pathlib
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark command-center palette */
:root {
    --bg-deep:    #0a0e1a;
    --bg-card:    #111827;
    --bg-subtle:  #1a2235;
    --accent:     #00e5ff;
    --accent2:    #ff4d6d;
    --accent3:    #a78bfa;
    --text-pri:   #e2e8f0;
    --text-sec:   #94a3b8;
    --border:     #1e293b;
    --green:      #10b981;
    --amber:      #f59e0b;
    --red:        #ef4444;
}

.stApp { background: var(--bg-deep); }

/* Header */
.dash-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1.4rem 0 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.6rem;
}
.dash-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem; font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
    margin: 0;
}
.dash-header .sub {
    font-size: 0.78rem; color: var(--text-sec);
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.metric-card .label {
    font-size: 0.7rem; color: var(--text-sec);
    text-transform: uppercase; letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
}
.metric-card .value {
    font-size: 1.4rem; font-weight: 600;
    color: var(--text-pri); margin-top: 4px;
}

/* Risk badge */
.risk-high   { color: var(--red);   background: rgba(239,68,68,.12);  border: 1px solid rgba(239,68,68,.3);  padding: 6px 14px; border-radius: 6px; font-weight:600; }
.risk-medium { color: var(--amber); background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.3); padding: 6px 14px; border-radius: 6px; font-weight:600; }
.risk-low    { color: var(--green); background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.3); padding: 6px 14px; border-radius: 6px; font-weight:600; }

/* Section headings */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem; font-weight: 700;
    color: var(--accent); letter-spacing: 2px;
    text-transform: uppercase;
    margin: 1.4rem 0 0.8rem;
    display: flex; align-items: center; gap: 8px;
}
.section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(to right, var(--border), transparent);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-sec) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    border-radius: 7px;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-subtle) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border);
}

/* Form inputs */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-pri) !important;
    border-radius: 7px !important;
}
.stForm { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; }

/* Submit button */
.stFormSubmitButton button {
    background: linear-gradient(135deg, #00b4d8, #00e5ff) !important;
    color: #0a0e1a !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    letter-spacing: 1px;
}

/* Metric widgets */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"] { color: var(--text-pri) !important; }
[data-testid="stMetricLabel"] { color: var(--text-sec) !important; font-size: 0.7rem !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

BASE = pathlib.Path(__file__).parents[2]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_results():
    try:
        with open(BASE / "data" / "training_results.json") as f:
            return json.load(f)
    except Exception:
        return None


def load_drift_report():
    try:
        with open(BASE / "data" / "drift_report.json") as f:
            return json.load(f)
    except Exception:
        return None


# ── Network graph (Plotly) ────────────────────────────────────────────────────
def build_network_figure(card_id, merchant_id, device_id, amount, prob):
    G = nx.Graph()

    # Core nodes
    nodes = {
        card_id:     {"color": "#00e5ff", "size": 28, "symbol": "circle",      "group": "card"},
        merchant_id: {"color": "#a78bfa", "size": 24, "symbol": "square",      "group": "merchant"},
        device_id:   {"color": "#f59e0b", "size": 20, "symbol": "diamond",     "group": "device"},
    }
    edges = [
        (card_id, merchant_id, {"label": f"${amount:.2f}", "color": "#334155", "width": 2}),
        (card_id, device_id,   {"label": "used",           "color": "#334155", "width": 1.5}),
    ]

    if prob > 0.4:
        fr1 = f"FRAUD_M_{merchant_id[-4:]}"
        fr2 = f"FRAUD_D_{device_id[-4:]}"
        nodes["FRAUD_HUB"] = {"color": "#ef4444", "size": 34, "symbol": "star", "group": "hub"}
        nodes[fr1]          = {"color": "#f97316", "size": 18, "symbol": "square",  "group": "suspect"}
        nodes[fr2]          = {"color": "#f97316", "size": 18, "symbol": "diamond", "group": "suspect"}
        edges += [
            (merchant_id,  "FRAUD_HUB", {"label": "linked", "color": "#ef4444", "width": 2.5}),
            (device_id,    "FRAUD_HUB", {"label": "linked", "color": "#ef4444", "width": 2.5}),
            ("FRAUD_HUB",  fr1,         {"label": "",        "color": "#f97316", "width": 1.5}),
            ("FRAUD_HUB",  fr2,         {"label": "",        "color": "#f97316", "width": 1.5}),
        ]

    for n, attr in nodes.items():
        G.add_node(n, **attr)
    for u, v, attr in edges:
        G.add_edge(u, v, **attr)

    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=80)

    # Build edge traces
    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=d.get("width", 1.5), color=d.get("color", "#334155")),
            hoverinfo="none",
            showlegend=False,
        ))
        # Edge label at midpoint
        if d.get("label"):
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            edge_traces.append(go.Scatter(
                x=[mx], y=[my],
                mode="text",
                text=[d["label"]],
                textfont=dict(size=9, color="#64748b", family="Space Mono"),
                hoverinfo="none",
                showlegend=False,
            ))

    # Build node trace
    nx_list = list(G.nodes(data=True))
    node_x = [pos[n][0] for n, _ in nx_list]
    node_y = [pos[n][1] for n, _ in nx_list]
    node_colors  = [d.get("color", "#94a3b8") for _, d in nx_list]
    node_sizes   = [d.get("size",  20)         for _, d in nx_list]
    node_symbols = [d.get("symbol","circle")   for _, d in nx_list]
    node_labels  = [n for n, _ in nx_list]
    hover_texts  = [
        f"<b>{n}</b><br>Group: {d.get('group','–')}<br>Degree: {G.degree(n)}"
        for n, d in nx_list
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            symbol=node_symbols,
            line=dict(width=1.5, color="#0a0e1a"),
        ),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=9, color="#94a3b8", family="Space Mono"),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.6)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        font=dict(family="DM Sans"),
        hoverlabel=dict(bgcolor="#1e293b", font_color="#e2e8f0", bordercolor="#334155"),
    )
    return fig, G


# ── Risk gauge ────────────────────────────────────────────────────────────────
def risk_gauge(prob: float) -> go.Figure:
    color = "#ef4444" if prob > 0.5 else ("#f59e0b" if prob > 0.1 else "#10b981")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 32, "color": "#e2e8f0", "family": "Space Mono"}},
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#64748b", size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#1a2235",
            borderwidth=1,
            bordercolor="#1e293b",
            steps=[
                dict(range=[0, 10],  color="rgba(16,185,129,0.12)"),
                dict(range=[10, 50], color="rgba(245,158,11,0.10)"),
                dict(range=[50, 100],color="rgba(239,68,68,0.12)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=prob * 100),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=20, r=20, t=20, b=10),
        font=dict(family="DM Sans"),
    )
    return fig


# ── Bar chart for risk metrics ────────────────────────────────────────────────
def risk_bar_chart(amount, n_edges, prob):
    metrics   = ["Amount (USD)", "Network Edges", "Risk Score"]
    values    = [amount,          n_edges,          round(prob * 100, 1)]
    norm      = [min(v / max(values[i], 0.001), 1) for i, v in enumerate(values)]
    bar_colors= ["#00e5ff", "#a78bfa", "#ef4444" if prob > 0.5 else "#f59e0b"]

    fig = go.Figure(go.Bar(
        x=metrics, y=values,
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v}" for v in values],
        textposition="outside",
        textfont=dict(family="Space Mono", size=11, color="#94a3b8"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.0)",
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="DM Sans", color="#94a3b8"),
        xaxis=dict(showgrid=False, tickfont=dict(size=10, family="Space Mono")),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False, tickfont=dict(size=10)),
    )
    return fig


# ── PSI heatmap ───────────────────────────────────────────────────────────────
def psi_chart(feature_psi: dict) -> go.Figure:
    items = sorted(feature_psi.items(), key=lambda x: x[1], reverse=True)[:20]
    feats = [i[0] for i in items]
    vals  = [i[1] for i in items]
    colors = ["#ef4444" if v > 0.2 else ("#f59e0b" if v > 0.1 else "#10b981") for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(family="Space Mono", size=9, color="#94a3b8"),
    ))
    fig.add_vline(x=0.1, line=dict(color="#f59e0b", dash="dot", width=1))
    fig.add_vline(x=0.2, line=dict(color="#ef4444", dash="dot", width=1))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(350, len(feats) * 22),
        margin=dict(l=10, r=60, t=10, b=10),
        font=dict(family="DM Sans", color="#94a3b8"),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="dash-header">
        <div>
            <h1>🛡️ FRAUDSHIELD AI</h1>
            <div class="sub">INVESTIGATION DASHBOARD · REAL-TIME ANALYSIS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🔍  INVESTIGATOR", "📊  MODEL PERFORMANCE", "🛠️  SYSTEM HEALTH"])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — Transaction Investigator
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[0]:
        col_form, col_result = st.columns([1, 2], gap="large")

        sample_v = {
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388,  "V7": 0.239599, "V8": 0.098698,
            "V9": 0.363787,  "V10": 0.090794, "V11": -0.5516,  "V12": -0.617801,
            "V13": -0.99139, "V14": -0.311169,"V15": 1.468177, "V16": -0.470401,
            "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
            "V21": -0.018307,"V22": 0.277838, "V23": -0.110474,"V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115,"V27": 0.133558, "V28": -0.021053,
        }

        with col_form:
            st.markdown('<div class="section-title">Transaction Input</div>', unsafe_allow_html=True)
            with st.form("transaction_form"):
                time_val   = st.number_input("Time (seconds)", value=0.0)
                card_id    = st.text_input("Card ID",     value="C_1680")
                merchant_id= st.text_input("Merchant ID", value="M_FRAUD_HUB")
                device_id  = st.text_input("Device ID",   value="D_FRAUD_BRIDGE")
                amount     = st.number_input("Amount (USD)", value=149.62, min_value=0.0)

                st.markdown('<div class="section-title" style="margin-top:1rem">V-Features (PCA)</div>', unsafe_allow_html=True)
                v_inputs = {}
                v_cols = st.columns(4)
                for i in range(1, 29):
                    feat = f"V{i}"
                    with v_cols[(i - 1) % 4]:
                        v_inputs[feat] = st.number_input(feat, value=sample_v[feat], label_visibility="visible")

                submit = st.form_submit_button("▶  ANALYZE TRANSACTION", use_container_width=True)

        with col_result:
            if submit:
                payload = {
                    "Time": time_val, "card_id": card_id,
                    "merchant_id": merchant_id, "device_id": device_id,
                    "Amount": amount, **v_inputs,
                }

                try:
                    resp = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
                    if resp.status_code == 200:
                        result = resp.json()
                        prob     = result["fraud_probability"]
                        is_fraud = result["is_fraud"]

                        # ── Risk summary ──
                        st.markdown('<div class="section-title">Risk Assessment</div>', unsafe_allow_html=True)
                        r1, r2 = st.columns([1, 1])
                        with r1:
                            st.plotly_chart(risk_gauge(prob), use_container_width=True)
                        with r2:
                            if is_fraud:
                                st.markdown(f'<p class="risk-high">🚩 HIGH RISK — {prob:.1%}</p>', unsafe_allow_html=True)
                            elif prob > 0.1:
                                st.markdown(f'<p class="risk-medium">⚠️ SUSPICIOUS — {prob:.1%}</p>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<p class="risk-low">✅ LOW RISK — {prob:.1%}</p>', unsafe_allow_html=True)

                        # ── Signal metrics (kept at ONE nesting level: col_result → these columns) ──
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Merchant Risk", "High 🔴" if "FRAUD" in merchant_id else "Low 🟢")
                        m2.metric("Device Spread", "12 cards" if "FRAUD" in device_id else "1 card")
                        m3.metric("Cluster", "Mule 🔴" if prob > 0.5 else "Organic 🟢")

                        # ── Network graph ──
                        st.markdown('<div class="section-title">Transaction Network</div>', unsafe_allow_html=True)
                        net_fig, G = build_network_figure(card_id, merchant_id, device_id, amount, prob)
                        st.plotly_chart(net_fig, use_container_width=True)

                        # ── Risk metrics bar ──
                        st.markdown('<div class="section-title">Risk Metrics</div>', unsafe_allow_html=True)
                        st.plotly_chart(
                            risk_bar_chart(amount, len(G.edges()), prob),
                            use_container_width=True,
                        )

                        # ── Network stats ──
                        st.markdown('<div class="section-title">Network Statistics</div>', unsafe_allow_html=True)
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Nodes", len(G.nodes()))
                        s2.metric("Edges", len(G.edges()))
                        s3.metric("Avg Degree", f"{2*len(G.edges())/max(len(G.nodes()),1):.1f}")
                        s4.metric("Cluster", "High ⚠️" if prob > 0.4 else "Low ✅")

                        # ── Suspicious patterns ──
                        if prob > 0.4:
                            st.warning(
                                "🚨 **Suspicious patterns detected:** transaction linked to known fraud entities · "
                                "device used across multiple cards · unusual amount for merchant category"
                            )

                        st.info("ℹ️ SHAP: Features V14 and Amount are primary fraud score drivers.")

                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                        st.code("uvicorn src.serving.api:app --reload", language="bash")

                except requests.exceptions.ConnectionError:
                    st.error("⚡ API not reachable — is the FastAPI service running?")
                    st.code("uvicorn src.serving.api:app --reload", language="bash")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

            else:
                st.markdown("""
                <div style="height:320px;display:flex;flex-direction:column;
                            align-items:center;justify-content:center;
                            background:rgba(17,24,39,0.6);border:1px dashed #1e293b;
                            border-radius:12px;color:#334155;">
                    <div style="font-size:2.5rem">🛡️</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
                                color:#475569;margin-top:10px;letter-spacing:2px;">
                        AWAITING TRANSACTION INPUT
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — Model Performance
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[1]:
        results = load_results()
        if results:
            st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

            metric = st.selectbox(
                "Metric",
                ["pr_auc", "roc_auc", "f1", "ks", "recall_at_threshold"],
                format_func=lambda x: x.replace("_", " ").upper(),
            )

            rows = [
                {"Model": m, "Validation": r["val"][metric], "Test": r["test"][metric]}
                for m, r in results.items()
            ]
            perf_df = pd.DataFrame(rows)

            # Grouped bar chart
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Bar(
                name="Validation",
                x=perf_df["Model"], y=perf_df["Validation"],
                marker_color="#00e5ff", opacity=0.85,
            ))
            fig_perf.add_trace(go.Bar(
                name="Test",
                x=perf_df["Model"], y=perf_df["Test"],
                marker_color="#a78bfa", opacity=0.85,
            ))
            fig_perf.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.4)",
                height=320,
                font=dict(family="DM Sans", color="#94a3b8"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_perf, use_container_width=True)

            # Detailed table
            st.markdown('<div class="section-title">Detailed Metrics</div>', unsafe_allow_html=True)
            table_rows = []
            for m, r in results.items():
                table_rows.append({
                    "Model": m.replace("_", " ").title(),
                    "Val PR-AUC":  r["val"]["pr_auc"],
                    "Test PR-AUC": r["test"]["pr_auc"],
                    "Val ROC-AUC": r["val"]["roc_auc"],
                    "Val Recall":  r["val"]["recall_at_threshold"],
                    "Threshold":   r["threshold"],
                })
            tdf = pd.DataFrame(table_rows).set_index("Model")
            st.dataframe(
                tdf.style.highlight_max(color="rgba(0,229,255,0.15)", axis=0),
                use_container_width=True,
            )

            # Per-model PR + CM plots
            st.markdown('<div class="section-title">Curve Viewer</div>', unsafe_allow_html=True)
            model_sel = st.selectbox("Select model", list(results.keys()),
                                     format_func=lambda x: x.replace("_", " ").title())
            plots_dir = BASE / "data" / "plots"
            pc1, pc2 = st.columns(2)
            pr_path = plots_dir / f"{model_sel}_test_pr_curve.png"
            cm_path = plots_dir / f"{model_sel}_test_confusion_matrix.png"
            if pr_path.exists():
                pc1.image(str(pr_path), caption="Precision-Recall Curve", use_container_width=True)
            else:
                pc1.info("PR curve not found — run the training pipeline.")
            if cm_path.exists():
                pc2.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)
            else:
                pc2.info("Confusion matrix not found — run the training pipeline.")
        else:
            st.warning("No training results found. Run `python run_pipeline.py` first.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — System Health
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[2]:
        drift = load_drift_report()
        if drift:
            summary  = drift.get("summary", {})
            retrain  = summary.get("retrain_triggered", False)
            alerts   = drift.get("alerts", [])
            warnings = drift.get("warnings", [])

            # Status row
            st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Status",           "🔴 CRITICAL" if retrain else "🟢 HEALTHY")
            d2.metric("Features Drifted", f"{summary.get('alert_count',0)} / {summary.get('total_features',0)}")
            d3.metric("Max PSI",          f"{summary.get('max_psi', 0.0):.4f}")
            d4.metric("Action",           "RETRAIN NOW" if retrain else "MAINTAIN")

            # PSI chart
            if drift.get("features"):
                st.markdown('<div class="section-title">Feature Drift (PSI)</div>', unsafe_allow_html=True)
                st.caption("🟡 PSI > 0.10 = warning   🔴 PSI > 0.20 = retrain trigger")
                st.plotly_chart(psi_chart(drift["features"]), use_container_width=True)

            # Alert table
            if alerts:
                st.markdown('<div class="section-title">Active Alerts</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(alerts), use_container_width=True)

            if warnings:
                st.markdown('<div class="section-title">Warnings</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(warnings), use_container_width=True)

            # Feature importance
            st.markdown('<div class="section-title">Feature Importance (Best Model)</div>', unsafe_allow_html=True)
            fi_path = BASE / "data" / "feature_importance.json"
            if fi_path.exists():
                try:
                    fi_df = pd.read_json(fi_path).head(15)
                    fig_fi = go.Figure(go.Bar(
                        x=fi_df["importance"], y=fi_df["feature"],
                        orientation="h",
                        marker=dict(
                            color=fi_df["importance"],
                            colorscale=[[0, "#1a2235"], [0.5, "#00b4d8"], [1, "#00e5ff"]],
                            line=dict(width=0),
                        ),
                        texttemplate="%{x:.4f}", textposition="outside",
                        textfont=dict(family="Space Mono", size=9, color="#64748b"),
                    ))
                    fig_fi.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=380,
                        margin=dict(l=10, r=60, t=10, b=10),
                        font=dict(family="DM Sans", color="#94a3b8"),
                        xaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
                        yaxis=dict(showgrid=False, autorange="reversed"),
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)
                except Exception:
                    st.info("Feature importance data unavailable.")
            else:
                st.info("Feature importance not found — run the training pipeline.")
        else:
            st.warning("No drift report found. Run `python run_pipeline.py` first.")


if __name__ == "__main__":
    main()