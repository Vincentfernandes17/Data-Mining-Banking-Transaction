"""
dashboard.py — Phase 5: Visualization & Knowledge Presentation

Dashboard interaktif (Plotly Dash) untuk mengomunikasikan temuan kepada
audiens non-teknis. Menampilkan empat hal yang diminta proyek:
  - Cluster map      : peta segmen nasabah di ruang rasio perilaku
  - Rule network     : jaringan association rules (Apriori)
  - Outlier plot     : sebaran & klasifikasi anomali
  - Distributions    : profil & distribusi relevan

Sumber data (hasil Phase 1–4):
  - data/dataset_final.csv          (5000 nasabah + label cluster + label anomali)
  - outputs/phase3/rules_all.csv    (SEMUA association rules lolos filter;
                                     fallback: top_rules.csv untuk repo lama)

Cara menjalankan:
    python main.py --phase 5
    # atau
    python src/dashboard.py
    # lalu buka http://127.0.0.1:8050 di browser
"""

import os
import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from dash import Dash, dcc, html, dash_table, Input, Output

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
PHASE2_DIR = os.path.join(BASE_DIR, 'outputs', 'phase2')
PHASE3_DIR = os.path.join(BASE_DIR, 'outputs', 'phase3')

RATIOS = ['CC_Utilization', 'Transaction_to_Balance_Ratio', 'Loan_to_Balance_Ratio']
# Fitur demografis (urut dari paling membedakan segmen) untuk profil tambahan
DEMOGRAPHICS = ['Account Type', 'Age_Group', 'Loan Status', 'Gender',
                'Card Type', 'Loan Type']
RATIO_LABEL = {
    'CC_Utilization': 'Utilisasi Kartu (saldo/limit)',
    'Transaction_to_Balance_Ratio': 'Transaksi/Saldo',
    'Loan_to_Balance_Ratio': 'Pinjaman/Saldo',
}
SEGMENT_COLORS = {
    'Mainstream / Balanced': '#2ecc71',
    'Credit-Stressed / Over-Limit': '#e67e22',
    'Liquidity-Stressed / High-Leverage': '#e74c3c',
}
CLASS_COLORS = {
    'Normal': '#bdc3c7',
    'Rare but Valid': '#3498db',
    'Data Error / Quality': '#9b59b6',
    'Risk Signal': '#e74c3c',
}
METHOD_LABEL = {
    'KMeans_Segment': 'K-Means (segmen bernama)',
    'DBSCAN_Cluster': 'DBSCAN (-1 = noise/outlier)',
    'Hierarchical_Cluster': 'Hierarchical (Ward)',
}


# ════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════
def load_dashboard_data():
    """Muat dua sumber data dashboard: dataset_final.csv (5000 nasabah,
    hasil gabungan Phase 1-4) dan tabel association rules.

    Rules diambil dari rules_all.csv (SEMUA rule lolos filter — supaya KPI
    'Association Rules' konsisten dengan angka di laporan), dengan fallback
    ke top_rules.csv bila file lengkap belum ada. Kolom cluster non-KMeans
    di-cast ke string agar diperlakukan kategorikal oleh Plotly.
    """
    fpath = os.path.join(DATA_DIR, 'dataset_final.csv')
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"{fpath} tidak ada. Jalankan pipeline dulu: python main.py --phase all")
    df = pd.read_csv(fpath)
    for col in ['DBSCAN_Cluster', 'Hierarchical_Cluster']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    rules_path = os.path.join(PHASE3_DIR, 'rules_all.csv')
    if not os.path.exists(rules_path):                    # fallback repo lama
        rules_path = os.path.join(PHASE3_DIR, 'top_rules.csv')
    rules = pd.read_csv(rules_path) if os.path.exists(rules_path) else pd.DataFrame()
    return df, rules


# ════════════════════════════════════════════════════════════
# FIGURE BUILDERS (dipisah agar bisa di-smoke-test tanpa server)
# ════════════════════════════════════════════════════════════
def fig_cluster_map(df, method, x, y):
    """Peta segmen: scatter dua rasio perilaku (log-log) diwarnai label
    cluster dari metode terpilih — visual 'cluster map' utama dashboard."""
    color_map = SEGMENT_COLORS if method == 'KMeans_Segment' else None
    fig = px.scatter(
        df, x=x, y=y, color=method,
        color_discrete_map=color_map,
        opacity=0.55, log_x=True, log_y=True,
        labels={x: RATIO_LABEL.get(x, x), y: RATIO_LABEL.get(y, y)},
        hover_data=['Age', 'Account Balance', 'Credit Limit', 'Loan Amount'],
        title=f'Peta Segmen — {METHOD_LABEL.get(method, method)}',
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(legend_title_text='Segmen', height=480,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_segment_sizes(df, method):
    """Donut chart proporsi nasabah per segmen untuk metode terpilih."""
    counts = df[method].value_counts().reset_index()
    counts.columns = ['Segment', 'Jumlah']
    color_map = SEGMENT_COLORS if method == 'KMeans_Segment' else None
    fig = px.pie(counts, names='Segment', values='Jumlah', hole=0.45,
                 color='Segment', color_discrete_map=color_map,
                 title='Proporsi Nasabah per Segmen')
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_segment_profile(df, method):
    """Bar chart median 3 rasio perilaku per segmen (log-y) — profil
    kuantitatif yang membedakan tiap segmen."""
    prof = df.groupby(method)[RATIOS].median().reset_index()
    long = prof.melt(id_vars=method, var_name='Rasio', value_name='Median')
    long['Rasio'] = long['Rasio'].map(RATIO_LABEL)
    color_map = SEGMENT_COLORS if method == 'KMeans_Segment' else None
    fig = px.bar(long, x='Rasio', y='Median', color=method, barmode='group',
                 color_discrete_map=color_map, log_y=True,
                 title='Profil Rasio Perilaku per Segmen (median, skala log)')
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10),
                      legend_title_text='Segmen')
    return fig


def fig_segment_demographics(df, method, cat):
    """Komposisi demografis (kategorikal) tiap segmen — 100% stacked bar.
    Pada data sintetis ini distribusinya cenderung seragam antar segmen,
    yang justru menegaskan: segmen dibentuk PERILAKU, bukan demografi."""
    if cat not in df.columns:
        return go.Figure().update_layout(title=f'Kolom {cat} tidak tersedia')
    ct = (pd.crosstab(df[method], df[cat], normalize='index') * 100).reset_index()
    long = ct.melt(id_vars=method, var_name=cat, value_name='Persen')
    spread = (ct.drop(columns=[method]).max() - ct.drop(columns=[method]).min()).max()
    note = 'membedakan' if spread >= 10 else 'nyaris seragam'
    fig = px.bar(long, x=method, y='Persen', color=cat, barmode='stack',
                 title=f'Komposisi {cat} per Segmen (%) — spread {spread:.0f} pp ({note})',
                 labels={method: 'Segmen', 'Persen': '% dalam segmen'})
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10),
                      legend_title_text=cat, yaxis_range=[0, 100])
    return fig


def fig_feature_selection():
    """Validasi pemilihan fitur: silhouette per set fitur (dibaca dari CSV Phase 2).
    Pencarian otomatis memilih 3 rasio yang sama dengan pilihan domain."""
    path = os.path.join(PHASE2_DIR, 'feature_selection_comparison.csv')
    if not os.path.exists(path):
        return go.Figure().update_layout(
            title='feature_selection_comparison.csv belum dibuat (jalankan Phase 2)')
    comp = pd.read_csv(path)
    palette = ['#bdc3c7', '#bdc3c7', '#f39c12', '#2ecc71']
    colors = [palette[i] if i < len(palette) else '#2ecc71' for i in range(len(comp))]
    fig = go.Figure(go.Bar(
        x=comp['Feature Set'], y=comp['Silhouette'], marker_color=colors,
        text=[f'{v:.3f}' for v in comp['Silhouette']], textposition='outside'))
    fig.update_layout(
        title='Validasi Pemilihan Fitur — Silhouette per Set Fitur (K-Means, K=3)',
        yaxis_title='Silhouette Score', yaxis_range=[0, 0.65],
        height=400, margin=dict(l=10, r=10, t=50, b=90))
    return fig


def fig_rule_network(rules, min_lift):
    """Jaringan association rules: node = item, edge = relasi IF→THEN
    (difilter lift ≥ slider). Ukuran/warna node = seberapa sering item
    muncul di rule (degree). Layout spring dengan seed tetap."""
    if rules.empty:
        return go.Figure().update_layout(title='Tidak ada rules')
    sub = rules[rules['Lift'] >= min_lift]
    if sub.empty:
        return go.Figure().update_layout(title=f'Tidak ada rule dengan Lift ≥ {min_lift}')

    G = nx.DiGraph()
    for _, r in sub.iterrows():
        ants = [a.strip() for a in str(r['Antecedent (IF)']).split(',')]
        cons = [c.strip() for c in str(r['Consequent (THEN)']).split(',')]
        for a in ants:
            for c in cons:
                G.add_edge(a, c, lift=r['Lift'])

    pos = nx.spring_layout(G, k=0.9, seed=42)
    edge_x, edge_y = [], []
    for a, c in G.edges():
        edge_x += [pos[a][0], pos[c][0], None]
        edge_y += [pos[a][1], pos[c][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=1, color='#aaaaaa'),
                            hoverinfo='none')

    # ukuran node = derajat (seberapa sering item muncul di rule)
    deg = dict(G.degree())
    short = lambda s: s.split('=')[-1].strip()   # 'Loan Status=Closed' → 'Closed'
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode='markers+text',
        text=[short(n) for n in G.nodes()],
        textposition='top center', textfont=dict(size=10),
        marker=dict(size=[8 + 4 * deg[n] for n in G.nodes()],
                    color=[deg[n] for n in G.nodes()],
                    colorscale='YlOrRd', showscale=True,
                    colorbar=dict(title='Keterhubungan')),
        hovertext=[f'{n} (muncul di {deg[n]} relasi)' for n in G.nodes()],
        hoverinfo='text')
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    padx = (max(xs) - min(xs)) * 0.25 + 0.15   # ruang ekstra agar label tepi tak terpotong
    pady = (max(ys) - min(ys)) * 0.20 + 0.15
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title=f'Jaringan Association Rules (Lift ≥ {min_lift}) — {len(sub)} rule',
        showlegend=False, height=600,
        xaxis=dict(visible=False, range=[min(xs) - padx, max(xs) + padx]),
        yaxis=dict(visible=False, range=[min(ys) - pady, max(ys) + pady]),
        margin=dict(l=40, r=40, t=50, b=30))
    return fig


def fig_rule_scatter(rules, min_lift):
    """Scatter support vs confidence semua rule lolos slider; ukuran &
    warna = lift, hover = teks rule lengkap."""
    if rules.empty:
        return go.Figure().update_layout(title='Tidak ada rules')
    sub = rules[rules['Lift'] >= min_lift].copy()
    sub['Rule'] = sub['Antecedent (IF)'] + '  →  ' + sub['Consequent (THEN)']
    fig = px.scatter(sub, x='Support', y='Confidence', size='Lift', color='Lift',
                     color_continuous_scale='YlOrRd', hover_name='Rule',
                     title='Support vs Confidence (ukuran & warna = Lift)')
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_anomaly_breakdown(df):
    """Bar jumlah nasabah per kelas anomali (Normal / Rare / Data Error /
    Risk Signal) — ringkasan hasil klasifikasi Phase 4."""
    order = ['Normal', 'Rare but Valid', 'Data Error / Quality', 'Risk Signal']
    counts = df['classification'].value_counts().reindex(order).fillna(0).reset_index()
    counts.columns = ['Klasifikasi', 'Jumlah']
    fig = px.bar(counts, x='Klasifikasi', y='Jumlah', color='Klasifikasi',
                 color_discrete_map=CLASS_COLORS, text='Jumlah',
                 title='Klasifikasi Anomali (semua 5000 nasabah)')
    fig.update_layout(height=420, showlegend=False,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_anomaly_scatter(df, x, y):
    """Scatter rasio perilaku (log-log) diwarnai kelas anomali; sumbu bisa
    dipilih. Diurutkan agar titik anomali tergambar di atas titik normal."""
    d = df.sort_values('n_methods')  # normal di belakang, anomali di depan
    fig = px.scatter(
        d, x=x, y=y, color='classification',
        color_discrete_map=CLASS_COLORS, opacity=0.6,
        log_x=True, log_y=True,
        labels={x: RATIO_LABEL.get(x, x), y: RATIO_LABEL.get(y, y)},
        hover_data=['n_methods', 'KMeans_Segment'],
        title='Sebaran Anomali pada Ruang Rasio Perilaku')
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(height=480, legend_title_text='Klasifikasi',
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_anomaly_by_segment(df):
    """Stacked bar komposisi kelas anomali per segmen K-Means — visual
    cross-reference Phase 2 × Phase 4 (risiko menggumpal di segmen mana)."""
    ct = (df.groupby(['KMeans_Segment', 'classification']).size()
          .reset_index(name='Jumlah'))
    fig = px.bar(ct, x='KMeans_Segment', y='Jumlah', color='classification',
                 color_discrete_map=CLASS_COLORS, barmode='stack',
                 title='Anomali per Segmen (cross-reference Phase 2 × Phase 4)')
    fig.update_layout(height=440, legend_title_text='Klasifikasi',
                      xaxis_title='', margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ════════════════════════════════════════════════════════════
# KPI cards
# ════════════════════════════════════════════════════════════
def kpi_card(title, value, sub, color):
    """Kartu KPI kecil (judul, angka besar berwarna, subjudul) untuk baris
    ringkasan di atas dashboard."""
    return html.Div([
        html.Div(title, style={'fontSize': '13px', 'color': '#555'}),
        html.Div(value, style={'fontSize': '30px', 'fontWeight': '700',
                               'color': color}),
        html.Div(sub, style={'fontSize': '11px', 'color': '#888'}),
    ], style={'background': 'white', 'borderRadius': '10px', 'padding': '16px',
              'boxShadow': '0 1px 4px rgba(0,0,0,0.08)', 'flex': '1',
              'minWidth': '150px'})


# ════════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════════
def build_app():
    """Rakit aplikasi Dash lengkap: muat data, hitung KPI, susun layout
    3 tab (Segmentasi / Association Rules / Anomali), dan daftarkan
    callback interaktif. Dipisah dari run_dashboard agar bisa di-smoke-test
    tanpa menyalakan server."""
    df, rules = load_dashboard_data()
    app = Dash(__name__)
    app.title = 'Banking KDD Dashboard'

    n_total = len(df)
    n_seg = df['KMeans_Segment'].nunique()
    n_rules = len(rules)
    n_anom = int(df['is_consensus_anomaly'].sum())
    n_risk = int((df['classification'] == 'Risk Signal').sum())
    lift_min = float(rules['Lift'].min()) if not rules.empty else 1.0
    lift_max = float(rules['Lift'].max()) if not rules.empty else 2.0
    # floor ke 2 desimal: kalau max di-round KE ATAS (mis. 1.585→1.59), posisi
    # slider tertinggi menyaring SEMUA rule → grafik kosong. Floor mencegahnya.
    slider_min = int(lift_min * 100) / 100
    slider_max = int(lift_max * 100) / 100

    tab_style = {'padding': '8px', 'fontWeight': '600'}

    app.layout = html.Div([
        html.Div([
            html.H2('🏦 Banking Transaction — Knowledge Discovery Dashboard',
                    style={'margin': '0'}),
            html.Div('Group 6 · KDD 5-Fase · Segmentasi · Association Rules · '
                     'Anomaly Detection',
                     style={'color': '#666', 'fontSize': '13px'}),
        ], style={'padding': '14px 20px', 'background': '#1f3a5f',
                  'color': 'white'}),

        # KPI row
        html.Div([
            kpi_card('Total Nasabah', f'{n_total:,}', 'records dianalisis', '#1f3a5f'),
            kpi_card('Segmen', f'{n_seg}', 'profil bernama (K-Means)', '#2ecc71'),
            kpi_card('Association Rules', f'{n_rules}', 'non-trivial (lift ≥ 1.4)', '#e67e22'),
            kpi_card('Anomali Konsensus', f'{n_anom}', '≥2 metode setuju', '#9b59b6'),
            kpi_card('Risk Signal', f'{n_risk}', 'perlu eskalasi', '#e74c3c'),
        ], style={'display': 'flex', 'gap': '12px', 'padding': '16px 20px',
                  'flexWrap': 'wrap', 'background': '#eef1f5'}),

        dcc.Tabs([
            # ── TAB 1: SEGMEN ──
            dcc.Tab(label='1 · Segmentasi Nasabah', style=tab_style, children=[
                html.Div([
                    html.Div([
                        html.Label('Metode clustering'),
                        dcc.Dropdown(
                            id='seg-method',
                            options=[{'label': v, 'value': k}
                                     for k, v in METHOD_LABEL.items()],
                            value='KMeans_Segment', clearable=False),
                    ], style={'flex': '1'}),
                    html.Div([
                        html.Label('Sumbu X'),
                        dcc.Dropdown(id='seg-x',
                                     options=[{'label': RATIO_LABEL[r], 'value': r}
                                              for r in RATIOS],
                                     value=RATIOS[1], clearable=False),
                    ], style={'flex': '1'}),
                    html.Div([
                        html.Label('Sumbu Y'),
                        dcc.Dropdown(id='seg-y',
                                     options=[{'label': RATIO_LABEL[r], 'value': r}
                                              for r in RATIOS],
                                     value=RATIOS[2], clearable=False),
                    ], style={'flex': '1'}),
                ], style={'display': 'flex', 'gap': '12px', 'padding': '14px 20px'}),
                html.Div([
                    dcc.Graph(id='cluster-map', style={'flex': '3'}),
                    dcc.Graph(id='segment-pie', style={'flex': '2'}),
                ], style={'display': 'flex', 'gap': '8px', 'padding': '0 16px'}),
                dcc.Graph(id='segment-profile', style={'padding': '0 16px'}),

                # Profil demografis tambahan (kategorikal) per segmen
                html.Div([
                    html.Label('Fitur demografis untuk profil segmen'),
                    dcc.Dropdown(
                        id='seg-demo',
                        options=[{'label': c, 'value': c}
                                 for c in DEMOGRAPHICS if c in df.columns],
                        value=next((c for c in DEMOGRAPHICS if c in df.columns), None),
                        clearable=False),
                ], style={'padding': '8px 20px'}),
                dcc.Graph(id='segment-demographics', style={'padding': '0 16px'}),

                # Validasi pemilihan fitur (statis) — kenapa 3 rasio ini
                html.Div('Kenapa 3 rasio ini dipakai? Pencarian otomatis atas SEMUA '
                         'kombinasi 3-fitur memilih ketiga rasio yang sama (peringkat #1), '
                         'jauh mengungguli fitur mentah & PCA — pilihan domain tervalidasi '
                         'secara data-driven.',
                         style={'padding': '12px 24px 0', 'color': '#555',
                                'fontSize': '13px'}),
                dcc.Graph(id='feature-selection', figure=fig_feature_selection(),
                          style={'padding': '0 16px 18px'}),
            ]),

            # ── TAB 2: ASSOCIATION RULES ──
            dcc.Tab(label='2 · Association Rules', style=tab_style, children=[
                html.Div([
                    html.Label(f'Minimum Lift  (rentang {slider_min:.2f}–{slider_max:.2f})'),
                    dcc.Slider(id='lift-slider', min=slider_min,
                               max=slider_max, step=0.01,
                               value=slider_min,
                               marks={slider_min: f'{slider_min:.2f}',
                                      slider_max: f'{slider_max:.2f}'},
                               tooltip={'placement': 'bottom', 'always_visible': True}),
                ], style={'padding': '14px 24px'}),
                dcc.Graph(id='rule-network', style={'padding': '0 16px'}),
                dcc.Graph(id='rule-scatter', style={'padding': '0 16px'}),
                html.Div(id='rule-table', style={'padding': '12px 20px'}),
            ]),

            # ── TAB 3: ANOMALI ──
            dcc.Tab(label='3 · Anomaly Detection', style=tab_style, children=[
                html.Div([
                    dcc.Graph(id='anom-breakdown', style={'flex': '1'},
                              figure=fig_anomaly_breakdown(df)),
                    dcc.Graph(id='anom-by-seg', style={'flex': '1'},
                              figure=fig_anomaly_by_segment(df)),
                ], style={'display': 'flex', 'gap': '8px', 'padding': '12px 16px'}),
                html.Div([
                    html.Label('Sumbu X / Y anomaly scatter'),
                    html.Div([
                        dcc.Dropdown(id='anom-x',
                                     options=[{'label': RATIO_LABEL[r], 'value': r}
                                              for r in RATIOS],
                                     value=RATIOS[1], clearable=False,
                                     style={'flex': '1'}),
                        dcc.Dropdown(id='anom-y',
                                     options=[{'label': RATIO_LABEL[r], 'value': r}
                                              for r in RATIOS],
                                     value=RATIOS[2], clearable=False,
                                     style={'flex': '1'}),
                    ], style={'display': 'flex', 'gap': '12px'}),
                ], style={'padding': '8px 20px'}),
                dcc.Graph(id='anom-scatter', style={'padding': '0 16px'}),
            ]),
        ]),
        html.Div('Dibuat dengan Python Dash · data: Phase 1–4 pipeline',
                 style={'textAlign': 'center', 'color': '#999',
                        'fontSize': '11px', 'padding': '12px'}),
    ], style={'fontFamily': 'Segoe UI, sans-serif', 'background': '#f7f9fb'})

    # ── Callbacks ──
    @app.callback(
        Output('cluster-map', 'figure'), Output('segment-pie', 'figure'),
        Output('segment-profile', 'figure'), Output('segment-demographics', 'figure'),
        Input('seg-method', 'value'), Input('seg-x', 'value'), Input('seg-y', 'value'),
        Input('seg-demo', 'value'))
    def _seg(method, x, y, demo):
        return (fig_cluster_map(df, method, x, y),
                fig_segment_sizes(df, method),
                fig_segment_profile(df, method),
                fig_segment_demographics(df, method, demo))

    @app.callback(
        Output('rule-network', 'figure'), Output('rule-scatter', 'figure'),
        Output('rule-table', 'children'),
        Input('lift-slider', 'value'))
    def _rules(min_lift):
        sub = rules[rules['Lift'] >= min_lift] if not rules.empty else rules
        table = dash_table.DataTable(
            data=sub.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in sub.columns],
            style_cell={'fontSize': '12px', 'textAlign': 'left',
                        'whiteSpace': 'normal', 'height': 'auto'},
            style_header={'fontWeight': 'bold', 'background': '#eef1f5'},
            page_size=10, sort_action='native')
        return (fig_rule_network(rules, min_lift),
                fig_rule_scatter(rules, min_lift), table)

    @app.callback(
        Output('anom-scatter', 'figure'),
        Input('anom-x', 'value'), Input('anom-y', 'value'))
    def _anom(x, y):
        return fig_anomaly_scatter(df, x, y)

    return app


def run_dashboard(host='127.0.0.1', port=8050, debug=False):
    """Bangun aplikasi lalu jalankan server Dash (blocking) di host:port."""
    app = build_app()
    print(f"\n🚀 Dashboard berjalan di http://{host}:{port}  (Ctrl+C untuk stop)")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_dashboard()
