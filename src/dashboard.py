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
        render_mode='webgl',   # WebGL: render 5000 titik jauh lebih cepat dari SVG
        labels={x: RATIO_LABEL.get(x, x), y: RATIO_LABEL.get(y, y)},
        hover_data=['Age', 'Account Balance', 'Credit Limit', 'Loan Amount'],
        title=f'Peta Segmen — {METHOD_LABEL.get(method, method)}',
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(legend_title_text='Segmen', height=480,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_segment_sizes(df, method):
    """Donut chart proporsi nasabah per segmen untuk metode terpilih.

    Sengaja memakai go.Pie dengan list Python biasa (bukan DataFrame/np.array)
    agar nilai TIDAK diserialisasi sebagai typed-array biner (bdata). Serialisasi
    biner membuat Plotly.js kadang gagal mendeteksi perubahan saat jumlah irisan
    tetap sama sehingga donut tidak ikut ter-repaint ketika metode diganti. Nama
    metode juga dimasukkan ke judul supaya layout selalu berubah (pasti repaint)
    dan pengguna tahu donut ini mengikuti pilihan METODE, bukan sumbu X/Y."""
    counts = df[method].value_counts()
    labels = [str(x) for x in counts.index.tolist()]
    values = [int(v) for v in counts.values.tolist()]
    colors = ([SEGMENT_COLORS.get(l) for l in labels]
              if method == 'KMeans_Segment' else None)
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.45, sort=False,
                           marker=dict(colors=colors) if colors else None))
    fig.update_layout(
        title=f'Proporsi Nasabah per Segmen — {METHOD_LABEL.get(method, method)}',
        height=480, margin=dict(l=10, r=10, t=50, b=10))
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
                 title=f'Profil Rasio Perilaku per Segmen (median, skala log) — '
                       f'{METHOD_LABEL.get(method, method)}')
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


# ── Phase 1: Preprocessing (figur dihitung ulang dari data, bukan angka hafalan)
# Fitur mentah yang tersedia di dataset_final untuk perbandingan struktur.
# Loan Term & Minimum Payment Due ikut dievaluasi di preprocess.py tetapi tidak
# terbawa ke dataset_final (dibuang saat feature selection), jadi tidak muncul.
RAW_NUMERIC = ['Age', 'Account Balance', 'Transaction Amount', 'Loan Amount',
               'Credit Limit', 'Credit Card Balance', 'Rewards Points',
               'Interest Rate', 'Account_Age_Years',
               'Days_Since_Last_Transaction']
# Ambang keputusan (identik dengan detect_outliers_prescaling di preprocess.py)
SKEW_THRESHOLD = 1.0     # |skew| > 1 → distribusi miring
OUTLIER_THRESHOLD = 5.0  # outlier IQR > 5% → butuh scaler kebal outlier


def _preprocessing_stats(df):
    """Hitung |skew| dan persentase outlier IQR tiap fitur SATU KALI, lalu
    tentukan scaler yang dipilih dengan aturan yang sama seperti Phase 1.

    Dipakai bersama oleh kedua figur preprocessing supaya urutan barisnya
    identik kiri dan kanan, sehingga penonton bisa membaca keduanya sebagai
    satu tabel: fitur yang miring di kiri adalah fitur yang ber-outlier di
    kanan. Diurutkan menaik agar |skew| terbesar tampil di baris teratas."""
    rows = []
    for col in RAW_NUMERIC + RATIOS:
        if col not in df.columns:
            continue
        s = df[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        pct = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100)
        skew = abs(float(s.skew()))
        rows.append({
            'Fitur': RATIO_LABEL.get(col, col),
            'Skew': skew,
            'Persen': pct,
            'Kelompok': 'Rasio hasil rekayasa' if col in RATIOS else 'Fitur mentah',
            'Scaler': ('RobustScaler'
                       if (pct > OUTLIER_THRESHOLD or skew > SKEW_THRESHOLD)
                       else 'MinMaxScaler'),
        })
    return pd.DataFrame(rows).sort_values('Skew')


def _bar_room(fig, values, pad=0.22):
    """Beri ruang kosong di kanan sumbu-x supaya label angka di ujung batang
    tidak terpotong tepi grafik (masalah klasik textposition='outside')."""
    top = max(values) if len(values) else 1.0
    fig.update_xaxes(range=[0, top * (1 + pad)])
    return fig


def fig_raw_vs_ratio_skew(stats):
    """Inti keputusan Phase 1: kenapa clustering TIDAK memakai fitur mentah.

    Fitur mentah nyaris uniform (|skew| ~ 0 = tidak ada gumpalan, tidak ada
    cluster alami), sedangkan rasio antar dua fitur uniform melahirkan
    distribusi berekor panjang yang bisa disegmentasi. Garis putus-putus di
    |skew| = 1 adalah ambang yang dipakai Phase 1 untuk menyebut sebuah
    distribusi miring."""
    fig = px.bar(stats, x='Skew', y='Fitur', orientation='h', color='Kelompok',
                 color_discrete_map={'Fitur mentah': '#bdc3c7',
                                     'Rasio hasil rekayasa': '#2ecc71'},
                 text=[f'{v:.2f}' for v in stats['Skew']],
                 title='Fitur mentah nyaris uniform, rasio justru berstruktur')
    fig.update_traces(textposition='outside', cliponaxis=False,
                      textfont_size=11)
    fig.add_vline(x=SKEW_THRESHOLD, line_dash='dot', line_color='#7f8c8d',
                  annotation_text='ambang miring |skew| = 1',
                  annotation_position='bottom right', annotation_font_size=10)
    fig.update_layout(
        height=470, xaxis_title='|Skewness|   (0 = rata, tidak ada struktur)',
        yaxis_title='', legend_title_text='',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='left', x=0),
        margin=dict(l=10, r=20, t=90, b=40))
    return _bar_room(fig, stats['Skew'])


def fig_prescaling_scaler_choice(stats):
    """Deteksi outlier pra-scaling (IQR) yang menentukan pilihan scaler.

    Outlier sengaja TIDAK dihapus karena bisa jadi sinyal nyata yang
    diselidiki di Phase 4; persentasenya hanya dipakai memilih scaler per
    fitur — RobustScaler (median/IQR, kebal outlier) bila outlier >5% atau
    |skew|>1, selain itu MinMaxScaler. Urutan barisnya sama dengan figur
    skewness di sebelahnya agar keduanya bisa dibaca berdampingan."""
    fig = px.bar(stats, x='Persen', y='Fitur', orientation='h', color='Scaler',
                 color_discrete_map={'RobustScaler': '#e74c3c',
                                     'MinMaxScaler': '#2ecc71'},
                 text=[f'{v:.1f}%' for v in stats['Persen']],
                 title='Outlier IQR menentukan scaler tiap fitur')
    fig.update_traces(textposition='outside', cliponaxis=False,
                      textfont_size=11)
    fig.add_vline(x=OUTLIER_THRESHOLD, line_dash='dot', line_color='#7f8c8d',
                  annotation_text='ambang 5%',
                  annotation_position='bottom right', annotation_font_size=10)
    fig.update_layout(
        height=470, xaxis_title='% baris di luar pagar IQR',
        yaxis_title='', legend_title_text='',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='left', x=0),
        margin=dict(l=10, r=20, t=90, b=40))
    return _bar_room(fig, stats['Persen'])


# Ambang binning domain-tetap (dipakai Phase 3 / Apriori) + alasannya.
BINNING_ROWS = [
    ('Age_Group', 'Young Adult · Early Career · Established · Pre-Retirement · Senior',
     'Tahap hidup finansial (Life-Cycle Hypothesis), bukan kuantil'),
    ('Balance_Bucket', 'Below-Minimum (<$1.500) · Mass-Market · Comfortable (≥$5.000)',
     'Ambang saldo minimum perbankan ritel'),
    ('Transaction_Size', 'Everyday (<$1.000) · Large · Very Large (>$3.000)',
     'Tier pemantauan nominal transaksi'),
    ('Loan_Size', 'Small (<$5rb) · Medium · Large · Very Large (>$35rb)',
     'Bracket pinjaman konsumen, cap tanpa agunan ~$35rb'),
    ('Rate_Category', 'Low (≤4%) · Moderate · High (>7%)',
     'Tier prime / elevated / higher-priced'),
    ('CC_Utilization_Category', 'Low (<30%) · Moderate · High · Over-Limit (>100%)',
     'Pedoman credit-scoring FICO 30/70/100%'),
]


def _short(s):
    """'Loan Status=Closed' → 'Closed' untuk label node yang ringkas."""
    return str(s).split('=')[-1].strip()


def build_rule_graph(rules):
    """Bangun SATU DiGraph berarah dari seluruh rules: node = item, edge
    berarah dari tiap antecedent (IF) ke tiap consequent (THEN). Bila sebuah
    pasangan item muncul di banyak rule, edge menyimpan lift TERTINGGI dan
    jumlah kemunculannya. Layout dihitung SEKALI di sini (spring, seed tetap)
    lalu dipakai ulang saat slider digeser, sehingga interaksi tetap cepat
    dan posisi node stabil. Mengembalikan (G, pos)."""
    G = nx.DiGraph()
    if rules.empty:
        return G, {}
    for _, r in rules.iterrows():
        ants = [a.strip() for a in str(r['Antecedent (IF)']).split(',')]
        cons = [c.strip() for c in str(r['Consequent (THEN)']).split(',')]
        lift = float(r['Lift'])
        for a in ants:
            for c in cons:
                if a == c:
                    continue
                if G.has_edge(a, c):
                    G[a][c]['lift'] = max(G[a][c]['lift'], lift)
                    G[a][c]['n'] += 1
                else:
                    G.add_edge(a, c, lift=lift, n=1)
    pos = nx.spring_layout(G, k=0.9, seed=42) if G.number_of_nodes() else {}
    return G, pos


def fig_rule_network(G, pos, min_lift, max_arrows=140):
    """Jaringan association rules BERARAH. Panah menunjuk dari kondisi (IF)
    ke hasil (THEN), difilter lift ≥ slider. Panah digambar sebagai anotasi
    dengan arrowhead sehingga arah relasi jelas terbaca. Ukuran & warna node
    = derajat (seberapa sering item terlibat). Layout `pos` dihitung sekali
    di build_rule_graph agar callback slider ringan (<100ms)."""
    if G.number_of_nodes() == 0:
        return go.Figure().update_layout(title='Tidak ada rules')

    # Edge yang lolos filter lift; simpul yang dipakai hanya yang tersentuh edge.
    kept = [(a, c, d) for a, c, d in G.edges(data=True) if d['lift'] >= min_lift]
    if not kept:
        return go.Figure().update_layout(
            title=f'Tidak ada relasi dengan Lift ≥ {min_lift:.2f}')
    nodes = sorted({n for a, c, _ in kept for n in (a, c)})
    subdeg = {n: 0 for n in nodes}
    for a, c, _ in kept:
        subdeg[a] += 1
        subdeg[c] += 1

    # Panah: satu anotasi berarah per edge (dibatasi max_arrows demi kecepatan
    # & keterbacaan; edge lift tertinggi diprioritaskan).
    kept_sorted = sorted(kept, key=lambda t: t[2]['lift'], reverse=True)
    shown = kept_sorted[:max_arrows]
    lifts = [d['lift'] for _, _, d in shown]
    lo, hi = (min(lifts), max(lifts)) if lifts else (1.0, 1.0)
    def _grey(lift):
        t = 0.0 if hi == lo else (lift - lo) / (hi - lo)   # lift tinggi = lebih gelap
        v = int(170 - 120 * t)
        return f'rgb({v},{v},{v})'
    annotations = []
    for a, c, d in shown:
        ax_, ay_ = pos[a]
        cx_, cy_ = pos[c]
        annotations.append(dict(
            x=cx_, y=cy_, ax=ax_, ay=ay_,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=3, arrowsize=1.3,
            arrowwidth=1.1 + 1.4 * (0 if hi == lo else (d['lift'] - lo) / (hi - lo)),
            arrowcolor=_grey(d['lift']), opacity=0.85, standoff=6, startstandoff=6))

    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode='markers+text',
        text=[_short(n) for n in nodes],
        textposition='top center', textfont=dict(size=10),
        marker=dict(size=[9 + 4 * subdeg[n] for n in nodes],
                    color=[subdeg[n] for n in nodes],
                    colorscale='YlOrRd', showscale=True,
                    colorbar=dict(title='Keterhubungan'),
                    line=dict(width=1, color='#555')),
        hovertext=[f'{n} (terlibat di {subdeg[n]} relasi)' for n in nodes],
        hoverinfo='text')

    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    padx = (max(xs) - min(xs)) * 0.25 + 0.15
    pady = (max(ys) - min(ys)) * 0.20 + 0.15
    capped = '' if len(kept) <= max_arrows else f' (panah dibatasi {max_arrows} lift tertinggi)'
    fig = go.Figure([node_trace])
    fig.update_layout(
        title=f'Jaringan Association Rules BERARAH (Lift ≥ {min_lift:.2f}) — '
              f'{len(kept)} relasi{capped}. Panah IF → THEN.',
        showlegend=False, height=600, annotations=annotations,
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
                     render_mode='webgl',
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
        log_x=True, log_y=True, render_mode='webgl',
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
    ringkasan di atas dashboard. Ukuran & pembungkusannya diatur class
    .kpi-card di assets/dashboard.css agar bisa jadi dua kolom di layar kecil."""
    return html.Div([
        html.Div(title, style={'fontSize': '13px', 'color': '#555'}),
        html.Div(value, className='kpi-value',
                 style={'fontSize': '30px', 'fontWeight': '700', 'color': color}),
        html.Div(sub, style={'fontSize': '11px', 'color': '#888'}),
    ], className='kpi-card')


# ════════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════════
# Kredit: sumber data & tim penyusun (ditampilkan di section terakhir)
REPO_URL = 'https://github.com/Vincentfernandes17/Data-Mining-Banking-Transaction'
DATASET_URL = 'https://github.com/ahsan084/Banking-Dataset'
TEAM = [
    ('Stephen Christopher', '2802540080', 'stephen.christopher@binus.ac.id',
     'Insight Communicator'),
    ('Vincent Fernandes', '2802538372', 'vincent.fernandes@binus.ac.id',
     'Data Engineer'),
    ('Crisvito', '2802540093', 'crisvito@binus.ac.id', 'Pattern Analyst'),
    ('Hillary Violen Christalia Sanjaya', '2502538656',
     'hillary.sanjaya@binus.ac.id', 'Segmentation Specialist'),
    ('Aryaka Syahrezki', '2802540244', 'aryaka.syahrezki@binus.ac.id',
     'Data Engineer'),
]

# Tautan tampil sebagai teks singkat (URL panjang disembunyikan di balik href)
_LINK = {'color': '#1f3a5f', 'fontWeight': '600', 'textDecoration': 'underline'}

_TH = {'textAlign': 'left', 'padding': '6px 8px', 'background': '#eef1f5',
       'borderBottom': '2px solid #dde3ea'}
_TD = {'padding': '6px 8px', 'borderBottom': '1px solid #eef1f5',
       'verticalAlign': 'top'}
_TD_B = dict(_TD, fontWeight='600')

# Section dashboard: (anchor id, label navbar). Dipakai dua kali — untuk
# membangun tautan navbar dan untuk memberi id pada tiap section.
SECTIONS = [
    ('sec-data', 'Data'),
    ('sec-segmen', 'Segmentasi'),
    ('sec-rules', 'Rules'),
    ('sec-anomali', 'Anomali'),
    ('sec-kredit', 'Kredit'),
]


def navbar():
    """Navbar tipis yang menempel di atas layar berisi tautan lompat ke tiap
    section. Di layar sempit daftar tautannya digeser horizontal (diatur di
    assets/dashboard.css) sehingga tetap lengkap tanpa memakan tinggi."""
    return html.Div([
        html.Span('🏦 Banking KDD', className='navbar-brand'),
        html.Nav([html.A(label, href=f'#{sid}') for sid, label in SECTIONS],
                 className='navbar-links'),
    ], className='navbar')


def pp_chip(value, label):
    """Chip ringkas untuk angka kunci Phase 1 — supaya bagian preprocessing
    bisa dipindai sekilas tanpa harus membaca paragraf lengkapnya."""
    return html.Div([
        html.Div(value, style={'fontSize': '19px', 'fontWeight': '700',
                               'color': '#1f3a5f', 'lineHeight': '1.2'}),
        html.Div(label, style={'fontSize': '11.5px', 'color': '#666'}),
    ], className='chip')


def _section_header(num, title, subtitle):
    """Header satu section pada dashboard single-page (nomor, judul, subjudul)."""
    return html.Div([
        html.H3(f'{num} · {title}', style={'margin': '0', 'color': '#1f3a5f'}),
        html.Div(subtitle, style={'color': '#666', 'fontSize': '13px',
                                  'marginTop': '2px'}),
    ], className='section-header')


def build_app():
    """Rakit aplikasi Dash SINGLE-PAGE (tanpa tab, semua section tersusun
    vertikal supaya tim bisa memindai seluruh temuan cepat tanpa klik).
    Muat data, hitung KPI, precompute graph rule berarah + layout SEKALI
    (agar slider ringan), susun layout, daftarkan callback. Dipisah dari
    run_dashboard agar bisa di-smoke-test tanpa menyalakan server."""
    df, rules = load_dashboard_data()
    # assets_folder eksplisit: CSS tata letak & media query ada di
    # assets/dashboard.css pada root project, bukan di sebelah src/.
    app = Dash(__name__, assets_folder=os.path.join(BASE_DIR, 'assets'),
               meta_tags=[{'name': 'viewport',
                           'content': 'width=device-width, initial-scale=1'}])
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
    # Nilai awal slider sedikit di atas minimum agar tampilan awal jaringan tidak
    # menjadi "bola benang"; pengguna bisa menurunkannya untuk melihat semua.
    default_lift = round(float(rules['Lift'].median()), 2) if not rules.empty else slider_min
    default_lift = min(max(default_lift, slider_min), slider_max)

    # Precompute graph rule BERARAH + layout SEKALI (bukan tiap callback) → cepat.
    rule_graph, rule_pos = build_rule_graph(rules)

    # Figur statis (tidak bergantung input) dibangun sekali.
    fig_feat = fig_feature_selection()
    # Statistik preprocessing dihitung SEKALI lalu dipakai dua figur sekaligus.
    pp_stats = _preprocessing_stats(df)
    fig_skew = fig_raw_vs_ratio_skew(pp_stats)
    fig_scaler = fig_prescaling_scaler_choice(pp_stats)
    n_robust = int((pp_stats['Scaler'] == 'RobustScaler').sum())
    fig_anom_break = fig_anomaly_breakdown(df)
    fig_anom_seg = fig_anomaly_by_segment(df)

    app.layout = html.Div([
        navbar(),

        # KPI row
        html.Div([
            kpi_card('Total Nasabah', f'{n_total:,}', 'records dianalisis', '#1f3a5f'),
            kpi_card('Segmen', f'{n_seg}', 'profil bernama (K-Means)', '#2ecc71'),
            kpi_card('Association Rules', f'{n_rules}', 'non-trivial (lift ≥ 1.4)', '#e67e22'),
            kpi_card('Anomali Konsensus', f'{n_anom}', '≥2 metode setuju', '#9b59b6'),
            kpi_card('Risk Signal', f'{n_risk}', 'perlu eskalasi', '#e74c3c'),
        ], className='kpi-row'),

        # ── SECTION 1: DATA & PREPROCESSING ──
        html.Div(id='sec-data', className='section', children=[
            _section_header('1', 'Data & Preprocessing',
                            'Seperti apa datasetnya dan apa saja yang '
                            'dikerjakan sebelum data layak ditambang'),
            html.Div([
                pp_chip('5.000 × 40', 'baris dan kolom mentah'),
                pp_chip('0', 'nilai kosong dan duplikat'),
                pp_chip('18', 'kolom dilepas (PII, kunci, tanggal)'),
                pp_chip('5', 'fitur baru (2 temporal, 3 rasio)'),
                pp_chip('6', 'fitur didiskretisasi untuk Apriori'),
                pp_chip(f'{n_robust}', 'fitur perlu RobustScaler'),
            ], className='chip-row'),
            html.Div([
                html.P([html.Strong('Datanya sudah bersih, jadi pekerjaan '
                                    'beratnya ada di tempat lain. '),
                        'Tidak ada nilai kosong dan tidak ada duplikat, '
                        'sehingga fokus Phase 1 bergeser ke validasi aturan '
                        'bisnis dan rekayasa fitur.']),
                html.P([html.Strong('Tiga keputusan validasi, semuanya '
                                    'mempertahankan data. '),
                        'Ada 1.703 baris yang saldo akhirnya tidak cocok '
                        'dengan jenis transaksi, tetapi proporsi anomalinya '
                        'sama saja dengan baris yang konsisten (5,7 persen '
                        'lawan 6,2 persen), jadi bukan penanda apa pun dan '
                        'tidak dibuang. Ada 855 nasabah dengan saldo kartu '
                        'melewati limit, dipertahankan karena justru menjadi '
                        'sinyal tekanan kredit di Phase 4. Umur 18 sampai 69 '
                        'tahun dan bunga 1 sampai 10 persen, seluruhnya wajar.']),
                html.P([html.Strong('Kenapa clustering memakai rasio, bukan '
                                    'angka aslinya? '),
                        'Grafik kiri menjawabnya. Seluruh fitur mentah punya '
                        'skewness di bawah 0,05, artinya distribusinya rata '
                        'dan tidak ada gumpalan yang bisa dijadikan cluster. '
                        'Begitu dua fitur rata itu dibagi, misalnya saldo '
                        'kartu dibagi limit, hasilnya berekor panjang dan di '
                        'situlah segmen nasabah muncul. Grafik kanan memakai '
                        'urutan baris yang sama: fitur yang miring di kiri '
                        'persis fitur yang ber-outlier di kanan, dan itulah '
                        'yang menentukan scaler mana yang dipakai. Outlier '
                        'sendiri tidak dibuang karena bisa jadi temuan nyata '
                        'yang diselidiki di Phase 4.']),
            ], className='prose'),
            html.Div([
                dcc.Graph(id='pp-skew', figure=fig_skew,
                          className='chart-half', responsive=True),
                dcc.Graph(id='pp-scaler', figure=fig_scaler,
                          className='chart-half', responsive=True),
            ], className='chart-row'),
            html.Div('Kedua grafik memakai urutan fitur yang sama, jadi bisa '
                     'dibaca berdampingan baris per baris. Sepuluh fitur mentah '
                     'di bagian bawah tidak punya satu pun outlier IQR, batang '
                     'nol persennya memang tidak terlihat, dan semuanya cukup '
                     'memakai MinMaxScaler.', className='note'),
            html.Div([
                html.P([html.Strong('Diskretisasi untuk Association Rules. '),
                        'Enam fitur kontinu diubah menjadi kategori memakai '
                        'ambang domain tetap, bukan kuantil, supaya kategorinya '
                        'punya arti yang sama walau datanya berganti.'],
                       style={'marginBottom': '6px'}),
                html.Table(
                    [html.Tr([html.Th('Kolom kategori', style=_TH),
                              html.Th('Kategori', style=_TH),
                              html.Th('Dasar ambang', style=_TH)])] +
                    [html.Tr([html.Td(k, style=_TD_B), html.Td(v, style=_TD),
                              html.Td(w, style=_TD)])
                     for k, v, w in BINNING_ROWS],
                    style={'fontSize': '12.5px'}),
            ], className='table-wrap',
               style={'color': '#333', 'fontSize': '14px', 'lineHeight': '1.6'}),
        ]),

        # ── SECTION 2: SEGMENTASI ──
        html.Div(id='sec-segmen', className='section', children=[
            _section_header('2', 'Segmentasi Nasabah',
                            'Peta segmen, proporsi, profil rasio, dan komposisi '
                            'demografis per segmen'),
            html.Div([
                html.Div([
                    html.Label('Metode clustering'),
                    dcc.Dropdown(
                        id='seg-method',
                        options=[{'label': v, 'value': k}
                                 for k, v in METHOD_LABEL.items()],
                        value='KMeans_Segment', clearable=False),
                ]),
                html.Div([
                    html.Label('Sumbu X'),
                    dcc.Dropdown(id='seg-x',
                                 options=[{'label': RATIO_LABEL[r], 'value': r}
                                          for r in RATIOS],
                                 value=RATIOS[1], clearable=False),
                ]),
                html.Div([
                    html.Label('Sumbu Y'),
                    dcc.Dropdown(id='seg-y',
                                 options=[{'label': RATIO_LABEL[r], 'value': r}
                                          for r in RATIOS],
                                 value=RATIOS[2], clearable=False),
                ]),
            ], className='control-row'),
            html.Div('Sumbu X dan Y hanya mengubah peta segmen di kiri. Donut '
                     'proporsi, profil rasio, dan komposisi demografis '
                     'mengikuti pilihan METODE, karena keanggotaan segmen tiap '
                     'nasabah tidak berubah hanya karena sudut pandangnya '
                     'digeser.', className='note'),
            html.Div([
                dcc.Graph(id='cluster-map', className='chart-3', responsive=True),
                dcc.Graph(id='segment-pie', className='chart-2', responsive=True),
            ], className='chart-row'),
            dcc.Graph(id='segment-profile', className='chart-full',
                      responsive=True),
            html.Div([
                html.Label('Fitur demografis untuk profil segmen'),
                dcc.Dropdown(
                    id='seg-demo',
                    options=[{'label': c, 'value': c}
                             for c in DEMOGRAPHICS if c in df.columns],
                    value=next((c for c in DEMOGRAPHICS if c in df.columns), None),
                    clearable=False),
            ], className='control-row'),
            dcc.Graph(id='segment-demographics', className='chart-full',
                      responsive=True),
            html.Div('Kenapa 3 rasio ini dipakai? Pencarian otomatis atas SEMUA '
                     'kombinasi 3-fitur memilih ketiga rasio yang sama (peringkat #1), '
                     'jauh mengungguli fitur mentah & PCA — pilihan domain tervalidasi '
                     'secara data-driven.', className='note'),
            dcc.Graph(id='feature-selection', figure=fig_feat,
                      className='chart-full', responsive=True),
        ]),

        # ── SECTION 3: ASSOCIATION RULES ──
        html.Div(id='sec-rules', className='section', children=[
            _section_header('3', 'Association Rules',
                            'Jaringan relasi BERARAH (panah IF → THEN), sebaran '
                            'support–confidence, dan tabel rule'),
            html.Div([
                html.Label(f'Minimum Lift  (rentang {slider_min:.2f}–{slider_max:.2f})'),
                dcc.Slider(id='lift-slider', min=slider_min,
                           max=slider_max, step=0.01,
                           value=default_lift,
                           marks={slider_min: f'{slider_min:.2f}',
                                  slider_max: f'{slider_max:.2f}'},
                           tooltip={'placement': 'bottom', 'always_visible': True}),
            ], className='control-row'),
            dcc.Graph(id='rule-network', className='chart-full', responsive=True),
            dcc.Graph(id='rule-scatter', className='chart-full', responsive=True),
            html.Div(id='rule-table', className='table-wrap'),
        ]),

        # ── SECTION 4: ANOMALI ──
        html.Div(id='sec-anomali', className='section', children=[
            _section_header('4', 'Anomaly Detection',
                            'Klasifikasi anomali, konsentrasi risiko per segmen, '
                            'dan sebarannya di ruang rasio perilaku'),
            html.Div([
                dcc.Graph(id='anom-breakdown', className='chart-half',
                          figure=fig_anom_break, responsive=True),
                dcc.Graph(id='anom-by-seg', className='chart-half',
                          figure=fig_anom_seg, responsive=True),
            ], className='chart-row'),
            html.Div([
                html.Label('Sumbu X / Y anomaly scatter'),
                html.Div([
                    dcc.Dropdown(id='anom-x',
                                 options=[{'label': RATIO_LABEL[r], 'value': r}
                                          for r in RATIOS],
                                 value=RATIOS[1], clearable=False),
                    dcc.Dropdown(id='anom-y',
                                 options=[{'label': RATIO_LABEL[r], 'value': r}
                                          for r in RATIOS],
                                 value=RATIOS[2], clearable=False),
                ], className='control-row', style={'padding': '0'}),
            ], className='control-row'),
            dcc.Graph(id='anom-scatter', className='chart-full', responsive=True),
        ]),

        # ── SECTION 5: KREDIT ──
        html.Div(id='sec-kredit', className='section', children=[
            _section_header('5', 'Kredit',
                            'Sumber data dan tim penyusun'),
            html.Div([
                html.P([html.Strong('Dataset: '),
                        'Banking Dataset — ',
                        html.A('lihat sumber dataset', href=DATASET_URL,
                               target='_blank', rel='noopener',
                               style=_LINK)]),
                html.P([html.Strong('Repositori proyek: '),
                        html.A('buka repositori', href=REPO_URL,
                               target='_blank', rel='noopener',
                               style=_LINK)]),
                html.P(html.Strong('Kelompok 6:'),
                       style={'marginBottom': '4px'}),
                html.Table(
                    [html.Tr([html.Th('Nama', style=_TH),
                              html.Th('NIM', style=_TH),
                              html.Th('Email', style=_TH),
                              html.Th('Role', style=_TH)])] +
                    [html.Tr([html.Td(nama, style=_TD_B),
                              html.Td(nim, style=_TD),
                              html.Td(html.A(email, href=f'mailto:{email}'),
                                      style=_TD),
                              html.Td(role, style=_TD)])
                     for nama, nim, email, role in TEAM],
                    style={'fontSize': '13px', 'marginTop': '4px'}),
            ], className='table-wrap',
               style={'color': '#333', 'fontSize': '14px', 'lineHeight': '1.6'}),
        ]),

        html.Div('Banking Transaction · Knowledge Discovery · Python Dash',
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
        # Pakai graph + layout yang sudah di-precompute → callback ringan.
        return (fig_rule_network(rule_graph, rule_pos, min_lift),
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
