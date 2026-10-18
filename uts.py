# ============================================================
# UTS - INTEGRASI DATA MINING & BUSINESS INTELLIGENCE  (versi Streamlit)
# Sumber data : HANYA sheet "Data_Raw" (sheet lain diabaikan)
# Tahap       : 1 EDA -> 2 Preprocessing -> 3 Apriori -> 4 K-Means
#               -> 5 Dashboard BI -> 6 Rekomendasi Keputusan
# Style       : Calm Teal + animasi modern + grafik interaktif (Plotly)
#
# Instalasi   : pip install streamlit plotly pandas numpy scikit-learn openpyxl
# Jalankan    : streamlit run app_uts_streamlit.py
# Login       : adel / adelia123
# ============================================================

import re
import math
import warnings
import traceback
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

st.set_page_config(page_title="UTS DM & BI · Adelia Mutiara", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

USERNAME_VALID = "adel"
PASSWORD_VALID = "adelia123"
NA_TOKENS = {'', 'nan', 'none', 'null', 'na', 'n/a', '-', '--', '#n/a', 'nat', 'tidak ada'}
PAL = ['#3d74c9', '#4cb8d8', '#ef8a9b', '#e3b060', '#7b8cf0', '#9db7dd', '#f2a65a', '#7fb77e', '#b39ddb', '#5a8fd0']
BASE_COLORS = {'Loyal': '#3d74c9', 'Baru': '#4cb8d8', 'Potensial': '#f2a65a', 'Berisiko': '#ef8a9b'}

# ------------------------------------------------------------
# STATE (semua hasil analisis disimpan di session_state)
# ------------------------------------------------------------
def _ver():
    try:
        return tuple(int(x) for x in re.findall(r'\d+', st.__version__)[:2])
    except Exception:
        return (1, 0)


# Streamlit baru memakai width='stretch'; versi lama memakai use_container_width
STRETCH = {'width': 'stretch'} if _ver() >= (1, 50) else {'use_container_width': True}

S = st.session_state
for k, v in {'logged_in': False, 'raw': None, 'raw_name': '', 'audit': {}, 'clean': {}, 'feat': {},
             'apr': {}, 'km': {}, 'file_sig': None, 'page': 'eda'}.items():
    if k not in S:
        S[k] = v


# ------------------------------------------------------------
# FOTO IDENTITAS (taruh file foto di folder yang sama, atau folder static/assets)
# nama bebas selama diawali "adel", "foto", atau "photo" (jpg/jpeg/png/webp)
# ------------------------------------------------------------
import base64
from pathlib import Path


@st.cache_data(show_spinner=False)
def _find_photo():
    try:
        base = Path(__file__).resolve().parent
    except Exception:
        base = Path.cwd()
    for d in [base, base / 'static', base / 'assets', base / 'images', Path.cwd(), Path.cwd() / 'static']:
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp') and f.stem.lower().startswith(('adel', 'foto', 'photo')):
                    mime = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png', '.webp': 'webp'}[f.suffix.lower()]
                    return f'data:image/{mime};base64,' + base64.b64encode(f.read_bytes()).decode()
    return None


@st.cache_data(show_spinner=False)
def _find_asset(stem):
    try:
        base = Path(__file__).resolve().parent
    except Exception:
        base = Path.cwd()
    for d in [base, base / 'static', base / 'assets', base / 'images', Path.cwd(), Path.cwd() / 'static']:
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.stem.lower() == stem and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
                    mime = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png', '.webp': 'webp'}[f.suffix.lower()]
                    return f'data:image/{mime};base64,' + base64.b64encode(f.read_bytes()).decode()
    return None


def banner_uri():
    p = _find_asset('download')
    if p:
        return p
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 600 400'><defs><linearGradient id='a' x1='0' y1='0' x2='1' y2='1'>"
           "<stop offset='0' stop-color='#2f66c0'/><stop offset='1' stop-color='#9cc9f0'/></linearGradient></defs><rect width='600' height='400' fill='url(#a)'/>"
           "<g fill='white' fill-opacity='.9'><rect x='110' y='230' width='60' height='110' rx='10'/><rect x='200' y='180' width='60' height='160' rx='10'/>"
           "<rect x='290' y='130' width='60' height='210' rx='10'/><rect x='380' y='80' width='60' height='260' rx='10'/></g></svg>")
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()


def photo_uri():
    if S.get('photo_bytes'):
        return f"data:image/{S.get('photo_mime', 'jpeg')};base64," + base64.b64encode(S['photo_bytes']).decode()
    p = _find_photo()
    if p:
        return p
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
           "<stop offset='0' stop-color='#3d74c9'/><stop offset='1' stop-color='#7b8cf0'/></linearGradient></defs><rect width='100' height='100' fill='url(#g)'/>"
           "<text x='50' y='64' font-size='40' fill='white' text-anchor='middle' font-family='Arial' font-weight='bold'>AM</text></svg>")
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()


def photo_uploader(label='📷 Unggah / ganti foto', key='photo_up'):
    f = st.file_uploader(label, type=['jpg', 'jpeg', 'png', 'webp'], key=key)
    if f is not None and S.get('photo_sig') != (f.name, f.size):
        S['photo_bytes'] = f.getvalue()
        S['photo_mime'] = 'png' if f.name.lower().endswith('png') else ('webp' if f.name.lower().endswith('webp') else 'jpeg')
        S['photo_sig'] = (f.name, f.size)
        st.rerun()


# ============================================================
# HELPER UMUM
# ============================================================
def fmt_id(v, dec=0):
    try:
        if v is None or pd.isna(v):
            return '-'
        s = f"{float(v):,.{dec}f}"
        return s.replace(',', 'X').replace('.', ',').replace('X', '.')
    except Exception:
        return str(v)


def fi(v):
    return fmt_id(v, 0)


def rp(v):
    return 'Rp ' + fmt_id(v, 0)


def pct(v, dec=1):
    try:
        return fmt_id(float(v) * 100, dec) + '%'
    except Exception:
        return '-'


def fmt_cell(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return '-'
    if isinstance(v, pd.Timestamp):
        return v.strftime('%Y-%m-%d')
    if isinstance(v, (bool, np.bool_)):
        return 'Ya' if v else 'Tidak'
    if isinstance(v, (int, np.integer)):
        return fi(v)
    if isinstance(v, (float, np.floating)):
        return fmt_id(v, 0) if float(v).is_integer() and abs(v) >= 1000 else fmt_id(v, 2)
    return str(v)


def df_to_table(df, n=50):
    sub = df.head(n)
    return [str(c) for c in sub.columns], [[fmt_cell(v) for v in r] for r in sub.itertuples(index=False, name=None)]


def norm(s):
    return re.sub(r"[\s_\-\.]+", "", str(s).lower().strip())


def is_na(v):
    if v is None or v is pd.NaT:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip().lower() in NA_TOKENS


def parse_number(v):
    if is_na(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().lower().replace('rp', '').replace(' ', '').replace('%', '')
    if ',' in s and '.' in s:
        s = s.replace('.', '').replace(',', '.') if s.rfind(',') > s.rfind('.') else s.replace(',', '')
    elif ',' in s:
        p = s.split(',')
        s = s.replace(',', '') if (len(p) > 2 or len(p[-1]) == 3) else s.replace(',', '.')
    elif s.count('.') > 1 or re.fullmatch(r'-?\d{1,3}(\.\d{3})+', s):
        s = s.replace('.', '')
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_dates(series):
    s = series.where(~series.map(is_na), None)
    try:
        return pd.to_datetime(s, errors='coerce', dayfirst=True, format='mixed')
    except Exception:
        return pd.to_datetime(s, errors='coerce', dayfirst=True)


def canon_series(s):
    """Seragamkan teks: abaikan huruf besar/kecil & spasi, pakai varian paling sering."""
    t = s.map(lambda v: np.nan if is_na(v) else re.sub(r"\s+", " ", str(v)).strip())
    key = t.str.lower()
    tmp = pd.DataFrame({'k': key, 't': t}).dropna()
    mp = {k: g.value_counts().idxmax() for k, g in tmp.groupby('k')['t']}
    return key.map(mp)


def detect_elbow(ks, inertia):
    x, y = np.array(ks, float), np.array(inertia, float)
    xn = (x - x.min()) / ((x.max() - x.min()) or 1)
    yn = (y - y.min()) / ((y.max() - y.min()) or 1)
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    d = []
    for a, b in zip(xn, yn):
        v = p1 - np.array([a, b])
        d.append(abs((p2 - p1)[0] * v[1] - (p2 - p1)[1] * v[0]) / (np.linalg.norm(p2 - p1) or 1))
    return int(ks[int(np.argmax(d))])


def segcolor(name):
    return BASE_COLORS.get(re.sub(r' \d+$', '', str(name)), '#3d74c9')


# ============================================================
# MAPPING KOLOM
# ============================================================
ALIASES = {
    "TransactionID": ["transactionid", "idtransaksi", "invoice", "invoiceno", "orderid", "notransaksi", "trxid"],
    "Date": ["date", "tanggal", "transactiondate", "orderdate", "tgl", "tanggaltransaksi", "tgltransaksi"],
    "CustomerID": ["customerid", "idpelanggan", "memberid", "custid", "idcustomer", "pelanggan"],
    "CustomerName": ["customername", "namapelanggan", "nama", "name"],
    "ProductID": ["productid", "kodeproduk", "sku", "idproduk", "itemid", "itemcode"],
    "ProductName": ["productname", "namaproduk", "product", "produk", "item", "itemname", "namabarang"],
    "Category": ["category", "kategori", "productcategory", "kategoriproduk"],
    "Qty": ["qty", "quantity", "jumlah", "kuantitas", "jumlahbeli"],
    "UnitPrice": ["unitprice", "price", "harga", "hargasatuan", "hargaproduk"],
    "Discount": ["discount", "diskon", "discountpct", "diskonpct", "discountpercent"],
    "Payment": ["paymentmethod", "payment", "metodepembayaran", "metodebayar", "pembayaran"],
    "Total": ["totalprice", "totalamount", "total", "netsales", "sales", "subtotal", "linetotal", "totalharga"],
}


def prepare(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    mapping, used = {}, set()
    for std, al in ALIASES.items():
        for c in df.columns:
            if c not in used and norm(c) in al:
                mapping[c] = std
                used.add(c)
                break
    orig_cols = list(df.columns)
    df = df.rename(columns=mapping)
    if 'ProductName' not in df.columns and 'ProductID' in df.columns:
        df['ProductName'] = df['ProductID']
    need = ['TransactionID', 'Date', 'CustomerID', 'ProductName', 'Qty', 'UnitPrice']
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError("Kolom wajib tidak ditemukan: " + ', '.join(miss) + ". Kolom terbaca: " + ', '.join(map(str, orig_cols)))
    return df, mapping


# ============================================================
# TAHAP 1 - AUDIT / EDA
# ============================================================
def iqr_stats(s):
    s = s.dropna()
    if len(s) < 4:
        return 0, 0.0
    q1, q3 = s.quantile(.25), s.quantile(.75)
    fence = q3 + 1.5 * (q3 - q1)
    return int((s > fence).sum()), float(fence)


def run_audit():
    res = {'success': False, 'error': None}
    try:
        df, mapping = prepare(S['raw'])
        n = len(df)
        dates = parse_dates(df['Date'])
        qty = df['Qty'].map(parse_number)
        price = df['UnitPrice'].map(parse_number)
        disc = df['Discount'].map(parse_number).fillna(0) if 'Discount' in df.columns else pd.Series(0.0, index=df.index)
        disc = disc.where(disc <= 1, disc / 100).clip(0, 1)
        line = qty * price * (1 - disc)
        if 'Total' in df.columns:
            tot = df['Total'].map(parse_number)
            sales = tot.where(tot.notna(), line)
        else:
            sales = line
        sales_ok = sales.where(sales > 0)

        inv = {v: k for k, v in mapping.items()}
        dtypes = []
        for c in df.columns:
            tipe = 'numerik' if c in ('Qty', 'UnitPrice', 'Discount', 'Total') else ('tanggal' if c == 'Date' else 'teks / kategori')
            ex = df[c].dropna()
            dtypes.append({'Kolom': c, 'Nama asli': inv.get(c, c), 'dtype': str(df[c].dtype), 'Tipe': tipe,
                           'Terisi': int(df[c].notna().sum()), 'Unik': int(df[c].nunique(dropna=True)),
                           'Contoh': str(ex.iloc[0]) if len(ex) else '-'})

        missing = {c: int(df[c].map(is_na).sum()) for c in df.columns}
        missing = {c: v for c, v in missing.items() if v > 0}
        raw_txt = df.astype(str).apply(lambda c: c.str.strip())
        dup_exact = int(raw_txt.duplicated().sum())
        dup_line = int(df.duplicated(subset=['TransactionID', 'ProductName']).sum())

        incons = []
        for col, lbl in [('Category', 'Kategori'), ('ProductName', 'Nama produk'), ('Payment', 'Metode pembayaran'),
                         ('CustomerID', 'ID pelanggan')]:
            if col in df.columns:
                s = df[col].dropna().astype(str)
                s = s[~s.str.strip().str.lower().isin(NA_TOKENS)]
                key = s.str.replace(r"\s+", "", regex=True).str.lower()
                grp = s.groupby(key).agg(lambda x: sorted(set(x)))
                multi = grp[grp.map(len) > 1]
                incons.append({'Kolom': lbl, 'Unik raw': int(s.nunique()), 'Unik seragam': int(key.nunique()),
                               'Varian ganda': int(len(multi)),
                               'Contoh': ' | '.join(' / '.join(repr(i) for i in v[:3]) for v in multi.head(3))})

        outl = []
        for lbl, ser in [('Qty', qty), ('UnitPrice', price), ('Nilai baris (penjualan)', sales)]:
            n_o, fence = iqr_stats(ser.where(ser > 0))
            outl.append({'Variabel': lbl, 'Nilai <= 0 / bukan angka': int((ser <= 0).sum()), 'Outlier': n_o,
                         'Batas atas IQR': fence, 'Min': float(ser.min()), 'Maks': float(ser.max())})

        bad = (df[['TransactionID', 'Date', 'CustomerID', 'ProductName', 'Qty', 'UnitPrice']].apply(
            lambda c: c.map(is_na)).any(axis=1)) | (qty <= 0) | (price <= 0) | dates.isna()

        monthly = sales_ok.groupby(dates.dt.to_period('M')).sum().dropna()
        cat_raw = df['Category'].dropna().astype(str).value_counts().head(12) if 'Category' in df.columns else pd.Series(dtype=int)

        insights = []
        try:
            mi = monthly.idxmax()
            insights.append(f"Penjualan sementara tertinggi terjadi pada bulan {mi} sebesar {rp(monthly.max())} "
                            f"({pct(monthly.max() / monthly.sum())} dari total).")
        except Exception:
            pass
        try:
            cc = canon_series(df['Category'])
            cs = sales_ok.groupby(cc).sum().sort_values(ascending=False)
            insights.append(f"Kategori terbesar adalah {cs.index[0]} dengan kontribusi {pct(cs.iloc[0] / cs.sum())} dari penjualan.")
        except Exception:
            pass
        insights.append(f"Sekitar {pct(bad.mean())} baris ({fi(bad.sum())} dari {fi(n)}) bermasalah pada kolom kunci, "
                        f"sehingga preprocessing wajib dilakukan sebelum Apriori dan K-Means.")
        try:
            pm = canon_series(df['Payment']).value_counts()
            insights.append(f"Metode pembayaran dominan: {pm.index[0]} ({pct(pm.iloc[0] / pm.sum())} baris).")
        except Exception:
            pass
        try:
            tpt = df.groupby('TransactionID')['ProductName'].nunique()
            insights.append(f"Rata-rata sementara {tpt.mean():.2f} produk unik per transaksi, "
                            f"sehingga pola kombinasi produk terbatas dan perlu support rendah pada Apriori.")
        except Exception:
            pass

        res.update({
            'rows': n, 'cols': len(df.columns), 'dtypes': pd.DataFrame(dtypes), 'missing': missing,
            'missing_total': int(sum(missing.values())),
            'date_min': dates.min(), 'date_max': dates.max(), 'date_invalid': int(dates.isna().sum()),
            'n_tx': int(df['TransactionID'].nunique()), 'n_cust': int(df['CustomerID'].nunique()),
            'n_prod': int(canon_series(df['ProductName']).nunique()), 'sales_prov': float(sales_ok.sum()),
            'dup_exact': dup_exact, 'dup_line': dup_line, 'incons': pd.DataFrame(incons), 'outliers': pd.DataFrame(outl),
            'bad_rows': int(bad.sum()),
            'monthly': (monthly.index.astype(str).tolist(), [float(v) for v in monthly.values]),
            'cat_raw': (list(cat_raw.index), [int(v) for v in cat_raw.values]),
            'insights': insights, 'success': True})
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    S['audit'] = res
    return res


# ============================================================
# TAHAP 2 - PREPROCESSING (Tugas 4-5)
# ============================================================
def run_cleaning():
    res = {'success': False, 'error': None}
    try:
        if S['raw'] is None:
            raise ValueError('Belum ada data.')
        df, _ = prepare(S['raw'].copy())          # salinan kerja, Data_Raw tidak ditimpa
        for c in df.columns:
            df[c] = df[c].map(lambda v: np.nan if is_na(v) else v)
        n_start = len(df)
        miss_start = int(df.isna().sum().sum())
        log = []

        def add(masalah, keputusan, jenis, n, b, a):
            log.append({'Masalah': masalah, 'Keputusan': keputusan, 'Jenis': jenis, 'Terdampak': int(n),
                        'Baris sebelum': int(b), 'Baris sesudah': int(a)})

        b = len(df)
        d = df.astype(str).duplicated()
        df = df[~d]
        add('Duplikat baris identik', 'Hapus, simpan 1 baris', 'hapus', d.sum(), b, len(df))

        changed = 0
        for c in ['CustomerID', 'TransactionID', 'ProductID']:
            if c in df.columns:
                new = df[c].map(lambda v: np.nan if is_na(v) else re.sub(r'\s+', '', str(v)).upper())
                changed += int(((df[c].astype(str) != new.astype(str)) & new.notna()).sum())
                df[c] = new
        for c in ['ProductName', 'Category', 'Payment', 'CustomerName']:
            if c in df.columns:
                new = canon_series(df[c])
                changed += int(((df[c].astype(str) != new.astype(str)) & new.notna()).sum())
                df[c] = new
        if 'ProductID' in df.columns:
            m = df.dropna(subset=['ProductID', 'ProductName']).groupby('ProductID')['ProductName'].agg(lambda s: s.value_counts().idxmax())
            df['ProductName'] = df['ProductID'].map(m).fillna(df['ProductName'])
        add('Inkonsistensi teks (huruf, spasi, ejaan ID/nama/kategori/metode bayar)',
            'Seragamkan ke varian paling sering; ID di-UPPERCASE tanpa spasi', 'perbaiki', changed, len(df), len(df))

        df['Date'] = parse_dates(df['Date'])
        b = len(df)
        m = df['Date'].isna()
        df = df[~m]
        add('Tanggal kosong / format tidak valid', 'Hapus baris (Recency dan tren tidak bisa dihitung)', 'hapus', m.sum(), b, len(df))
        year = int(df['Date'].dt.year.mode().iloc[0])
        b = len(df)
        m = df['Date'].dt.year != year
        df = df[~m]
        add(f'Tanggal di luar tahun {year}', 'Hapus baris (di luar periode data)', 'hapus', m.sum(), b, len(df))

        for c in ['Qty', 'UnitPrice', 'Discount', 'Total']:
            if c in df.columns:
                df[c] = df[c].map(parse_number)
        if 'Discount' not in df.columns:
            df['Discount'] = 0.0
        n_dm = int(df['Discount'].isna().sum())
        df['Discount'] = df['Discount'].fillna(0)
        if df['Discount'].quantile(.99) > 1:
            df['Discount'] = df['Discount'] / 100.0
        m = (df['Discount'] < 0) | (df['Discount'] > 1)
        df.loc[m, 'Discount'] = 0.0
        add('Diskon kosong / di luar 0-100%', 'Imputasi 0 (dianggap tanpa diskon); persen diubah ke pecahan', 'imputasi', n_dm + m.sum(), len(df), len(df))

        b = len(df)
        m = df['TransactionID'].isna() | df['ProductName'].isna()
        df = df[~m]
        add('Transaction_ID / produk kosong', 'Hapus baris (tidak bisa dimasukkan ke basket)', 'hapus', m.sum(), b, len(df))
        n_c = int(df['CustomerID'].isna().sum())
        df['CustomerID'] = df['CustomerID'].fillna('UNKNOWN')
        add('ID pelanggan kosong', "Isi 'UNKNOWN' dan tandai: tetap dipakai untuk penjualan/Apriori, dikeluarkan dari RFM",
            'tandai', n_c, len(df), len(df))
        if 'Category' not in df.columns:
            df['Category'] = np.nan
        cmap = df.dropna(subset=['Category']).groupby('ProductName')['Category'].agg(lambda s: s.value_counts().idxmax())
        n_cat = int(df['Category'].isna().sum())
        df['Category'] = df['Category'].fillna(df['ProductName'].map(cmap)).fillna('Tidak Diketahui')
        add('Kategori kosong', "Imputasi dari kategori paling sering pada produk yang sama, sisanya 'Tidak Diketahui'",
            'imputasi', n_cat, len(df), len(df))
        if 'Payment' not in df.columns:
            df['Payment'] = np.nan
        n_pay = int(df['Payment'].isna().sum())
        df['Payment'] = df['Payment'].fillna('Tidak Diketahui')
        add('Metode pembayaran kosong', "Imputasi 'Tidak Diketahui'", 'imputasi', n_pay, len(df), len(df))

        b = len(df)
        m = df['Qty'].isna() | (df['Qty'] <= 0)
        df = df[~m]
        add('Qty kosong / <= 0 / bukan angka', 'Hapus baris', 'hapus', m.sum(), b, len(df))
        med = df[df['UnitPrice'] > 0].groupby('ProductName')['UnitPrice'].median()
        pm = df['ProductName'].map(med)
        bad_p = df['UnitPrice'].isna() | (df['UnitPrice'] <= 0)
        df.loc[bad_p, 'UnitPrice'] = pm[bad_p]
        ratio = df['UnitPrice'] / pm
        anom = (ratio > 10) | (ratio < 0.1)
        df.loc[anom, 'UnitPrice'] = pm[anom]
        add('Harga kosong / <= 0 / tidak wajar (>10x atau <0,1x median produk)',
            'Imputasi dengan median harga produk yang sama', 'imputasi', bad_p.sum() + anom.sum(), len(df), len(df))
        b = len(df)
        m = df['UnitPrice'].isna()
        df = df[~m]
        add('Harga tidak bisa diimputasi (produk tanpa harga valid)', 'Hapus baris', 'hapus', m.sum(), b, len(df))
        q1, q3 = df['Qty'].quantile(.25), df['Qty'].quantile(.75)
        iqr = max(q3 - q1, 1)
        b = len(df)
        m = df['Qty'] > q3 + 5 * iqr
        df = df[~m]
        add('Qty ekstrem (> Q3 + 5xIQR)', 'Hapus baris (dianggap salah input)', 'hapus', m.sum(), b, len(df))
        df['Flag_Outlier'] = df['Qty'] > q3 + 1.5 * iqr

        df['Net_Sales'] = (df['Qty'] * df['UnitPrice'] * (1 - df['Discount'])).round(2)
        n_inc = 0
        if 'Total' in df.columns:
            inc = df['Total'].notna() & ((df['Total'] - df['Net_Sales']).abs() > df['Net_Sales'].abs() * 0.01 + 1)
            n_inc = int(inc.sum())
            df = df.drop(columns=['Total'])
        add('Total baris tidak konsisten dengan Qty x Harga x (1-Diskon)', 'Hitung ulang sebagai Net_Sales', 'perbaiki', n_inc, len(df), len(df))

        b = len(df)
        d = df.duplicated(subset=['TransactionID', 'Date', 'CustomerID', 'ProductName', 'Qty', 'UnitPrice'])
        df = df[~d]
        add('Duplikat baris item setelah standarisasi', 'Hapus', 'hapus', d.sum(), b, len(df))
        key = ['TransactionID', 'ProductName']
        dm = df.duplicated(key, keep=False)
        n_dl = int(df.duplicated(key).sum())
        if dm.any():
            agg = {c: 'first' for c in df.columns if c not in key}
            agg['Qty'] = 'sum'
            agg['Net_Sales'] = 'sum'
            agg['Flag_Outlier'] = 'max'
            b = len(df)
            df = df.groupby(key, as_index=False, sort=False).agg(agg)
            add('Produk sama muncul >1 baris dalam satu transaksi',
                'Gabungkan (Qty dan Net_Sales dijumlah) agar 1 produk = 1 baris per transaksi', 'perbaiki', n_dl, b, len(df))

        conf = int((df.groupby('TransactionID')['CustomerID'].nunique() > 1).sum() + (df.groupby('TransactionID')['Date'].nunique() > 1).sum())
        df['CustomerID'] = df.groupby('TransactionID')['CustomerID'].transform('first')
        df['Date'] = df.groupby('TransactionID')['Date'].transform('min')
        add('Satu Transaction_ID punya >1 pelanggan / tanggal', 'Samakan ke nilai pertama / tanggal paling awal', 'perbaiki', conf, len(df), len(df))

        tx_tot = df.groupby('TransactionID')['Net_Sales'].transform('sum')
        t1, t3 = tx_tot.quantile(.25), tx_tot.quantile(.75)
        big = tx_tot > t3 + 5 * max(t3 - t1, 1)
        df['Flag_Outlier'] = df['Flag_Outlier'] | big
        add('Total transaksi sangat besar (> Q3 + 5xIQR per transaksi)', 'Tandai Flag_Outlier untuk diperiksa, tidak dihapus',
            'tandai', df.loc[big, 'TransactionID'].nunique(), len(df), len(df))

        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        cols = ['TransactionID', 'Date', 'Month', 'CustomerID'] + [c for c in ['CustomerName', 'ProductID'] if c in df.columns] + \
               ['ProductName', 'Category', 'Qty', 'UnitPrice', 'Discount', 'Payment', 'Net_Sales', 'Flag_Outlier']
        df = df[cols].sort_values(['Date', 'TransactionID']).reset_index(drop=True)

        au = S['audit']
        quality = pd.DataFrame([
            ('Jumlah baris', n_start, len(df)),
            ('Sel kosong', miss_start, int(df.isna().sum().sum())),
            ('Transaksi unik', au.get('n_tx', 0), int(df['TransactionID'].nunique())),
            ('Pelanggan unik', au.get('n_cust', 0), int(df['CustomerID'].nunique())),
            ('Produk unik', au.get('n_prod', 0), int(df['ProductName'].nunique())),
            ('Total penjualan (Rp)', au.get('sales_prov', 0), float(df['Net_Sales'].sum()))],
            columns=['Metrik', 'Sebelum (Data_Raw)', 'Sesudah (Data_Clean)'])
        res.update({'df': df, 'log': pd.DataFrame(log), 'quality': quality, 'rows_before': n_start, 'rows_after': len(df),
                    'year': year, 'success': True})
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    S['clean'] = res
    S['feat'], S['apr'], S['km'] = {}, {}, {}
    return res


# ============================================================
# TAHAP 2 (lanjutan) - BASKET, MATRIKS, RFM, SCALING (Tugas 6-8)
# ============================================================
def run_features():
    res = {'success': False, 'error': None}
    try:
        if not S['clean'].get('success'):
            raise ValueError('Lakukan Preprocessing terlebih dahulu.')
        d = S['clean']['df']
        tx = d.groupby('TransactionID').agg(Date=('Date', 'min'), CustomerID=('CustomerID', 'first'),
                                            basket_size=('ProductName', 'nunique'), Total_Qty=('Qty', 'sum'),
                                            Net_Sales=('Net_Sales', 'sum')).reset_index()
        matrix = (pd.crosstab(d['TransactionID'], d['ProductName']) > 0).astype(int)

        ref = d['Date'].max().normalize() + pd.Timedelta(days=1)
        t = tx[tx['CustomerID'] != 'UNKNOWN']
        g = t.groupby('CustomerID')
        rfm = pd.DataFrame({'Last_Transaction': g['Date'].max(),
                            'Recency': (ref - g['Date'].max()).dt.days,
                            'Frequency': g['TransactionID'].nunique(),
                            'Monetary': g['Net_Sales'].sum().round(2)}).reset_index()
        rfm['Log_Frequency'] = np.log1p(rfm['Frequency'])
        rfm['Log_Monetary'] = np.log1p(rfm['Monetary'])
        Z = StandardScaler().fit_transform(rfm[['Recency', 'Log_Frequency', 'Log_Monetary']])
        rfm['Z_Recency'], rfm['Z_Log_Frequency'], rfm['Z_Log_Monetary'] = Z[:, 0], Z[:, 1], Z[:, 2]
        rfm = rfm.sort_values('CustomerID').reset_index(drop=True)

        sk = {'Frequency': float(rfm['Frequency'].skew()), 'Log_Frequency': float(rfm['Log_Frequency'].skew()),
              'Monetary': float(rfm['Monetary'].skew()), 'Log_Monetary': float(rfm['Log_Monetary'].skew())}
        bs = tx['basket_size'].value_counts().sort_index()
        res.update({'tx': tx, 'matrix': matrix, 'rfm': rfm, 'ref': ref, 'skew': sk,
                    'avg_basket': float(tx['basket_size'].mean()),
                    'basket_dist': (bs.index.tolist(), [int(v) for v in bs.values]),
                    'n_unknown': int((tx['CustomerID'] == 'UNKNOWN').sum()), 'success': True})
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    S['feat'] = res
    S['apr'], S['km'] = {}, {}
    return res


# ============================================================
# TAHAP 3 - APRIORI (implementasi sendiri, tanpa mlxtend)
# ============================================================
def apriori_fast(B, min_sup, max_len=3):
    n = B.shape[0]
    sup1 = B.sum(0) / n
    freq = {(int(i),): float(sup1[i]) for i in np.where(sup1 >= min_sup)[0]}
    current = sorted(freq)
    k = 2
    while current and k <= max_len and len(freq) < 20000:
        cs = set(current)
        cand = set()
        for a in current:
            for b in current:
                if a[:-1] == b[:-1] and a[-1] < b[-1]:
                    c = a + (b[-1],)
                    if all(c[:j] + c[j + 1:] in cs for j in range(k)):
                        cand.add(c)
        nxt = []
        for c in cand:
            s = float(B[:, list(c)].all(axis=1).mean())
            if s >= min_sup:
                freq[c] = s
                nxt.append(c)
        current = sorted(nxt)
        k += 1
    return freq


def short(s, n=22):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + '…'


def network_fig(rules):
    """Network graph aturan (Plotly): panah antecedent -> consequent, tebal garis = lift."""
    nodes = []
    for r in rules:
        for it in list(r['ant']) + list(r['con']):
            if it not in nodes:
                nodes.append(it)
    if not nodes:
        return None
    pos = {nd: (math.cos(2 * math.pi * i / len(nodes) - math.pi / 2), math.sin(2 * math.pi * i / len(nodes) - math.pi / 2))
           for i, nd in enumerate(nodes)}
    ml = max(r['lift'] for r in rules) or 1
    fig = go.Figure()
    for r in rules:
        for a in r['ant']:
            for c in r['con']:
                (x1, y1), (x2, y2) = pos[a], pos[c]
                fig.add_annotation(x=x2, y=y2, ax=x1, ay=y1, xref='x', yref='y', axref='x', ayref='y', showarrow=True,
                                   arrowhead=3, arrowsize=1.1, arrowwidth=1.2 + 3.2 * r['lift'] / ml,
                                   arrowcolor='rgba(61,116,201,.55)', standoff=10, startstandoff=10)
    fig.add_trace(go.Scatter(x=[p[0] for p in pos.values()], y=[p[1] for p in pos.values()], mode='markers+text',
                             text=[short(n) for n in pos], textposition='top center',
                             marker=dict(size=18, color='#4cb8d8', line=dict(color='white', width=2.5)),
                             hovertext=list(pos.keys()), hoverinfo='text', textfont=dict(size=11, color='#1b2a4a')))
    fig.update_xaxes(visible=False, range=[-1.6, 1.6])
    fig.update_yaxes(visible=False, range=[-1.4, 1.4], scaleanchor='x')
    fig.update_layout(showlegend=False, height=460, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig


def run_apriori(min_sup, min_conf):
    res = {'success': False, 'error': None, 'min_sup': min_sup, 'min_conf': min_conf}
    try:
        if not S['feat'].get('success'):
            raise ValueError('Lakukan Basket & RFM terlebih dahulu.')
        M = S['feat']['matrix']
        items = list(M.columns)
        B = M.values.astype(bool)
        n = B.shape[0]
        freq = apriori_fast(B, min_sup, 3)
        rules = []
        for iset, sup in freq.items():
            if len(iset) < 2:
                continue
            for r in range(1, len(iset)):
                for ant in combinations(iset, r):
                    con = tuple(sorted(set(iset) - set(ant)))
                    sa, sc = freq.get(ant), freq.get(con)
                    if not sa or not sc:
                        continue
                    conf = sup / sa
                    if conf >= min_conf:
                        rules.append({'ant': tuple(items[i] for i in ant), 'con': tuple(items[i] for i in con),
                                      'support': sup, 'confidence': conf, 'lift': conf / sc,
                                      'leverage': sup - sa * sc, 'sup_ant': sa, 'sup_con': sc, 'n': int(round(sup * n))})
        pos = [r for r in rules if r['lift'] > 1]
        pos.sort(key=lambda r: (r['lift'], r['confidence'], r['support']), reverse=True)
        top = pos[:10]
        fs = sorted([(tuple(items[i] for i in k), v) for k, v in freq.items() if len(k) >= 2], key=lambda x: -x[1])[:10]
        fs_df = pd.DataFrame({'Itemset': [' + '.join(k) for k, _ in fs], 'Ukuran': [len(k) for k, _ in fs],
                              'Support': [round(v, 4) for _, v in fs], 'Transaksi': [int(round(v * n)) for _, v in fs]})
        cols = ['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift', 'Leverage', 'Transaksi']
        rules_df = pd.DataFrame([{'Antecedent': ' + '.join(r['ant']), 'Consequent': ' + '.join(r['con']),
                                  'Support': round(r['support'], 4), 'Confidence': round(r['confidence'], 4),
                                  'Lift': round(r['lift'], 3), 'Leverage': round(r['leverage'], 4), 'Transaksi': r['n']}
                                 for r in pos]) if pos else pd.DataFrame(columns=cols)
        res.update({'n_tx': n, 'n_items': len(items), 'n_freq': len(freq), 'n_rules_all': len(rules), 'n_rules': len(pos),
                    'top': top, 'rules_df': rules_df, 'top_df': rules_df.head(10), 'fs_df': fs_df,
                    'network': network_fig(top[:8]) if top else None,
                    'scatter': pd.DataFrame([{'Support': r['support'], 'Confidence': r['confidence'], 'Lift': r['lift'],
                                              'Aturan': ' + '.join(r['ant']) + ' → ' + ' + '.join(r['con'])} for r in pos[:80]]),
                    'min_count': int(math.ceil(min_sup * n)), 'success': True})
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    S['apr'] = res
    return res


# ============================================================
# TAHAP 4 - K-MEANS
# ============================================================
STRAT = {
    'Loyal': ('Pelanggan bernilai tinggi: belanja baru-baru ini, sering, dan nominal besar.',
              'Program loyalitas bertingkat, akses awal produk baru, reward non-diskon, dan bundling premium. Hindari diskon besar.'),
    'Baru': ('Pelanggan yang aktif baru-baru ini tetapi frekuensi dan nilai belanja masih rendah.',
             'Voucher sambutan, rekomendasi produk pelengkap (dari aturan Apriori), dan follow-up untuk transaksi ke-2 dan ke-3.'),
    'Potensial': ('Pelanggan dengan aktivitas dan nilai menengah yang bisa ditingkatkan.',
                  'Cross-selling berdasarkan aturan asosiasi, promo kelipatan belanja, dan penawaran personal yang naik bertahap.'),
    'Berisiko': ('Pelanggan yang lama tidak kembali dengan frekuensi/nilai belanja relatif rendah.',
                 'Kampanye win-back berbatas waktu, survei alasan berhenti, pengingat personal, dan hentikan iklan massal ke segmen ini.'),
}


def run_kmeans(k_req=None):
    res = {'success': False, 'error': None}
    try:
        if not S['feat'].get('success'):
            raise ValueError('Lakukan Basket & RFM terlebih dahulu.')
        rfm = S['feat']['rfm'].copy()
        X = rfm[['Z_Recency', 'Z_Log_Frequency', 'Z_Log_Monetary']].values
        n = len(rfm)
        kmax = min(8, n - 1)
        if kmax < 3:
            raise ValueError('Jumlah pelanggan terlalu sedikit untuk K-Means.')
        ks, inertia, sil = list(range(2, kmax + 1)), [], []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=500).fit(X)
            inertia.append(float(km.inertia_))
            sil.append(float(silhouette_score(X, km.labels_, sample_size=min(n, 5000), random_state=42)))
        cand = [k for k in ks if 3 <= k <= 6] or ks
        rec_k = max(cand, key=lambda k: sil[ks.index(k)])
        elbow_k = detect_elbow(ks, inertia)
        k = int(k_req) if k_req and int(k_req) in ks else rec_k
        reason = (f"Elbow menyarankan k = {elbow_k}; Silhouette tertinggi pada rentang 3 sampai 6 ada di k = {rec_k} "
                  f"(skor {sil[ks.index(rec_k)]:.3f}). Rentang 3 sampai 6 dipilih karena cukup untuk membedakan segmen "
                  f"loyal, baru, potensial, dan berisiko tetapi tidak terlalu rumit untuk dikelola tim pemasaran. "
                  f"k yang dipakai: {k} (silhouette {sil[ks.index(k)]:.3f}).")
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=500).fit(X)
        rfm['Cluster'] = km.labels_

        prof = rfm.groupby('Cluster').agg(
            Jumlah=('CustomerID', 'count'), Avg_Recency=('Recency', 'mean'), Med_Recency=('Recency', 'median'),
            Avg_Frequency=('Frequency', 'mean'), Med_Frequency=('Frequency', 'median'),
            Avg_Monetary=('Monetary', 'mean'), Med_Monetary=('Monetary', 'median'),
            Revenue=('Monetary', 'sum'), Z_R=('Z_Recency', 'mean'), Z_F=('Z_Log_Frequency', 'mean'),
            Z_M=('Z_Log_Monetary', 'mean')).reset_index()
        prof['Pct_Pelanggan'] = prof['Jumlah'] / prof['Jumlah'].sum()
        prof['Pct_Revenue'] = prof['Revenue'] / prof['Revenue'].sum()
        prof['Skor'] = -prof['Z_R'] + prof['Z_F'] + prof['Z_M']
        order = prof.sort_values('Skor', ascending=False)['Cluster'].tolist()
        med_r, mean_f = rfm['Recency'].median(), rfm['Frequency'].mean()
        labels = {order[0]: 'Loyal'}
        for c in order[1:]:
            r = prof.loc[prof.Cluster == c].iloc[0]
            if c == order[-1] and r.Avg_Recency > med_r:
                labels[c] = 'Berisiko'
            elif r.Avg_Recency <= med_r and r.Avg_Frequency <= mean_f:
                labels[c] = 'Baru'
            else:
                labels[c] = 'Potensial'
        if 'Berisiko' not in labels.values():
            labels[order[-1]] = 'Berisiko'
        seen, final = {}, {}
        for c in order:
            base = labels[c]
            seen[base] = seen.get(base, 0) + 1
            final[c] = base if seen[base] == 1 else f"{base} {seen[base]}"
        prof['Segmen'] = prof['Cluster'].map(final)
        prof['Basis'] = prof['Segmen'].map(lambda s: re.sub(r' \d+$', '', s))
        prof['Karakteristik'] = prof.apply(lambda r: f"{STRAT[r.Basis][0]} Median Recency {r.Med_Recency:.0f} hari, "
                                           f"Frequency {r.Med_Frequency:.0f}x, Monetary {rp(r.Med_Monetary)}.", axis=1)
        prof['Strategi'] = prof['Basis'].map(lambda b: STRAT[b][1])
        prof = prof.sort_values('Skor', ascending=False).reset_index(drop=True)
        rfm = rfm.merge(prof[['Cluster', 'Segmen']], on='Cluster', how='left')

        ptab = prof[['Cluster', 'Segmen', 'Jumlah', 'Avg_Recency', 'Med_Recency', 'Avg_Frequency', 'Med_Frequency',
                     'Avg_Monetary', 'Med_Monetary', 'Pct_Revenue']].copy()
        ptab['Pct_Revenue'] = ptab['Pct_Revenue'].map(lambda v: pct(v))
        ptab = ptab.round(1)
        res.update({'ks': ks, 'inertia': inertia, 'sil': sil, 'rec_k': rec_k, 'elbow_k': elbow_k, 'k': k, 'reason': reason,
                    'profile': prof, 'ptable': ptab, 'labeled': rfm,
                    'scatter': rfm.sample(min(len(rfm), 1000), random_state=1)[['Frequency', 'Monetary', 'Recency', 'Segmen']],
                    'seg_names': prof['Segmen'].tolist(), 'success': True})
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    S['km'] = res
    return res


# ============================================================
# TAHAP 5 - DASHBOARD (dengan filter)
# ============================================================
def build_dashboard(start, end, cat, seg):
    d = S['clean']['df']
    f = d[(d['Date'] >= start) & (d['Date'] < end + pd.Timedelta(days=1))]
    if cat and cat != 'Semua':
        f = f[f['Category'] == cat]
    if S['km'].get('success'):
        seg_map = S['km']['labeled'].set_index('CustomerID')['Segmen']
        f = f.assign(Segmen=f['CustomerID'].map(seg_map))
        if seg and seg != 'Semua':
            f = f[f['Segmen'] == seg]
    out = {'empty': len(f) == 0}
    if f.empty:
        return out
    bs = f.groupby('TransactionID')['ProductName'].nunique()
    known = f[f['CustomerID'] != 'UNKNOWN']
    out['kpi'] = {'sales': float(f['Net_Sales'].sum()), 'tx': int(f['TransactionID'].nunique()),
                  'cust': int(known['CustomerID'].nunique()), 'basket': float(bs.mean())}
    out['monthly'] = f.groupby('Month').agg(Sales=('Net_Sales', 'sum'), Tx=('TransactionID', 'nunique')).reset_index()
    out['top_prod'] = f.groupby('ProductName')['Net_Sales'].sum().sort_values(ascending=False).head(10)
    out['top_cat'] = f.groupby('Category')['Net_Sales'].sum().sort_values(ascending=False).head(10)
    out['pay'] = f.groupby('Payment')['Net_Sales'].sum().sort_values(ascending=False)
    if 'Segmen' in f.columns:
        rows = []
        for sg, g in f.dropna(subset=['Segmen']).groupby('Segmen'):
            tp3 = g.groupby('ProductName')['Net_Sales'].sum().sort_values(ascending=False).head(3)
            rows.append({'Segmen': sg, 'Pelanggan': g['CustomerID'].nunique(), 'Penjualan': rp(g['Net_Sales'].sum()),
                         '3 produk teratas': ', '.join(tp3.index)})
        out['seg_pattern'] = pd.DataFrame(rows)
    ins = []
    tp, tc, mo = out['top_prod'], out['top_cat'], out['monthly']
    if len(tp):
        ins.append(f"Produk teratas {tp.index[0]} menyumbang {pct(tp.iloc[0] / f['Net_Sales'].sum())} penjualan pada filter ini.")
    if len(mo) > 1:
        r = mo.loc[mo['Sales'].idxmax()]
        ins.append(f"Bulan puncak penjualan: {r['Month']} ({rp(r['Sales'])}).")
    if len(tc):
        ins.append(f"Kategori terkuat: {tc.index[0]} ({pct(tc.iloc[0] / tc.sum())} dari 10 kategori teratas).")
    out['insights'] = ins
    return out


# ============================================================
# TAHAP 6 - REKOMENDASI
# ============================================================
def build_recs():
    ap, km, d = S['apr'], S['km'], S['clean']['df']
    n_tx = len(S['feat']['tx'])
    recs = []
    avg_item = d.groupby('ProductName')['Net_Sales'].mean()
    if ap.get('top'):
        r = ap['top'][0]
        a, c = ' + '.join(r['ant']), ' + '.join(r['con'])
        n_a = int(round(r['sup_ant'] * n_tx))
        price_c = float(np.mean([avg_item.get(i, avg_item.mean()) for i in r['con']]))
        extra = n_a * 0.05 * price_c
        recs.append({
            'judul': f'Bundling dan penempatan rak: {a} dengan {c}', 'icon': '📦', 'warna': '#3d74c9',
            'bukti': f"Aturan teratas: support {pct(r['support'], 2)}, confidence {pct(r['confidence'])}, lift {r['lift']:.2f} "
                     f"({fi(r['n'])} transaksi memuat keduanya, {fi(n_a)} transaksi memuat {a}).",
            'tindakan': f"Tawarkan paket {a} + {c} (diskon ringan) dan letakkan keduanya berdekatan di rak/halaman produk.",
            'kpi': f"Attach rate {c} pada transaksi berisi {a}; rata-rata basket_size; penjualan paket.",
            'estimasi': f"Jika 5% dari {fi(n_a)} transaksi berisi {a} menambah {c}: tambahan penjualan sekitar {rp(extra)}; "
                        f"biaya diskon 10% sekitar {rp(extra * .1)}; manfaat bersih sekitar {rp(extra * .9)} per tahun.",
            'asumsi': 'Adopsi 5% dan harga rata-rata produk tambahan sama dengan histori. Angka ini asumsi, bukan hasil pengujian.',
            'evaluasi': 'Uji A/B: sebagian toko/hari memakai paket dan rak baru, sisanya tetap. Bandingkan attach rate dan basket_size selama 4 sampai 8 minggu.'})
    else:
        recs.append({'judul': 'Bundling dan penempatan rak', 'icon': '📦', 'warna': '#3d74c9',
                     'bukti': 'Belum ada aturan dengan lift > 1 pada parameter saat ini. Turunkan minimum support di menu Apriori.',
                     'tindakan': 'Jalankan ulang Apriori dengan support lebih rendah.', 'kpi': '-', 'estimasi': '-', 'asumsi': '-', 'evaluasi': '-'})
    p = km['profile']
    risk = p[p['Basis'] == 'Berisiko'].iloc[0] if (p['Basis'] == 'Berisiko').any() else p.iloc[-1]
    aov = risk.Avg_Monetary / max(risk.Avg_Frequency, 1)
    extra = risk.Jumlah * 0.10 * aov
    cost = extra * 0.10 + risk.Jumlah * 1000
    recs.append({
        'judul': f"Retensi pelanggan segmen {risk.Segmen}", 'icon': '🤝', 'warna': '#ef8a9b',
        'bukti': f"Segmen {risk.Segmen}: {fi(risk.Jumlah)} pelanggan ({pct(risk.Pct_Pelanggan)}), median Recency {risk.Med_Recency:.0f} hari, "
                 f"Frequency rata-rata {risk.Avg_Frequency:.1f}x, kontribusi revenue {pct(risk.Pct_Revenue)}.",
        'tindakan': 'Kampanye win-back berbatas waktu lewat WhatsApp/email, voucher kecil, dan survei alasan berhenti.',
        'kpi': 'Reactivation rate (kembali bertransaksi dalam 60 hari), repeat purchase rate, churn rate, ROI kampanye.',
        'estimasi': f"Jika 10% pelanggan kembali dengan nilai 1 transaksi rata-rata ({rp(aov)}): tambahan penjualan sekitar {rp(extra)}; "
                    f"biaya voucher 10% plus Rp 1.000 per pelanggan sekitar {rp(cost)}.",
        'asumsi': 'Reactivation 10% dan nilai transaksi sama dengan rata-rata historis segmen. Asumsi, belum teruji.',
        'evaluasi': 'Gunakan grup kontrol (sebagian pelanggan tidak dikirimi kampanye) dan bandingkan reactivation rate serta penjualan setelah 60 hari.'})
    top = p.iloc[0]
    mo = d.groupby('Month')['Net_Sales'].sum()
    pk = mo.idxmax()
    extra3 = top.Revenue * 0.03
    recs.append({
        'judul': f"Pemasaran terarah: reward {top.Segmen} dan promo di bulan puncak", 'icon': '📣', 'warna': '#4cb8d8',
        'bukti': f"Segmen {top.Segmen} berjumlah {pct(top.Pct_Pelanggan)} pelanggan tetapi menyumbang {pct(top.Pct_Revenue)} revenue; "
                 f"bulan puncak penjualan adalah {pk} ({rp(mo.max())}).",
        'tindakan': f"Program reward non-diskon untuk {top.Segmen} dan alihkan anggaran iklan massal ke kalender promo sekitar {pk}.",
        'kpi': 'Retensi segmen teratas, revenue per pelanggan, rasio biaya iklan terhadap penjualan, penjualan bulan puncak.',
        'estimasi': f"Jika program menaikkan belanja {top.Segmen} sebesar 3%: tambahan sekitar {rp(extra3)}; "
                    f"biaya program diasumsikan Rp 25.000 per pelanggan = {rp(top.Jumlah * 25000)}.",
        'asumsi': 'Kenaikan 3% dan biaya per pelanggan adalah asumsi perencanaan, bukan data aktual.',
        'evaluasi': 'Bandingkan pelanggan yang ikut program dengan yang tidak (difference-in-differences), pantau 1 sampai 2 kuartal.'})
    return recs


# ============================================================
# UI: STYLE & KOMPONEN
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input { font-family:'Plus Jakarta Sans',sans-serif !important; }
.stApp { background: radial-gradient(circle at 10% 10%,#dbe7fa 0,transparent 40%), radial-gradient(circle at 90% 90%,#e2dffa 0,transparent 40%), #eef3fb; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: linear-gradient(185deg,#13264a 0%,#1c3d78 55%,#2a5aa8 100%); }
section[data-testid="stSidebar"] * { color:#e3f0f0; }
section[data-testid="stSidebar"] .stRadio label { padding:6px 10px; border-radius:12px; transition:.25s; }
section[data-testid="stSidebar"] .stRadio label:hover { background:rgba(255,255,255,.1); transform:translateX(4px); }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section { background:rgba(255,255,255,.08); border:1px dashed rgba(255,255,255,.35); }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small { color:#b7cdf0; }
section[data-testid="stSidebar"] button { color:#1b2a4a !important; }
@keyframes up { from {opacity:0; transform:translateY(22px)} to {opacity:1; transform:none} }
@keyframes grad { 0%,100%{background-position:0 50%} 50%{background-position:100% 50%} }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(76,184,216,.5)} 50%{box-shadow:0 0 0 10px rgba(76,184,216,0)} }
.hero { background:linear-gradient(125deg,#14305f 0%,#2a5db0 35%,#4f8fdb 70%,#a5cdf2 110%); background-size:260% 260%;
  animation: grad 16s ease infinite, up .8s ease both; color:#fff; border-radius:26px; padding:28px 30px; margin-bottom:18px;
  box-shadow:0 30px 60px -28px rgba(31,79,158,.8); position:relative; overflow:hidden; }
.hero::after { content:''; position:absolute; top:-45%; right:-10%; width:480px; height:480px; border-radius:50%; background:radial-gradient(circle,rgba(255,255,255,.2),transparent 70%); }
.hero .tag { display:inline-block; background:rgba(255,255,255,.2); padding:6px 14px; border-radius:20px; font-size:12px; font-weight:700; margin-bottom:12px; }
.hero h1 { color:#fff !important; font-size:30px; font-weight:800; margin:0 0 4px; padding:0; }
.hero p { margin:0; font-size:14px; opacity:.92; max-width:760px; }
.hero .big { position:absolute; right:30px; top:24px; text-align:right; z-index:1; }
.hero .big b { font-size:42px; font-weight:800; display:block; line-height:1; }
.hero .big span { font-size:12px; opacity:.85; font-weight:600; }
.sec { background:#fff; border:1px solid rgba(61,116,201,.16); border-radius:22px; padding:18px 22px; margin:6px 0 14px;
  box-shadow:0 14px 34px -22px rgba(20,40,90,.5); animation: up .7s ease both; position:relative; transition:.3s; }
.sec:hover { box-shadow:0 24px 50px -24px rgba(61,116,201,.5); }
.sec::before { content:''; position:absolute; top:0; left:22px; right:22px; height:3px; border-radius:0 0 4px 4px; background:linear-gradient(90deg,#3d74c9,#7cb8ec,#ef8a9b); }
.sh { display:flex; align-items:center; gap:12px; }
.si { width:42px; height:42px; border-radius:13px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:19px; flex-shrink:0; }
.st2 { font-weight:800; font-size:16px; color:#1b2a4a; } .ss { font-size:11px; color:#7083a3; }
.chip { display:inline-block; padding:3px 10px; border-radius:14px; font-size:11px; font-weight:700; background:#e3ecfb; color:#27508f; margin-left:auto; }
.kc { background:#fff; border:1px solid rgba(61,116,201,.16); border-radius:18px; padding:16px; display:flex; gap:14px; align-items:center;
  box-shadow:0 10px 26px -18px rgba(20,40,90,.5); transition:.3s; animation: up .7s ease both; height:100%; }
.kc:hover { transform:translateY(-4px); box-shadow:0 22px 40px -18px rgba(61,116,201,.55); }
.kc .ki { width:46px; height:46px; border-radius:14px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:20px; flex-shrink:0; }
.kc .lb { font-size:11px; font-weight:700; color:#7083a3; } .kc .vl { font-size:20px; font-weight:800; color:#1b2a4a; line-height:1.15; }
.qa { background:linear-gradient(135deg,#f3f7fd,#fff); border:1px solid rgba(61,116,201,.16); border-left:5px solid #e3b060; border-radius:16px; padding:12px 16px; margin:10px 0; font-size:13px; color:#44587a; animation: up .7s ease both; }
.qa b { color:#1b2a4a; }
.strat { border-radius:16px; padding:14px 18px; border-left:5px solid; margin-bottom:12px; background:#fff; transition:.3s; font-size:13px; color:#44587a; box-shadow:0 8px 20px -16px rgba(20,40,90,.5); animation: up .7s ease both; }
.strat:hover { transform:translateX(5px); }
.seg { display:inline-block; padding:4px 12px; border-radius:16px; font-size:11px; font-weight:800; color:#fff; margin-right:8px; }
.rc { border-radius:16px; padding:14px 16px; font-size:13px; color:#44587a; height:100%; }
.rc b { display:block; font-size:12px; margin-bottom:4px; }
.lg-card { background:rgba(255,255,255,.85); backdrop-filter:blur(18px); border-radius:28px; padding:30px; box-shadow:0 40px 90px -30px rgba(20,40,90,.45); animation: up .9s ease both; text-align:center; }
.lg-card h2 { background:linear-gradient(90deg,#1f4f9e,#4f8fdb,#8fb8de); -webkit-background-clip:text; background-clip:text; color:transparent; font-weight:800; margin:8px 0 2px; }
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border:0; border-radius:14px; font-weight:700; color:#fff;
  background:linear-gradient(120deg,#1f4f9e,#3d74c9,#5aa9e6); background-size:200% 200%; box-shadow:0 14px 28px -14px rgba(61,116,201,.9); transition:.25s; padding:.55rem 1.3rem; }
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover { transform:translateY(-2px); background-position:100% 0; color:#fff; border:0; }
.stButton > button:active { animation:pulse .6s; }
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid rgba(61,116,201,.16); }
h3 { color:#1b2a4a; font-weight:800 !important; }
.stTabs [data-baseweb="tab"] { font-weight:700; }

/* ===== KONSISTENSI WARNA (paksa tema terang, teks gelap-teal) ===== */
:root { color-scheme: light; }
.stApp { color:#1b2a4a; }
[data-testid="stMain"] p, [data-testid="stMain"] li, [data-testid="stMain"] label, [data-testid="stMain"] span,
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3, [data-testid="stMain"] h4,
[data-testid="stMain"] [data-testid="stCaptionContainer"], [data-testid="stMain"] [data-testid="stWidgetLabel"] * { color:#1b2a4a; }
[data-testid="stMain"] [data-testid="stCaptionContainer"] * { color:#7083a3; }
.stApp .hero, .stApp .hero * { color:#fff !important; }
.stApp .lg-card h2 { color:transparent !important; }
.stApp .si { color:#fff !important; }
.stApp .kc .ki, .stApp .seg, .stApp .stButton > button, .stApp .stDownloadButton > button, .stApp .stFormSubmitButton > button,
.stApp .stButton > button *, .stApp .stDownloadButton > button *, .stApp .stFormSubmitButton > button * { color:#fff !important; }
.stApp .kc .lb { color:#7083a3 !important; } .stApp .kc .vl { color:#1b2a4a !important; }
.stApp .ss { color:#7083a3 !important; } .stApp .st2 { color:#1b2a4a !important; }
.stApp .chip { color:#27508f !important; }
.stApp .qa, .stApp .qa * , .stApp .strat, .stApp .strat *, .stApp .rc, .stApp .rc * { color:#44587a; }
.stApp .qa b, .stApp .strat b { color:#1b2a4a; }
.stApp .rc b { color:inherit; }
[data-testid="stMain"] input, [data-testid="stMain"] textarea, [data-baseweb="select"] > div { background:#fff !important; color:#1b2a4a !important; border-radius:12px !important; }
[data-baseweb="popover"] * { color:#1b2a4a; }
[data-testid="stMetric"] { background:#fff; border:1px solid rgba(61,116,201,.16); border-radius:18px; padding:14px 16px;
  box-shadow:0 10px 26px -18px rgba(20,40,90,.5); transition:.3s; animation: up .7s ease both; border-top:4px solid #4cb8d8; }
[data-testid="stMetric"]:hover { transform:translateY(-4px); }
[data-testid="stMetricValue"] * { font-weight:800; color:#3d74c9 !important; }
[data-testid="stMetricLabel"] * { font-weight:700; color:#7083a3 !important; }
.stAlert { border-radius:16px; }
/* ===== FOTO ===== */
.ava { width:96px; height:96px; border-radius:28px; object-fit:cover; border:4px solid #fff; box-shadow:0 18px 40px -14px rgba(61,116,201,.65); animation:bob 4s ease-in-out infinite; }
.ava-s { width:48px; height:48px; border-radius:15px; object-fit:cover; border:2px solid rgba(255,255,255,.55); flex-shrink:0; }
@keyframes bob { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
.me { display:flex; align-items:center; gap:12px; background:rgba(255,255,255,.85); backdrop-filter:blur(14px); border:1px solid rgba(61,116,201,.16);
  border-radius:20px; padding:8px 16px 8px 8px; width:fit-content; margin:0 0 12px auto; box-shadow:0 10px 26px -18px rgba(20,40,90,.5); animation: up .6s ease both; }
.me img { width:40px; height:40px; border-radius:13px; object-fit:cover; }
.me b { font-size:13px; color:#1b2a4a !important; display:block; line-height:1.2; } .me span { font-size:11px; color:#7083a3 !important; font-weight:600; }
/* ===== STEPPER ===== */
.stepper { display:flex; gap:8px; margin:0 0 16px; flex-wrap:wrap; animation: up .6s ease both; }
.stp { flex:1; min-width:120px; background:#fff; border:1px solid rgba(61,116,201,.16); border-radius:16px; padding:10px 12px;
  display:flex; align-items:center; gap:10px; transition:.3s; box-shadow:0 8px 20px -16px rgba(20,40,90,.5); }
.stp:hover { transform:translateY(-3px); }
.stp .n { width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:800;
  background:#e3ecfb; color:#27508f !important; flex-shrink:0; }
.stp .t { font-size:11px; font-weight:700; color:#44587a !important; line-height:1.2; }
.stp.done .n { background:linear-gradient(135deg,#4cb8d8,#7cb8ec); color:#fff !important; }
.stp.on { background:linear-gradient(120deg,#1f4f9e,#3d74c9,#5aa9e6); border-color:transparent; box-shadow:0 14px 28px -14px rgba(61,116,201,.9); }
.stp.on .n { background:#fff; color:#1f4f9e !important; animation:pulse 2s infinite; } .stp.on .t { color:#fff !important; }
/* ===== PERCANTIK ===== */
.stApp::before { content:''; position:fixed; width:380px; height:380px; border-radius:50%; background:#b4d0f5; filter:blur(95px); opacity:.3; top:-110px; left:-90px; z-index:0; animation: fl 16s ease-in-out infinite; pointer-events:none; }
.stApp::after { content:''; position:fixed; width:340px; height:340px; border-radius:50%; background:#d9d3f7; filter:blur(95px); opacity:.28; bottom:-100px; right:-80px; z-index:0; animation: fl 18s ease-in-out infinite reverse; pointer-events:none; }
@keyframes fl { 0%,100%{transform:translate(0,0)} 50%{transform:translate(34px,-28px)} }
.sec .st2 { background:linear-gradient(90deg,#1b2a4a,#3d74c9); -webkit-background-clip:text; background-clip:text; color:transparent !important; }
.kc .vl { background:linear-gradient(90deg,#1b2a4a,#3d74c9); -webkit-background-clip:text; background-clip:text; color:transparent !important; }
.stApp .kc .vl { color:transparent !important; }
.hero .tag { backdrop-filter:blur(8px); border:1px solid rgba(255,255,255,.3); }
.stApp [data-testid="stMain"] hr { border-color:rgba(61,116,201,.15); }
::-webkit-scrollbar { width:9px; height:9px; } ::-webkit-scrollbar-thumb { background:#b4d0f5; border-radius:8px; }

/* ===== UPLOADER SIDEBAR (lebih kentara) ===== */
.up-title { display:flex; align-items:center; gap:8px; font-size:13px; font-weight:800; color:#fff; letter-spacing:.3px; margin:6px 0 6px;
  background:linear-gradient(90deg,rgba(124,184,236,.35),rgba(255,255,255,.05)); border-left:4px solid #8fe0f0; border-radius:10px; padding:8px 12px; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label p { font-size:13px !important; font-weight:800 !important; color:#fff !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section { border:2px dashed #8fe0f0 !important; background:rgba(143,224,240,.12) !important; border-radius:16px;
  animation:pulse 2.6s infinite; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section * { color:#e3f0ff !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section button { background:#fff !important; color:#1f4f9e !important; font-weight:800; border-radius:10px; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] { background:#fff; border-radius:12px; padding:4px 8px; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * { color:#1b2a4a !important; font-weight:700; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg { color:#3d74c9 !important; }
/* ===== BERANDA ===== */
.land { display:grid; grid-template-columns:1.1fr 1fr; gap:26px; align-items:center; background:linear-gradient(125deg,#14305f 0%,#2a5db0 40%,#4f8fdb 75%,#a5cdf2 115%);
  background-size:260% 260%; animation: grad 16s ease infinite, up .9s ease both; border-radius:32px; padding:38px; position:relative; overflow:hidden;
  box-shadow:0 40px 80px -34px rgba(31,79,158,.9); margin-bottom:18px; }
.land::before { content:''; position:absolute; width:420px; height:420px; border-radius:50%; background:radial-gradient(circle,rgba(255,255,255,.22),transparent 70%); top:-160px; left:-120px; }
.land::after { content:''; position:absolute; width:340px; height:340px; border-radius:50%; background:radial-gradient(circle,rgba(123,140,240,.5),transparent 70%); bottom:-150px; right:-60px; }
.land > div { position:relative; z-index:1; }
.stApp .land, .stApp .land * { color:#fff !important; }
.land .tag { display:inline-block; background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35); padding:7px 15px; border-radius:20px; font-size:12px; font-weight:700; margin-bottom:16px; backdrop-filter:blur(8px); }
.land h1 { font-size:40px; line-height:1.1; font-weight:800; letter-spacing:-1px; margin:0 0 14px; padding:0; }
.stApp .land h1 .grad { color:#bfe3ff !important; }
.land p { font-size:15px; opacity:.93; margin:0 0 18px; max-width:520px; }
.land .pills { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:20px; }
.land .pills span { background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25); padding:6px 13px; border-radius:14px; font-size:11px; font-weight:700; transition:.25s; }
.land .pills span:hover { background:rgba(255,255,255,.3); transform:translateY(-3px); }
.land .who { display:inline-flex; align-items:center; gap:12px; background:rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.28); border-radius:18px; padding:8px 18px 8px 8px; backdrop-filter:blur(8px); }
.land .who img { width:42px; height:42px; border-radius:13px; object-fit:cover; border:2px solid rgba(255,255,255,.7); }
.land .who b { font-size:13px; display:block; line-height:1.2; } .land .who span { font-size:11px; opacity:.85; }
.land .frame { position:relative; border-radius:30px; padding:10px; background:rgba(255,255,255,.2); border:1px solid rgba(255,255,255,.4); backdrop-filter:blur(10px);
  box-shadow:0 30px 60px -24px rgba(10,30,70,.8); animation:bob 6s ease-in-out infinite; }
.land .frame img { width:100%; height:300px; object-fit:cover; border-radius:22px; display:block; }
.land .float { position:absolute; background:#fff; border-radius:16px; padding:8px 14px; font-size:12px; font-weight:800; box-shadow:0 14px 30px -12px rgba(10,30,70,.6); animation:bob 5s ease-in-out infinite; }
.stApp .land .float { color:#1f4f9e !important; }
.land .f1 { top:-14px; left:-14px; } .land .f2 { bottom:30px; right:-16px; animation-delay:-2s; } .land .f3 { bottom:-14px; left:30px; animation-delay:-3.5s; }
.stepsrow { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:10px 0 18px; }
.stepc { background:#fff; border:1px solid rgba(61,116,201,.16); border-radius:22px; padding:18px; position:relative; overflow:hidden; transition:.3s; animation: up .8s ease both;
  box-shadow:0 14px 34px -22px rgba(20,40,90,.5); }
.stepc:hover { transform:translateY(-6px); box-shadow:0 26px 50px -24px rgba(61,116,201,.6); }
.stepc .no { position:absolute; right:14px; top:2px; font-size:60px; font-weight:800; color:rgba(61,116,201,.09) !important; }
.stepc .ic { width:46px; height:46px; border-radius:15px; display:flex; align-items:center; justify-content:center; font-size:22px; margin-bottom:10px; }
.stepc b { font-size:14px; color:#1b2a4a !important; display:block; margin-bottom:4px; } .stepc span { font-size:12px; color:#44587a !important; }
.feat { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:12px; }
.fc { background:linear-gradient(145deg,#fff,#f3f7fd); border:1px solid rgba(61,116,201,.16); border-radius:20px; padding:16px; transition:.3s; animation: up .9s ease both; }
.fc:hover { transform:translateY(-5px) rotate(-.4deg); box-shadow:0 22px 44px -22px rgba(61,116,201,.55); }
.fc .top { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.fc .e { font-size:24px; } .fc .w { font-size:11px; font-weight:800; background:#e3ecfb; color:#27508f !important; padding:3px 10px; border-radius:12px; }
.fc b { font-size:13px; color:#1b2a4a !important; display:block; } .fc span { font-size:11.5px; color:#44587a !important; }
.drop-h { text-align:center; margin:6px 0 8px; } .drop-h b { font-size:20px; color:#1b2a4a !important; font-weight:800; display:block; } .drop-h span { font-size:12px; color:#7083a3 !important; }
[data-testid="stMain"] [data-testid="stFileUploader"] section { border:2.5px dashed #3d74c9 !important; border-radius:24px; padding:26px;
  background:linear-gradient(135deg,rgba(61,116,201,.1),rgba(123,140,240,.1)) !important; transition:.3s; }
[data-testid="stMain"] [data-testid="stFileUploader"] section:hover { transform:scale(1.01); border-color:#7b8cf0 !important; }
[data-testid="stMain"] [data-testid="stFileUploader"] section * { color:#27508f !important; font-weight:600; }
[data-testid="stMain"] [data-testid="stFileUploader"] section button { background:linear-gradient(120deg,#1f4f9e,#3d74c9,#5aa9e6) !important; color:#fff !important; border:0; border-radius:12px; font-weight:800; }
[data-testid="stMain"] [data-testid="stFileUploader"] section button * { color:#fff !important; }
@media (max-width:900px) { .land, .stepsrow, .feat { grid-template-columns:1fr; } .land h1 { font-size:30px; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hero(tag, title, sub, big=None, big_label=''):
    b = f'<div class="big"><b>{big}</b><span>{big_label}</span></div>' if big is not None else ''
    st.markdown(f'<div class="hero"><span class="tag">{tag}</span><h1>{title}</h1><p>{sub}</p>{b}</div>', unsafe_allow_html=True)


def sec(icon, grad, title, sub='', task=''):
    chip = f'<span class="chip">{task}</span>' if task else ''
    st.markdown(f'<div class="sec"><div class="sh"><div class="si" style="background:{grad}">{icon}</div>'
                f'<div><div class="st2">{title}</div><div class="ss">{sub}</div></div>{chip}</div></div>', unsafe_allow_html=True)


G1 = 'linear-gradient(135deg,#3d74c9,#4f8fdb)'
G2 = 'linear-gradient(135deg,#4cb8d8,#7b8cf0)'
G3 = 'linear-gradient(135deg,#ef8a9b,#e3b060)'
G4 = 'linear-gradient(135deg,#3d74c9,#7cb8ec)'


def kpis(items):
    cols = st.columns(len(items))
    for col, (icon, label, value, grad) in zip(cols, items):
        col.markdown(f'<div class="kc"><div class="ki" style="background:{grad}">{icon}</div>'
                     f'<div><div class="lb">{label}</div><div class="vl">{value}</div></div></div>', unsafe_allow_html=True)
    st.write('')


def note(html, color='#e3b060'):
    st.markdown(f'<div class="qa" style="border-left-color:{color}">{html}</div>', unsafe_allow_html=True)


def show_df(df, height=None):
    kw = dict(STRETCH)
    if height:
        kw['height'] = height
    st.dataframe(df, hide_index=True, **kw)


def show_tb(tb):
    show_df(pd.DataFrame(tb[1], columns=tb[0]))


def style_fig(fig, h=360, legend=True):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Plus Jakarta Sans', color='#44587a'), showlegend=legend,
                      legend=dict(orientation='h', y=-0.18), colorway=PAL, transition=dict(duration=600))
    fig.update_xaxes(gridcolor='rgba(61,116,201,.08)')
    fig.update_yaxes(gridcolor='rgba(61,116,201,.08)')
    return fig


def plot(fig, h=360, legend=True):
    st.plotly_chart(style_fig(fig, h, legend), **STRETCH, config={'displayModeBar': False})


def csv_btn(df, name, label, key):
    st.download_button('⬇ ' + label, df.to_csv(index=False).encode('utf-8-sig'), file_name=name, mime='text/csv', key=key)


def gate(msg, target, btn):
    st.markdown(f'<div class="sec" style="text-align:center"><p style="font-size:14px;color:#44587a">{msg}</p></div>', unsafe_allow_html=True)
    if st.button('← ' + btn, key='gate_' + target):
        S['page'] = target
        st.rerun()


def next_btn(target, label):
    st.write('')
    c = st.columns([1, 2, 1])[1]
    if c.button('➜ ' + label, key='next_' + target, **STRETCH):
        S['page'] = target
        st.rerun()


# ============================================================
# LOGIN
# ============================================================
def login_page():
    st.write('')
    _, c, _ = st.columns([1, 1.2, 1])
    with c:
        st.markdown(f'<div class="lg-card"><img class="ava" src="{photo_uri()}" alt="foto"><h2>Selamat Datang</h2>'
                    '<div style="font-size:13px;color:#7083a3">Adelia Mutiara · 51230004</div>'
                    '<div style="font-size:12px;color:#44587a;margin-top:10px">UTS Integrasi Data Mining &amp; Business Intelligence<br>'
                    'EDA · Preprocessing · Apriori · K-Means · Dashboard · Rekomendasi</div></div>', unsafe_allow_html=True)
        with st.form('login'):
            u = st.text_input('Username', placeholder='Username')
            p = st.text_input('Password', type='password', placeholder='Password')
            ok = st.form_submit_button('Masuk', **STRETCH)
        if ok:
            if u.strip() == USERNAME_VALID and p.strip() == PASSWORD_VALID:
                S['logged_in'] = True
                st.rerun()
            else:
                st.error('Username atau password salah.')
        st.caption('Login: adel / adelia123')
        if _find_photo() is None and not S.get('photo_bytes'):
            st.caption('Foto belum ditemukan. Simpan file foto (misal adelll.jpg) di folder yang sama dengan skrip ini, atau unggah di sini:')
            photo_uploader('📷 Unggah foto', 'photo_login')


# ============================================================
# HALAMAN
# ============================================================
QUESTIONS = [
    ("Produk apa yang sering dibeli bersama dalam satu transaksi?", "Apriori / Market Basket (support, confidence, lift)"),
    ("Segmen pelanggan mana yang paling bernilai dan mana yang berisiko tidak kembali?", "RFM + K-Means (profil klaster)"),
    ("Kapan penjualan meningkat dan kategori/produk apa yang mendorongnya?", "Tren bulanan dan Top 10 di Dashboard BI"),
    ("Seberapa besar masalah kualitas data dan bagaimana dampaknya pada hasil analisis?", "Audit EDA dan preprocessing"),
    ("Tindakan pemasaran apa yang paling layak diprioritaskan manajemen?", "Rekomendasi keputusan berbasis bukti data"),
]


def page_upload_hint():
    st.markdown(f"""<div class="land"><div>
<span class="tag">✨ UTS · Data Mining &amp; Business Intelligence</span>
<h1>Ubah data transaksi menjadi <span class="grad">keputusan bisnis</span></h1>
<p>Temukan produk yang sering dibeli bersama, kenali pelanggan paling bernilai, dan dapatkan rekomendasi tindakan berbasis data, dalam satu alur yang rapi.</p>
<div class="pills"><span>🔍 EDA</span><span>🧹 Preprocessing</span><span>🔗 Apriori</span><span>🎯 K-Means</span><span>📊 Dashboard</span><span>💡 Rekomendasi</span></div>
<div class="who"><img src="{photo_uri()}" alt="foto"><div><b>Adelia Mutiara</b><span>NPM 51230004</span></div></div>
</div><div><div class="frame"><img src="{banner_uri()}" alt="banner">
<div class="float f1">🔗 Apriori</div><div class="float f2">🎯 K-Means</div><div class="float f3">📊 Dashboard BI</div></div></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="drop-h"><b>⬆️ Mulai dari sini: unggah file Excel UTS</b><span>Format .xlsx / .xls · hanya sheet Data_Raw yang dibaca, sheet lain diabaikan</span></div>', unsafe_allow_html=True)
    f = st.file_uploader('Seret file ke sini atau klik Browse', type=['xlsx', 'xls'], key='main_up', label_visibility='collapsed')
    handle_upload(f)

    st.markdown("""<div class="stepsrow">
<div class="stepc"><div class="no">1</div><div class="ic" style="background:linear-gradient(135deg,#3d74c9,#5aa9e6);color:#fff">📥</div><b>Unggah data</b><span>Pilih file Excel berisi sheet Data_Raw, audit kualitas data langsung berjalan.</span></div>
<div class="stepc"><div class="no">2</div><div class="ic" style="background:linear-gradient(135deg,#7b8cf0,#a5b4fa);color:#fff">⚙️</div><b>Jalankan tiap tahap</b><span>Preprocessing, Apriori, dan K-Means dengan satu klik per tahap.</span></div>
<div class="stepc"><div class="no">3</div><div class="ic" style="background:linear-gradient(135deg,#4cb8d8,#7fd3e8);color:#fff">🎯</div><b>Baca rekomendasi</b><span>Dashboard interaktif dan 3 rekomendasi prioritas siap dipresentasikan.</span></div></div>
<div class="feat">
<div class="fc"><div class="top"><span class="e">🔍</span><span class="w">10%</span></div><b>Business Understanding &amp; EDA</b><span>Pertanyaan bisnis, profil data, dan audit kualitas.</span></div>
<div class="fc"><div class="top"><span class="e">🧹</span><span class="w">25%</span></div><b>Data Preprocessing</b><span>Missing, duplikat, teks tidak konsisten, dan outlier.</span></div>
<div class="fc"><div class="top"><span class="e">🔗</span><span class="w">20%</span></div><b>Apriori / Market Basket</b><span>Support, confidence, lift, dan network graph aturan.</span></div>
<div class="fc"><div class="top"><span class="e">🎯</span><span class="w">20%</span></div><b>K-Means Segmentasi</b><span>RFM, Elbow, Silhouette, dan strategi per segmen.</span></div>
<div class="fc"><div class="top"><span class="e">📊</span><span class="w">15%</span></div><b>Dashboard BI</b><span>KPI, tren bulanan, top produk, dan filter interaktif.</span></div>
<div class="fc"><div class="top"><span class="e">💡</span><span class="w">10%</span></div><b>Rekomendasi Keputusan</b><span>Bukti, KPI, estimasi biaya/manfaat, dan cara evaluasi.</span></div></div>""", unsafe_allow_html=True)


def page_eda():
    au = S['audit']
    if not au.get('success'):
        gate(au.get('error') or 'Data belum bisa diaudit.', 'eda', 'Unggah ulang di panel kiri')
        return
    hero('💡 Tahap 1 · Business Understanding & EDA · 10%', 'Memahami bisnis dan kondisi data',
         'Audit dilakukan pada salinan Data_Raw. Data asli tidak diubah.', fi(au['bad_rows']),
         f"baris bermasalah ({pct(au['bad_rows'] / au['rows'])})")
    kpis([('🧾', 'Baris', fi(au['rows']), G1), ('🗂️', 'Kolom', fi(au['cols']), G2),
          ('🧮', 'Transaksi unik', fi(au['n_tx']), G3), ('👥', 'Pelanggan unik', fi(au['n_cust']), G4)])

    sec('❓', G1, 'Pertanyaan bisnis', 'Apa yang ingin dijawab manajemen toko', 'Tugas 1')
    cols = st.columns(2)
    for i, (q, t) in enumerate(QUESTIONS):
        cols[i % 2].markdown(f'<div class="qa" style="border-left-color:#3d74c9"><b>Q{i + 1}. {q}</b><br>'
                             f'<span style="font-size:11px;color:#7083a3">Dijawab lewat: {t}</span></div>', unsafe_allow_html=True)

    sec('📋', G2, 'Profil dataset', 'Baris, kolom, rentang tanggal, transaksi, pelanggan, produk, total penjualan sementara, dan tipe data', 'Tugas 2')
    c1, c2, c3 = st.columns(3)
    c1.metric('Rentang tanggal', f"{au['date_min']:%Y-%m-%d} s.d. {au['date_max']:%Y-%m-%d}", f"{au['date_invalid']} tanggal tidak terbaca", delta_color='off')
    c2.metric('Total penjualan sementara', rp(au['sales_prov']))
    c3.metric('Produk unik (setelah seragam)', fi(au['n_prod']))
    st.markdown('**Tipe data per kolom**')
    show_df(au['dtypes'])

    sec('🔍', G3, 'Audit kualitas data', 'Missing values, duplikasi, inkonsistensi kategori, dan outlier', 'Tugas 3')
    kpis([('🕳️', 'Sel kosong', fi(au['missing_total']), 'linear-gradient(135deg,#d99a4a,#e3b060)'),
          ('♊', 'Duplikat identik', fi(au['dup_exact']), G3),
          ('🔁', 'Produk dobel / transaksi', fi(au['dup_line']), G4),
          ('⚠️', 'Baris bermasalah', fi(au['bad_rows']), 'linear-gradient(135deg,#4f8fdb,#7b8cf0)')])
    a, b = st.columns(2)
    with a:
        st.markdown('**Missing per kolom**')
        if au['missing']:
            show_df(pd.DataFrame([{'Kolom': c, 'Kosong': n, 'Persen': pct(n / au['rows'])} for c, n in au['missing'].items()]))
        else:
            st.success('Tidak ada missing value')
    with b:
        st.markdown('**Inkonsistensi teks**')
        show_df(au['incons'])
    st.markdown('**Outlier dan nilai tidak wajar (aturan IQR 1,5)**')
    show_df(au['outliers'].round(0))

    st.markdown('### Visualisasi EDA')
    v1, v2 = st.columns(2)
    with v1:
        st.markdown('**Visualisasi 1 · Tren penjualan bulanan (data mentah)**')
        fig = go.Figure(go.Scatter(x=au['monthly'][0], y=au['monthly'][1], mode='lines+markers', fill='tozeroy',
                                   line=dict(color='#3d74c9', width=3), fillcolor='rgba(61,116,201,.15)', line_shape='spline'))
        plot(fig, legend=False)
    with v2:
        st.markdown('**Visualisasi 2 · Varian penulisan kategori (tanda inkonsistensi)**')
        plot(go.Figure(go.Bar(x=au['cat_raw'][0], y=au['cat_raw'][1], marker_color=PAL)), legend=False)
    v3, v4 = st.columns(2)
    with v3:
        st.markdown('**Visualisasi 3 · Missing value per kolom**')
        if au['missing']:
            plot(go.Figure(go.Bar(x=list(au['missing'].keys()), y=list(au['missing'].values()), marker_color='#ef8a9b')), legend=False)
        else:
            st.info('Tidak ada missing value.')
    with v4:
        st.markdown('**Visualisasi 4 · Outlier dan nilai tidak valid**')
        o = au['outliers']
        fig = go.Figure([go.Bar(name='Outlier', x=o['Variabel'], y=o['Outlier'], marker_color='#e3b060'),
                         go.Bar(name='Tidak valid', x=o['Variabel'], y=o['Nilai <= 0 / bukan angka'], marker_color='#3d74c9')])
        plot(fig)
    note('<b>✨ Insight awal</b><ul style="margin:6px 0 0 18px">' + ''.join(f'<li>{i}</li>' for i in au['insights']) + '</ul>', '#4cb8d8')
    next_btn('prep', 'Lanjut ke Preprocessing')


def page_prep():
    cl = S['clean']
    if not cl.get('success'):
        sec('🧹', G4, 'Data Preprocessing', 'Membuat salinan kerja dari Data_Raw lalu menangani missing value, duplikasi, format tanggal, '
            'teks tidak konsisten, qty/harga tidak valid, dan total tidak wajar.')
        if st.button('▶ Jalankan Preprocessing', key='run_prep'):
            with st.spinner('Membersihkan data...'):
                r = run_cleaning()
            if r.get('success'):
                st.toast(f"Preprocessing selesai: {fi(r['rows_after'])} baris bersih", icon='✅')
                st.rerun()
            else:
                st.error(r.get('error'))
        elif cl.get('error'):
            st.error(cl['error'])
        return
    hero('🧹 Tahap 2 · Data Preprocessing · 25%', 'Data_Clean siap dipakai',
         'Tugas 4 dan 5: salinan kerja, keputusan tiap masalah, dan jumlah baris sebelum/sesudah.',
         pct(cl['rows_after'] / cl['rows_before']), 'baris bertahan')
    kpis([('📥', 'Baris sebelum', fi(cl['rows_before']), G1), ('📤', 'Baris sesudah', fi(cl['rows_after']), G2),
          ('🗑️', 'Baris dihapus', fi(cl['rows_before'] - cl['rows_after']), G3),
          ('💰', 'Total penjualan bersih', rp(cl['df']['Net_Sales'].sum()), G4)])
    sec('✅', G1, 'Keputusan penanganan tiap masalah', 'Hapus, perbaiki, imputasi, atau tandai, lengkap dengan jumlah baris sebelum dan sesudah', 'Tugas 5')
    show_df(cl['log'].assign(**{'#': range(1, len(cl['log']) + 1)})[['#', 'Masalah', 'Keputusan', 'Jenis', 'Terdampak', 'Baris sebelum', 'Baris sesudah']])
    sec('⚖️', G2, 'Ringkasan kualitas sebelum vs sesudah', 'Data_Raw dibanding Data_Clean')
    q = cl['quality'].copy()
    for c in q.columns[1:]:
        q[c] = q[c].map(fi)
    show_df(q)
    note('<b>Catatan aturan preprocessing (reproducible)</b><br>Aturan dijalankan berurutan seperti tabel di atas dan selalu menghasilkan hasil yang sama '
         'dari Data_Raw yang sama. Net_Sales = Qty x UnitPrice x (1 - Diskon). Baris dengan ID pelanggan kosong dipertahankan untuk penjualan dan Apriori '
         'tetapi tidak ikut RFM. Baris <code>Flag_Outlier</code> tidak dihapus agar bisa diperiksa manual.', '#3d74c9')
    sec('✨', G3, 'Data_Clean', '15 baris pertama; data lengkap dapat diunduh')
    show_df(cl['df'].head(15))
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        csv_btn(cl['df'], 'Data_Clean.csv', 'Data_Clean (CSV)', 'dl_clean')
    with c2:
        csv_btn(cl['log'], 'dokumentasi_preprocessing.csv', 'Dokumentasi (CSV)', 'dl_log')
    next_btn('features', 'Lanjut ke Basket, RFM & Scaling')


def page_features():
    cl, ft = S['clean'], S['feat']
    if not cl.get('success'):
        gate('Lakukan Preprocessing terlebih dahulu.', 'prep', 'Ke Preprocessing')
        return
    if not ft.get('success'):
        sec('🧺', G4, 'Basket, RFM & Scaling', 'Membentuk basket_size per Transaction_ID, matriks transaksi x produk untuk Apriori, '
            'tabel RFM per pelanggan, serta log transform dan standardisasi.')
        if st.button('▶ Bentuk Fitur', key='run_feat'):
            with st.spinner('Membentuk basket dan RFM...'):
                r = run_features()
            if r.get('success'):
                st.toast('Fitur terbentuk', icon='✅')
                st.rerun()
            else:
                st.error(r.get('error'))
        elif ft.get('error'):
            st.error(ft['error'])
        return
    hero('🧺 Tahap 2 · Tugas 6 sampai 8', 'Basket, matriks, dan RFM',
         'Data agregat terpisah: transaksi-basket untuk Apriori dan pelanggan-RFM untuk K-Means.',
         fmt_id(ft['avg_basket'], 2), 'rata-rata basket_size')
    kpis([('🧾', 'Transaksi', fi(len(ft['tx'])), G1), ('📦', 'Produk (kolom matriks)', fi(ft['matrix'].shape[1]), G2),
          ('👥', 'Pelanggan di RFM', fi(len(ft['rfm'])), G3), ('📅', 'Tanggal acuan', f"{ft['ref']:%d %b %Y}", G4)])

    sec('🛒', G1, 'basket_size per Transaction_ID', 'basket_size = jumlah produk unik dalam satu transaksi', 'Tugas 6')
    a, b = st.columns(2)
    with a:
        show_df(ft['tx'].head(15))
    with b:
        x, y = ft['basket_dist']
        plot(go.Figure(go.Bar(x=[f'{i} produk' for i in x], y=y, marker_color=PAL)), 330, False)
    st.markdown(f"**Tabel transaksi-produk untuk Apriori (1 = dibeli, 0 = tidak)** · 10 x 10 pertama dari {fi(ft['matrix'].shape[0])} x {fi(ft['matrix'].shape[1])}")
    show_df(ft['matrix'].iloc[:10, :10].reset_index())
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        csv_btn(ft['tx'], 'basket_size.csv', 'basket_size (CSV)', 'dl_basket')
    with c2:
        csv_btn(ft['matrix'].reset_index(), 'matriks_transaksi_produk.csv', 'Matriks (CSV)', 'dl_matrix')

    sec('📈', G2, 'RFM per pelanggan', 'Recency, Frequency (transaksi unik), Monetary (penjualan bersih), plus log dan Z-score', 'Tugas 7')
    note(f"<b>Tanggal acuan: {ft['ref']:%d %B %Y}</b><br>Ditetapkan satu hari setelah transaksi terakhir pada Data_Clean "
         f"({(ft['ref'] - pd.Timedelta(days=1)):%d %B %Y}). Recency = tanggal acuan dikurangi tanggal transaksi terakhir pelanggan. "
         f"{fi(ft['n_unknown'])} transaksi dengan pelanggan UNKNOWN tidak ikut RFM.", '#4cb8d8')
    show_df(ft['rfm'].head(20))
    csv_btn(ft['rfm'].assign(Last_Transaction=ft['rfm']['Last_Transaction'].dt.strftime('%Y-%m-%d')), 'tabel_rfm.csv', 'Tabel RFM (CSV)', 'dl_rfm')

    sec('🪄', G3, 'Persiapan fitur K-Means', 'Log transform lalu standardisasi (mean 0, std 1)', 'Tugas 8')
    cols = st.columns(4)
    for c, (k, v) in zip(cols, ft['skew'].items()):
        c.metric(f'Skewness {k}', fmt_id(v, 2))
    a, b = st.columns(2)
    a.markdown('<div class="qa" style="border-left-color:#e3b060"><b>Mengapa distandardisasi?</b><br>K-Means memakai jarak Euclidean. Recency (hari), Frequency (kali), '
               'dan Monetary (rupiah) berbeda skala sehingga Monetary akan mendominasi. Frequency dan Monetary dilog-transform (LN(x+1)) karena miring ke kanan, '
               'lalu ketiga fitur distandardisasi agar kontribusinya seimbang.</div>', unsafe_allow_html=True)
    b.markdown(f'<div class="qa" style="border-left-color:#7b8cf0"><b>Strategi untuk data besar</b><br><b>Agregasi:</b> {fi(cl["rows_after"])} baris item diringkas menjadi '
               f'{fi(len(ft["tx"]))} transaksi lalu {fi(len(ft["rfm"]))} pelanggan. <b>Sampling:</b> Silhouette dihitung pada sampel hingga 5.000 titik. '
               f'<b>Chunking:</b> bila data jauh lebih besar, file dibaca per potongan (chunksize) dan diagregasi bertahap.</div>', unsafe_allow_html=True)
    next_btn('apriori', 'Lanjut ke Apriori')


def page_apriori():
    ft, ap = S['feat'], S['apr']
    if not ft.get('success'):
        gate('Lakukan Basket, RFM & Scaling terlebih dahulu.', 'features', 'Ke Basket & RFM')
        return
    hero('🔗 Tahap 3 · Apriori / Market Basket Analysis · 20%', 'Pola kombinasi produk',
         f"Matriks {fi(len(ft['tx']))} transaksi x {fi(ft['matrix'].shape[1])} produk. Satu transaksi dihitung satu kali, bukan per baris item.")
    with st.form('apr_form'):
        c1, c2, c3 = st.columns([1, 1, 1])
        sup = c1.number_input('Minimum support', 0.0005, 1.0, float(ap.get('min_sup', 0.01)), 0.001, format='%.4f')
        conf = c2.number_input('Minimum confidence', 0.01, 1.0, float(ap.get('min_conf', 0.2)), 0.01)
        c3.write('')
        go_ = c3.form_submit_button('▶ Jalankan Apriori', **STRETCH)
    if go_:
        with st.spinner('Mencari frequent itemsets dan aturan...'):
            r = run_apriori(sup, conf)
        if r.get('success'):
            st.toast(f"Apriori selesai: {r['n_rules']} aturan lift > 1", icon='✅')
            ap = r
        else:
            st.error(r.get('error'))
    sec('🎚️', G1, 'Alasan pemilihan parameter', 'Minimum support dan confidence', 'Tugas 10')
    a, b = st.columns(2)
    a.markdown(f'<div class="qa" style="border-left-color:#e3b060"><b>Minimum support (default 1%)</b><br>Rata-rata basket kecil dan katalog produk lebar, sehingga kombinasi produk jarang muncul. '
               f'Support 1% berarti itemset minimal ada di sekitar {fi(len(ft["tx"]) * 0.01)} transaksi: cukup banyak untuk tidak kebetulan, tetapi cukup rendah untuk menangkap pola nyata.</div>', unsafe_allow_html=True)
    b.markdown('<div class="qa" style="border-left-color:#4cb8d8"><b>Minimum confidence (default 20%)</b><br>Minimal 20% transaksi yang memuat antecedent juga memuat consequent. '
               'Nilai ini dipadukan dengan lift &gt; 1 agar aturan tidak sekadar mencerminkan produk yang memang populer.</div>', unsafe_allow_html=True)
    if not ap.get('success'):
        if ap.get('error'):
            st.error(ap['error'])
        else:
            st.info('Atur parameter lalu klik **Jalankan Apriori**.')
        return
    kpis([('🧾', 'Transaksi', fi(ap['n_tx']), G1), ('📦', 'Frequent itemset', fi(ap['n_freq']), G2),
          ('🔗', 'Aturan (lift > 1)', fi(ap['n_rules']), G3), ('🔎', 'Min. transaksi', fi(ap['min_count']), 'linear-gradient(135deg,#d99a4a,#e3b060)')])
    if not ap['top']:
        st.warning('Tidak ada aturan dengan lift > 1 pada parameter ini. Turunkan minimum support (misalnya 0,005) atau confidence lalu jalankan ulang.')
        return
    sec('🏆', G3, 'Top 10 aturan asosiasi', 'Support, confidence, dan lift', 'Tugas 11')
    show_df(ap['top_df'])
    note('<b>Kriteria pemeringkatan</b><br>Hanya aturan dengan lift &gt; 1 (asosiasi positif). Diurutkan berdasarkan <b>lift</b> tertinggi, lalu <b>confidence</b>, lalu <b>support</b>. '
         'Lift menunjukkan seberapa kuat kombinasi melebihi kebetulan, confidence menunjukkan keandalan aturan, dan support memastikan pola cukup sering muncul.', '#3d74c9')
    csv_btn(ap['rules_df'], 'aturan_asosiasi.csv', 'Semua aturan (CSV)', 'dl_rules')

    sec('🕸️', G2, 'Visualisasi aturan', 'Network graph (panah dari antecedent ke consequent, tebal = lift) dan scatter support vs confidence', 'Tugas 12')
    a, b = st.columns(2)
    with a:
        if ap['network'] is not None:
            st.plotly_chart(ap['network'], **STRETCH, config={'displayModeBar': False})
    with b:
        sc = ap['scatter']
        fig = px.scatter(sc, x='Support', y='Confidence', size='Lift', color='Lift', hover_name='Aturan',
                         color_continuous_scale=['#a5cdf2', '#3d74c9', '#1b2a4a'], size_max=26)
        plot(fig, 460, False)

    sec('💡', G4, 'Rekomendasi cross-selling, bundling, dan penempatan rak', '3 rekomendasi berdasarkan aturan teratas', 'Tugas 12')
    acts = ['Buat paket bundling {a} dengan {c}.',
            'Tampilkan {c} sebagai rekomendasi "sering dibeli bersama" saat {a} dipilih (cross-selling).',
            'Letakkan {a} dan {c} pada rak yang berdekatan atau satu lorong.']
    for i, r in enumerate(ap['top'][:3]):
        a_, c_ = ' + '.join(r['ant']), ' + '.join(r['con'])
        st.markdown(f'<div class="strat" style="border-color:{["#3d74c9", "#4cb8d8", "#ef8a9b"][i]}"><b style="font-size:14px;color:#1b2a4a">{i + 1}. {a_} → {c_}</b><br>'
                    f'<b>Bukti:</b> support {pct(r["support"], 2)}, confidence {pct(r["confidence"])}, lift {fmt_id(r["lift"], 2)} ({fi(r["n"])} transaksi).<br>'
                    f'<b>Aksi:</b> {acts[i].format(a=a_, c=c_)}</div>', unsafe_allow_html=True)
    sec('📑', G1, 'Frequent itemsets teratas', 'Itemset ukuran 2 dan 3 dengan support tertinggi')
    show_df(ap['fs_df'])
    next_btn('kmeans', 'Lanjut ke K-Means')


def page_kmeans():
    ft, km = S['feat'], S['km']
    if not ft.get('success'):
        gate('Lakukan Basket, RFM & Scaling terlebih dahulu.', 'features', 'Ke Basket & RFM')
        return
    if not km.get('success'):
        sec('🎯', G4, 'K-Means Segmentasi Pelanggan', 'Menguji k = 2 sampai 8 dengan Elbow Method dan Silhouette Score, memilih k, memprofilkan klaster, dan memberi label berbasis data.')
        if st.button('▶ Jalankan K-Means', key='run_km'):
            with st.spinner('Menguji kandidat k...'):
                r = run_kmeans()
            if r.get('success'):
                st.toast(f"K-Means selesai dengan k = {r['k']}", icon='✅')
                st.rerun()
            else:
                st.error(r.get('error'))
        elif km.get('error'):
            st.error(km['error'])
        return
    hero('🎯 Tahap 4 · K-Means / Segmentasi · 20%', 'Segmentasi pelanggan',
         f"Satu baris mewakili satu pelanggan ({fi(len(ft['rfm']))} pelanggan), fitur: Recency, Log_Frequency, Log_Monetary yang distandardisasi.",
         km['k'], 'k terpilih')
    sec('📉', G1, 'Elbow Method dan Silhouette Score', f"Kandidat k = 2 sampai {km['ks'][-1]}", 'Tugas 14')
    a, b = st.columns(2)
    with a:
        t = pd.DataFrame({'k': km['ks'], 'Inertia / WCSS': [round(v, 2) for v in km['inertia']], 'Silhouette': [round(v, 4) for v in km['sil']]})
        t['Catatan'] = [' '.join(x for x in [('⭐ dipakai' if k == km['k'] else ''), ('elbow' if k == km['elbow_k'] else ''),
                                             ('silhouette terbaik' if k == km['rec_k'] else '')] if x) for k in km['ks']]
        show_df(t)
    with b:
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Scatter(x=km['ks'], y=km['inertia'], name='Inertia', mode='lines+markers', line=dict(color='#3d74c9', width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=km['ks'], y=km['sil'], name='Silhouette', mode='lines+markers', line=dict(color='#4cb8d8', width=3)), secondary_y=True)
        fig.add_vline(x=km['k'], line_dash='dash', line_color='#ef8a9b')
        fig.update_xaxes(title='k', dtick=1)
        plot(fig, 330)
    note(f"<b>Alasan pemilihan k (metrik + interpretasi bisnis)</b><br>{km['reason']}", '#3d74c9')
    c1, c2, _ = st.columns([1, 1, 2])
    k_new = c1.selectbox('Ganti jumlah klaster (k)', km['ks'], index=km['ks'].index(km['k']))
    c2.write('')
    c2.write('')
    if c2.button('🔄 Jalankan ulang'):
        with st.spinner('Menghitung ulang...'):
            run_kmeans(k_new)
        st.rerun()

    p = km['profile']
    sec('📊', G2, 'Profil klaster dan label berbasis data', 'Jumlah pelanggan, rata-rata dan median RFM, kontribusi revenue', 'Tugas 15')
    show_df(km['ptable'])
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Bar(x=p['Segmen'], y=p['Med_Recency'], name='Median Recency (hari)', marker_color='#3d74c9'), secondary_y=False)
    fig.add_trace(go.Bar(x=p['Segmen'], y=p['Med_Frequency'], name='Median Frequency (x)', marker_color='#4cb8d8'), secondary_y=False)
    fig.add_trace(go.Bar(x=p['Segmen'], y=p['Med_Monetary'], name='Median Monetary (Rp, sumbu kanan)', marker_color='#e3b060'), secondary_y=True)
    fig.update_layout(barmode='group')
    plot(fig, 360)

    sec('🔵', G3, 'Scatter plot klaster', 'Frequency vs Monetary dan Recency vs Monetary (sampel hingga 1.000 pelanggan)', 'Tugas 15')
    sc = km['scatter'].copy()
    sc['Segmen'] = sc['Segmen'].astype(str)

    def seg_scatter(xcol, xlabel):
        fig = go.Figure()
        for sg in km['seg_names']:
            g = sc[sc['Segmen'] == str(sg)]
            if len(g):
                fig.add_trace(go.Scatter(x=g[xcol], y=g['Monetary'], mode='markers', name=str(sg),
                                         marker=dict(color=segcolor(sg), size=8, opacity=.75, line=dict(color='white', width=.6))))
        fig.update_xaxes(title=xlabel)
        fig.update_yaxes(title='Monetary (Rp)')
        return fig

    a, b = st.columns(2)
    with a:
        plot(seg_scatter('Frequency', 'Frequency (kali)'), 420)
    with b:
        plot(seg_scatter('Recency', 'Recency (hari)'), 420)

    sec('📣', G4, 'Strategi pemasaran per segmen', 'Label hanya dipakai bila didukung profil RFM', 'Tugas 16')
    for r in p.to_dict('records'):
        st.markdown(f'<div class="strat" style="border-color:{segcolor(r["Segmen"])}"><span class="seg" style="background:{segcolor(r["Segmen"])}">{r["Segmen"]}</span>'
                    f'<span style="font-size:12px;color:#7083a3;font-weight:600">{fi(r["Jumlah"])} pelanggan · {pct(r["Pct_Revenue"])} revenue</span><br>'
                    f'<b>Karakteristik:</b> {r["Karakteristik"]}<br><b>Strategi:</b> {r["Strategi"]}</div>', unsafe_allow_html=True)
    seg_df = km['labeled'].copy()
    seg_df['Last_Transaction'] = seg_df['Last_Transaction'].dt.strftime('%Y-%m-%d')
    csv_btn(seg_df, 'pelanggan_tersegmentasi.csv', 'Pelanggan + segmen (CSV)', 'dl_seg')
    next_btn('dashboard', 'Lanjut ke Dashboard BI')


def page_dashboard():
    cl, ap, km = S['clean'], S['apr'], S['km']
    if not cl.get('success'):
        gate('Lakukan Preprocessing terlebih dahulu.', 'prep', 'Ke Preprocessing')
        return
    d = cl['df']
    hero('📊 Tahap 5 · Dashboard BI · 15%', 'Dashboard Business Intelligence',
         'KPI, tren, top produk, hasil Apriori, dan profil K-Means. Gunakan filter untuk analisis. Untuk tangkapan layar, gunakan screenshot browser (Tugas 19).')
    with st.container():
        c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
        dmin, dmax = d['Date'].min().date(), d['Date'].max().date()
        rng = c1.date_input('Rentang tanggal', (dmin, dmax), min_value=dmin, max_value=dmax)
        cat = c2.selectbox('Kategori', ['Semua'] + sorted(d['Category'].unique().tolist()))
        seg = c3.selectbox('Segmen pelanggan', ['Semua'] + (km['seg_names'] if km.get('success') else []), disabled=not km.get('success'))
        c4.caption('Segmen aktif setelah K-Means dijalankan.' if not km.get('success') else ' ')
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        s, e = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    else:
        s, e = pd.Timestamp(dmin), pd.Timestamp(dmax)
    dash = build_dashboard(s, e, cat, seg)
    if dash['empty']:
        st.warning('Tidak ada data untuk filter ini.')
        return
    k = dash['kpi']
    kpis([('💰', 'Total Penjualan', rp(k['sales']), G1), ('🧾', 'Jumlah Transaksi', fi(k['tx']), G2),
          ('👥', 'Pelanggan Unik', fi(k['cust']), G3), ('🧺', 'Rata-rata basket_size', fmt_id(k['basket'], 2), 'linear-gradient(135deg,#d99a4a,#e3b060)')])
    a, b = st.columns(2)
    with a:
        sec('📈', G1, 'Tren penjualan bulanan', 'Penjualan dan jumlah transaksi')
        m = dash['monthly']
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Scatter(x=m['Month'], y=m['Sales'], name='Penjualan', fill='tozeroy', line=dict(color='#3d74c9', width=3, shape='spline'),
                                 fillcolor='rgba(61,116,201,.15)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=m['Month'], y=m['Tx'], name='Transaksi', line=dict(color='#4cb8d8', width=3, shape='spline')), secondary_y=True)
        plot(fig, 340)
    with b:
        sec('🏅', G2, 'Top 10 produk', 'Berdasarkan penjualan bersih')
        tp = dash['top_prod'][::-1]
        plot(go.Figure(go.Bar(x=tp.values, y=tp.index, orientation='h', marker_color=PAL[1])), 340, False)
    a, b = st.columns(2)
    with a:
        sec('🗂️', G3, 'Top kategori', 'Kontribusi penjualan')
        tc = dash['top_cat']
        plot(go.Figure(go.Bar(x=tc.index, y=tc.values, marker_color=PAL)), 340, False)
    with b:
        sec('💳', G4, 'Metode pembayaran', 'Proporsi penjualan')
        py_ = dash['pay']
        plot(go.Figure(go.Pie(labels=py_.index, values=py_.values, hole=.58, marker=dict(colors=PAL, line=dict(color='#fff', width=4)))), 340)

    sec('🔗', G4, 'Hasil Apriori', 'Top aturan (dari seluruh Data_Clean)')
    if ap.get('success') and ap.get('top'):
        show_df(ap['top_df'])
    else:
        st.caption('Belum ada hasil Apriori. Jalankan menu Apriori.')
    sec('🎯', G2, 'Profil K-Means dan pola produk per segmen (bonus)', 'Perbandingan RFM antar segmen dan produk teratas tiap segmen')
    if km.get('success'):
        show_df(km['ptable'])
        if 'seg_pattern' in dash:
            st.markdown('**Pola produk per segmen (mengikuti filter)**')
            show_df(dash['seg_pattern'])
    else:
        st.caption('Belum ada hasil K-Means. Jalankan menu K-Means.')
    ins = list(dash['insights'])
    if ap.get('success') and ap.get('top'):
        t0 = ap['top'][0]
        ins.append(f"Kombinasi produk terkuat: {' + '.join(t0['ant'])} dengan {' + '.join(t0['con'])} (lift {fmt_id(t0['lift'], 2)}).")
    if km.get('success'):
        p0 = km['profile'].iloc[0]
        ins.append(f"Segmen dengan kontribusi revenue terbesar: {p0.Segmen} ({pct(p0.Pct_Revenue)}).")
    sec('🪄', G3, 'Insight untuk manajemen', 'Dihitung otomatis dari filter yang aktif', 'Tugas 19')
    note('<ul style="margin:0 0 0 18px">' + ''.join(f'<li>{i}</li>' for i in ins) + '</ul>', '#4cb8d8')


def page_recommend():
    ap, km = S['apr'], S['km']
    if not (ap.get('success') and km.get('success')):
        gate('Jalankan Apriori dan K-Means terlebih dahulu agar rekomendasi berbasis bukti data.', 'apriori', 'Ke Apriori')
        return
    hero('🎯 Tahap 6 · Rekomendasi Keputusan · 10%', '3 rekomendasi prioritas',
         'Setiap rekomendasi memuat bukti data, tindakan, KPI, estimasi biaya/manfaat, asumsi, dan cara evaluasi dampak.')
    for i, r in enumerate(build_recs(), 1):
        sec(r['icon'], f"linear-gradient(135deg,{r['warna']},#7cb8ec)", f"Prioritas {i} · {r['judul']}", '', 'Tugas 20-21')
        cells = [('✅ FAKTA · Bukti data', r['bukti'], '#e3eefc', '#27508f'), ('Tindakan', r['tindakan'], '#e6eefb', '#3d74c9'),
                 ('KPI keberhasilan', r['kpi'], '#e6f0f9', '#2a5f8f'), ('📐 ESTIMASI · Biaya / manfaat', r['estimasi'], '#fbf0db', '#8a6020'),
                 ('Asumsi', r['asumsi'], '#fdf5e6', '#8a6020'), ('📏 DAMPAK AKTUAL · Cara evaluasi', r['evaluasi'], '#fdeaee', '#a8485c')]
        for row in (cells[:2], cells[2:4], cells[4:]):
            cols = st.columns(2)
            for c, (t, txt, bg, fg) in zip(cols, row):
                c.markdown(f'<div class="rc" style="background:{bg}"><b style="color:{fg}">{t}</b>{txt}</div>', unsafe_allow_html=True)
            st.write('')
    sec('⚖️', G1, 'Pembeda temuan faktual, estimasi, dan dampak aktual', '', 'Tugas 22')
    show_df(pd.DataFrame([
        ('Faktual', 'Support, confidence, lift aturan; profil RFM segmen; total penjualan; bulan puncak. Dihitung langsung dari Data_Clean.', 'Terbukti dari data historis'),
        ('Estimasi', 'Tambahan penjualan dan biaya pada setiap rekomendasi (adopsi 5%, reactivation 10%, kenaikan 3%).', 'Berbasis asumsi, belum diuji'),
        ('Dampak aktual', 'Kenaikan penjualan nyata setelah bundling, win-back, atau program reward dijalankan.', 'Belum ada. Perlu A/B test atau grup kontrol')],
        columns=['Jenis', 'Contoh pada laporan ini', 'Status']))
    st.caption('Kenaikan penjualan tidak boleh dinyatakan sebagai fakta sebelum diuji dengan implementasi terkontrol.')


def page_files():
    cl, ft, ap, km = S['clean'], S['feat'], S['apr'], S['km']
    hero('📁 Luaran', 'Berkas yang dikumpulkan', 'Unduh data hasil preprocessing, dokumentasi perubahan, dan dataset agregat.')
    items = []
    if cl.get('success'):
        items += [('Data_Clean (CSV)', 'Hasil preprocessing', cl['df'], 'Data_Clean.csv'),
                  ('Dokumentasi preprocessing (CSV)', 'Keputusan, jumlah terdampak, baris sebelum/sesudah', cl['log'], 'dokumentasi_preprocessing.csv')]
    if ft.get('success'):
        items += [('basket_size per transaksi (CSV)', 'Dataset transaksi-basket', ft['tx'], 'basket_size.csv'),
                  ('Matriks transaksi x produk (CSV)', 'Input Apriori', ft['matrix'].reset_index(), 'matriks_transaksi_produk.csv'),
                  ('Tabel RFM (CSV)', 'Dataset pelanggan-RFM', ft['rfm'].assign(Last_Transaction=ft['rfm']['Last_Transaction'].dt.strftime('%Y-%m-%d')), 'tabel_rfm.csv')]
    if ap.get('success'):
        items.append(('Aturan asosiasi (CSV)', 'Support, confidence, lift', ap['rules_df'], 'aturan_asosiasi.csv'))
    if km.get('success'):
        items.append(('Pelanggan + segmen (CSV)', 'Hasil K-Means', km['labeled'].assign(Last_Transaction=km['labeled']['Last_Transaction'].dt.strftime('%Y-%m-%d')), 'pelanggan_tersegmentasi.csv'))
    if not items:
        st.info('Belum ada berkas. Jalankan tahapan analisis terlebih dahulu.')
    for i, (title, desc, df, fname) in enumerate(items):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f'<div class="sec" style="margin:0"><div class="st2">{title}</div><div class="ss">{desc}</div></div>', unsafe_allow_html=True)
        with c2:
            st.write('')
            st.download_button('⬇ Unduh', df.to_csv(index=False).encode('utf-8-sig'), file_name=fname, mime='text/csv', key=f'f_{i}')
    note('<b>Berkas lain yang perlu disiapkan manual</b><br>Laporan analisis, tangkapan layar dashboard, dan presentasi ringkas untuk manajemen. '
         'Kode program ini (app_uts_streamlit.py) menjadi bukti reproduksi.', '#e3b060')


# ============================================================
# SIDEBAR + ROUTER
# ============================================================
def load_file(f):
    try:
        try:
            xls = pd.ExcelFile(f, engine='calamine')
        except Exception:
            f.seek(0)
            xls = pd.ExcelFile(f)
        names = xls.sheet_names
        raw_name = next((n for n in names if norm(n) == 'dataraw'), None) or \
                   next((n for n in names if 'raw' in n.lower() or 'mentah' in n.lower()), None)
        if raw_name is None:
            st.sidebar.error("Sheet 'Data_Raw' tidak ditemukan. Sheet di file: " + ', '.join(names))
            return
        df = pd.read_excel(xls, sheet_name=raw_name).dropna(how='all').reset_index(drop=True)
        df.columns = df.columns.astype(str).str.strip()
        S.update({'raw': df, 'raw_name': raw_name, 'audit': {}, 'clean': {}, 'feat': {}, 'apr': {}, 'km': {}, 'page': 'eda'})
        au = run_audit()
        if au.get('success'):
            st.toast(f"Sheet '{raw_name}' dimuat ({fi(len(df))} baris). Sheet lain diabaikan.", icon='✅')
        else:
            st.sidebar.error('Gagal mengaudit data: ' + str(au.get('error')))
    except Exception as e:
        st.sidebar.error(f'Gagal membaca file: {e}')


PAGES = [('eda', '1. Business & EDA', page_eda, lambda: S['audit'].get('success')),
         ('prep', '2. Preprocessing', page_prep, lambda: S['clean'].get('success')),
         ('features', '   Basket, RFM, Scaling', page_features, lambda: S['feat'].get('success')),
         ('apriori', '3. Apriori / MBA', page_apriori, lambda: S['apr'].get('success')),
         ('kmeans', '4. K-Means', page_kmeans, lambda: S['km'].get('success')),
         ('dashboard', '5. Dashboard BI', page_dashboard, lambda: False),
         ('recommend', '6. Rekomendasi', page_recommend, lambda: False),
         ('files', 'Berkas Dikumpulkan', page_files, lambda: False)]


def handle_upload(f):
    if f is not None:
        sig = (f.name, f.size)
        if S['file_sig'] != sig:
            S['file_sig'] = sig
            with st.spinner('Membaca Data_Raw...'):
                load_file(f)
            if S['raw'] is not None:
                st.rerun()


def main():
    if not S['logged_in']:
        login_page()
        return
    with st.sidebar:
        st.markdown('<div style="font-size:18px;font-weight:800;color:#fff">📊 UTS DM &amp; BI</div>'
                    '<div style="font-size:11px;color:#a9c8f2;font-weight:600">Apriori · K-Means · Dashboard</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="display:flex;align-items:center;gap:12px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.16);border-radius:18px;padding:10px 12px;margin:12px 0">'
                    f'<img class="ava-s" src="{photo_uri()}" alt="foto"><div><b style="font-size:12px;color:#fff">ADELIA MUTIARA</b><br>'
                    f'<span style="font-size:10px;color:#a9c8f2">NPM: 51230004</span></div></div>', unsafe_allow_html=True)
        with st.expander('📷 Ganti foto'):
            photo_uploader('Unggah foto', 'photo_side')
        st.markdown('<div class="up-title">⬆️ UNGGAH DATA EXCEL</div>', unsafe_allow_html=True)
        handle_upload(st.file_uploader('Sheet Data_Raw (.xlsx / .xls)', type=['xlsx', 'xls'], key='side_up'))
        if S['raw'] is not None:
            st.caption(f"{S['raw_name']} · {fi(len(S['raw']))} baris")
            labels = {k: (('✅ ' if done() else '') + lbl.strip()) for k, lbl, _, done in PAGES}
            keys = [k for k, *_ in PAGES]
            S['page'] = st.radio('Tahapan UTS', keys, index=keys.index(S['page']) if S['page'] in keys else 0,
                                 format_func=lambda k: labels[k], key='nav_radio_' + S['page'])
        if st.button('Keluar', **STRETCH):
            S['logged_in'] = False
            st.rerun()

    if S['raw'] is None:
        page_upload_hint()
        return
    st.markdown(f'<div class="me"><img src="{photo_uri()}" alt="foto"><div><b>Adelia Mutiara</b><span>NPM 51230004 · UTS Data Mining &amp; BI</span></div></div>', unsafe_allow_html=True)
    steps = [('eda', 'Business & EDA'), ('prep', 'Preprocessing'), ('apriori', 'Apriori'), ('kmeans', 'K-Means'),
             ('dashboard', 'Dashboard BI'), ('recommend', 'Rekomendasi')]
    done = {k: d for k, _, _, d in PAGES}
    html = ''.join(f'<div class="stp {"on" if S["page"] == k or (k == "prep" and S["page"] == "features") else ("done" if done[k]() else "")}">'
                   f'<div class="n">{"✓" if done[k]() and S["page"] != k else i}</div><div class="t">{t}</div></div>'
                   for i, (k, t) in enumerate(steps, 1))
    st.markdown(f'<div class="stepper">{html}</div>', unsafe_allow_html=True)
    dict((k, fn) for k, _, fn, _ in PAGES)[S['page']]()


main()
