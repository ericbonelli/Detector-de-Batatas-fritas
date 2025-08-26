# app.py — QC de Lotes • POC (CPU) + Upload/URL/HF Hub (privado/público)
import os, io, json, base64, tempfile
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import cv2
from skimage.morphology import skeletonize
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# -------------------- Config geral --------------------
st.set_page_config(page_title="QC de Lotes • POC (CPU)", layout="wide")
st.title("QC de Lotes • POC (CPU)")

BUCKETS = [">100mm", ">=75mm", ">=50mm,<75mm", "<50mm"]
DEFAULT_LIMITS = {"max_gt100": 0.10, "min_ge75": 0.65, "max_ge50_lt75": 0.20, "max_lt50": 0.05}
DEVICE = "cpu"  # Streamlit Community Cloud = CPU

# -------------------- Utils --------------------
def generate_lote_id(output_dir: str, prefix: str = "Lote") -> str:
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    existing = [f for f in os.listdir(output_dir) if f.startswith(f"{prefix}_{date_str}_")]
    idx = len(existing) + 1
    return f"{prefix}_{date_str}_{idx:03d}"

def ensure_all_buckets(totals_like) -> Dict[str, int]:
    counts = {k: 0 for k in BUCKETS}
    for k, v in dict(totals_like).items():
        if k in counts:
            counts[k] = int(v)
    return counts

def compute_percentages(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: counts[k] / total for k in counts}

def decide(perc: Dict[str, float], limits: Dict[str, float]):
    broken = []
    if perc[">100mm"] > limits["max_gt100"]:
        broken.append(f">100mm: {perc['>100mm']:.1%} > {limits['max_gt100']:.0%} (excedeu)")
    if perc[">=75mm"] < limits["min_ge75"]:
        broken.append(f">=75mm: {perc['>=75mm']:.1%} < {limits['min_ge75']:.0%} (abaixo do mínimo)")
    if perc[">=50mm,<75mm"] > limits["max_ge50_lt75"]:
        broken.append(f">=50mm,<75mm: {perc['>=50mm,<75mm']:.1%} > {limits['max_ge50_lt75']:.0%} (excedeu)")
    if perc["<50mm"] > limits["max_lt50"]:
        broken.append(f"<50mm: {perc['<50mm']:.1%} > {limits['max_lt50']:.0%} (excedeu)")
    decision = "ACEITE" if not broken else "REJEITE"
    return decision, broken

def bucket_from_length(mm: float) -> str:
    if mm > 100: return ">100mm"
    if mm >= 75: return ">=75mm"
    if mm >= 50: return ">=50mm,<75mm"
    return "<50mm"

def pca_length(mask: np.ndarray, px_per_mm: float) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) < 5: return 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    proj = pts @ v
    length_px = proj.max() - proj.min()
    return float(length_px / px_per_mm)

def skeleton_length(mask: np.ndarray, px_per_mm: float) -> float:
    sk = skeletonize((mask > 0).astype(np.uint8)).astype(np.uint8)
    length_px = sk.sum()
    return float(length_px / px_per_mm)

def measure_length_hybrid(crop_bgr: np.ndarray, px_per_mm: float):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)
    if thr.mean() > 127:  # fundo claro -> inverte
        thr = cv2.bitwise_not(thr)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, {"pca": 0.0, "skeleton": 0.0}
    cnt = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(thr)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    lp = pca_length(mask, px_per_mm)
    ls = skeleton_length(mask, px_per_mm)
    lh = max(lp, ls)  # híbrido POC: evita subestimativa
    return lh, {"pca": lp, "skeleton": ls}

def write_report_csv(path_csv: str, counts, perc, decision, broken_rules, limits, lote_id: str):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path_csv, "w", encoding="utf-8") as f:
        f.write(f"lote_id,{lote_id}\n")
        f.write(f"timestamp,{ts}\n")
        f.write("limits," + json.dumps(limits) + "\n\n")
        f.write("bucket,quantidade,percentual,limite_aplicado,condicao,ok\n")
        rows = [
            (">100mm", counts[">100mm"], perc[">100mm"], f"<= {limits['max_gt100']:.0%}", "máx", perc[">100mm"] <= limits["max_gt100"]),
            (">=75mm", counts[">=75mm"], perc[">=75mm"], f">= {limits['min_ge75']:.0%}", "mín", perc[">=75mm"] >= limits["min_ge75"]),
            (">=50mm,<75mm", counts[">=50mm,<75mm"], perc[">=50mm,<75mm"], f"<= {limits['max_ge50_lt75']:.0%}", "máx", perc[">=50mm,<75mm"] <= limits["max_ge50_lt75"]),
            ("<50mm", counts["<50mm"], perc["<50mm"], f"<= {limits['max_lt50']:.0%}", "máx", perc["<50mm"] <= limits["max_lt50"]),
        ]
        for b,q,p,lim,cond,ok in rows:
            f.write(f"{b},{q},{p:.4f},{lim},{cond},{'OK' if ok else 'FALHOU'}\n")
        f.write("\n")
        f.write(f"total_itens,{sum(counts.values())}\n")
        f.write(f"decisao,{decision}\n")
        if broken_rules:
            f.write("regras_quebradas," + "|".join(broken_rules) + "\n")

def df_download_link(df: pd.DataFrame, filename: str) -> str:
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Baixar CSV ({filename})</a>'

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Pesos do YOLO")
    weights_file = st.file_uploader("Upload .pt", type=["pt"])
    weights_url  = st.text_input("URL dos pesos (.pt)", value="")
    hf_repo_input = st.text_input("HF Hub ID", value="", help="Ex.: seu-usuario/batatas-yolo")
    hf_filename_input = st.text_input("Arquivo no Hub", value="best.pt")

    st.markdown("---")
    conf_thres = st.slider("Confiança (YOLO)", 0.1, 0.9, 0.25, 0.05)
    imgsz = st.select_slider("Imagem (YOLO imgsz)", options=[320, 416, 512, 640, 736, 832], value=512)

    st.markdown("---")
    st.markdown("**Conversão px → mm**")
    px_per_mm = st.number_input("Pixels por milímetro (px/mm)", 0.01, 100.0, 5.0, 0.1)

    st.markdown("---")
    st.markdown("**LIMITS**")
    limits = {
        "max_gt100": st.number_input(">100mm (máx %)", 0.0, 1.0, DEFAULT_LIMITS["max_gt100"], 0.01, format="%.2f"),
        "min_ge75":  st.number_input(">=75mm (mín %)", 0.0, 1.0, DEFAULT_LIMITS["min_ge75"],  0.01, format="%.2f"),
        "max_ge50_lt75": st.number_input(">=50mm,<75mm (máx %)", 0.0, 1.0, DEFAULT_LIMITS["max_ge50_lt75"], 0.01, format="%.2f"),
        "max_lt50":  st.number_input("<50mm (máx %)", 0.0, 1.0, DEFAULT_LIMITS["max_lt50"],  0.01, format="%.2f"),
    }

    st.markdown("---")
    output_dir = st.text_input("Pasta saída", value="qc_outputs")
    prefix = st.text_input("Prefixo do lote", value="Lote")

# -------------------- Loaders (3 rotas) --------------------
HF_TOKEN = os.getenv("HF_TOKEN", None)           # Secrets (Streamlit Cloud)
HF_REPO_DEFAULT = os.getenv("HF_REPO", "")       # opcional nos Secrets
HF_FILE_DEFAULT = os.getenv("HF_FILE", "best.pt")

@st.cache_resource(show_spinner=True)
def load_model_from_upload(weights_bytes: bytes):
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    tf.write(weights_bytes); tf.flush(); tf.close()
    model = YOLO(tf.name)
    try: model.to(DEVICE)
    except Exception: pass
    return model, tf.name

@st.cache_resource(show_spinner=True)
def load_model_from_url(url: str):
    import requests
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    tf.write(r.content); tf.flush(); tf.close()
    model = YOLO(tf.name)
    try: model.to(DEVICE)
    except Exception: pass
    return model, tf.name

@st.cache_resource(show_spinner=True)
def load_model_from_hub(repo_id: str, filename: str, token: str | None):
    path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    model = YOLO(path)
    try: model.to(DEVICE)
    except Exception: pass
    return model, path

# -------------------- Upload de imagens --------------------
uploaded_imgs = st.file_uploader("Envie 1..N imagens do lote", type=["jpg","jpeg","png"], accept_multiple_files=True)

# -------------------- Ação principal --------------------
if st.button("Processar lote"):
    if not uploaded_imgs:
        st.warning("Envie ao menos 1 imagem.")
        st.stop()

    # 1) Upload -> 2) URL -> 3) HF Hub (privado/público)
    model = None; tmp_path = None
    try:
        if weights_file is not None:
            model, tmp_path = load_model_from_upload(weights_file.read())
        elif weights_url.strip():
            model, tmp_path = load_model_from_url(weights_url.strip())
        else:
            repo = HF_REPO_DEFAULT or hf_repo_input.strip()
            fname = HF_FILE_DEFAULT if HF_REPO_DEFAULT else hf_filename_input.strip()
            if not repo:
                st.error("Forneça upload, URL ou HF Hub ID (público/privado).")
                st.stop()
            model, tmp_path = load_model_from_hub(repo, fname, HF_TOKEN)  # usa token se houver
    except Exception as e:
        st.error(f"Falha ao carregar o modelo: {e}")
        st.stop()

    lote_id = generate_lote_id(output_dir=output_dir, prefix=prefix)
    st.info(f"lote_id gerado: **{lote_id}**")

    lengths_mm: List[float] = []
    details_rows = []

    for file in uploaded_imgs:
        data = np.frombuffer(file.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            st.warning(f"Não consegui ler a imagem {file.name}.")
            continue

        # YOLO (CPU)
        results = model.predict(img, conf=conf_thres, imgsz=int(imgsz), device=DEVICE, verbose=False)
        dets = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

        st.write(f"📷 **{file.name}** — {len(dets)} batatas detectadas")
        preview = img.copy()
        for (x1,y1,x2,y2) in dets:
            cv2.rectangle(preview, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption=f"Detecções - {file.name}", use_container_width=True)

        # Medição por batata
        for j,(x1,y1,x2,y2) in enumerate(dets, start=1):
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            crop = img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            if crop.size == 0:
                continue
            l_h, parts = measure_length_hybrid(crop, px_per_mm=px_per_mm)
            lengths_mm.append(l_h)
            details_rows.append({
                "image": file.name, "obj": j,
                "length_mm_hybrid": round(l_h,2),
                "pca_mm": round(parts["pca"],2),
                "skeleton_mm": round(parts["skeleton"],2),
            })

    # Agregação → buckets
    totals = {k:0 for k in BUCKETS}
    for mm in lengths_mm:
        totals[bucket_from_length(mm)] += 1
    counts = ensure_all_buckets(totals)
    perc = compute_percentages(counts)
    decision, broken = decide(perc, limits)

    # Resumo
    df_qc = pd.DataFrame([
        [">100mm",        counts[">100mm"],        f"{perc['>100mm']:.2%}",       f"<= {limits['max_gt100']:.0%}"],
        [">=75mm",        counts[">=75mm"],        f"{perc['>=75mm']:.2%}",       f">= {limits['min_ge75']:.0%}"],
        [">=50mm,<75mm",  counts[">=50mm,<75mm"],  f"{perc['>=50mm,<75mm']:.2%}", f"<= {limits['max_ge50_lt75']:.0%}"],
        ["<50mm",         counts["<50mm"],         f"{perc['<50mm']:.2%}",        f"<= {limits['max_lt50']:.0%}"],
    ], columns=["Faixa","Quantidade","Percentual","Regra"])

    st.subheader("Resumo do Lote")
    st.dataframe(df_qc, use_container_width=True)
    st.success(f"Decisão do lote **{lote_id}**: **{decision}**")
    if broken:
        st.error("Regras quebradas:\n- " + "\n- ".join(broken))

    # Detalhes
    if details_rows:
        st.subheader("Detalhes por batata")
        st.dataframe(pd.DataFrame(details_rows), use_container_width=True)

    # CSV + download
    csv_path = os.path.join(output_dir, f"{lote_id}_relatorio.csv")
    write_report_csv(csv_path, counts, perc, decision, broken, limits, lote_id)
    st.info(f"Relatório salvo em: `{csv_path}`")
    try:
        df_csv = pd.read_csv(csv_path)
        st.markdown(df_download_link(df_csv, os.path.basename(csv_path)), unsafe_allow_html=True)
    except Exception:
        pass

else:
    st.caption("Escolha: Upload .pt, URL dos pesos ou HF Hub (público/privado via Secrets). Ajuste px/mm e LIMITS e clique em **Processar lote**.")
