import streamlit as st
import cv2
import numpy as np
import ezdxf
import io

# Sayfa Yapılandırması
st.set_page_config(page_title="NefeSoft Digitizer Mobile v2.0", layout="wide", initial_sidebar_state="expanded")

# --- ULTRA MODERN CAD DARK THEME (CSS) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0B0F17 !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1E293B !important;
    }
    .nefe-header {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .nefe-logo {
        font-size: 28px;
        font-weight: 900;
        letter-spacing: 1.5px;
        background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
    }
    .nefe-sub {
        font-size: 13px;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 4px;
    }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, input {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border-color: #334155 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] span {
        color: #F8FAFC !important;
    }
    label, .stRadio p, .stCheckbox label {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
    }
    .stButton>button, div.stDownloadButton>button {
        background: linear-gradient(90deg, #0284C7 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 14px 28px !important;
        font-size: 16px !important;
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.4) !important;
    }
    div[data-testid="stImage"] {
        background: #111827 !important;
        border-radius: 12px !important;
        padding: 10px !important;
        border: 1px solid #1E293B !important;
    }
    .stAlert {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border: 1px solid #0284C7 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header Banner
st.markdown("""
    <div class="nefe-header">
        <div class="nefe-logo">NefeSoft® Digitizer Mobile</div>
        <div class="nefe-sub">Endüstriyel Mobil Fotoğraftan Vektörel CAD/DXF Otomasyonu v2.0</div>
    </div>
""", unsafe_allow_html=True)

# --- SOL KONTROL PANELİ ---
with st.sidebar:
    st.markdown("### ⚙️ Kalibrasyon Ayarları")
    
    scale_method = st.radio(
        "Ölçek Neye Göre Hesaplansın?",
        ["Kalıbın Bilinen Bir Kenarı İle", "Kareli Zemin (Grid) İle"]
    )
    
    if scale_method == "Kalıbın Bilinen Bir Kenarı İle":
        ref_type = st.selectbox(
            "Referans Kenar:",
            ["Kalıbın Toplam Genişliği (Yatay - X)", "Kalıbın Toplam Yüksekliği (Dikey - Y)", "Kalıbın Taban Düz Kenarı"]
        )
        known_length_mm = st.number_input("Gerçek Ölçü (mm):", min_value=1.0, value=150.0, step=5.0)
        grid_unit_mm = 10.0
    else:
        grid_unit_mm = st.number_input("Zemindeki 1 Kare (mm):", min_value=1.0, value=10.0, step=1.0)
        ref_type = None
        known_length_mm = 100.0

    st.markdown("---")
    st.markdown("### 🧹 Algılama & Temizlik")
    only_largest = st.checkbox("🎯 Sadece Ana Kalıba Odaklan", value=True)
    auto_thresh = st.checkbox("🤖 Akıllı Otomatik Algılama (Otsu)", value=True)
    invert_thresh = st.checkbox("🔄 Renkleri Ters Çevir", value=False)
    
    if not auto_thresh:
        threshold_val = st.slider("Manuel Eşik Değeri:", 0, 255, 120)
    else:
        threshold_val = 120
        
    margin_crop = st.slider("Kenar Temizleme (Px):", 10, 150, 45)
    
    st.markdown("---")
    st.markdown("### ✨ Eğri Yumuşatma (Chaikin)")
    smooth_factor = st.slider("Köşe Sadeleştirme (Epsilon):", 1.0, 15.0, 4.0, step=0.5)
    chaikin_iters = st.slider("Chaikin Yumuşatma Seviyesi:", 0, 5, 3)

def chaikin_smooth(points, iterations=3):
    if len(points) < 3 or iterations == 0:
        return points
    pts = np.array(points, dtype=float)
    for _ in range(iterations):
        new_pts = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.append(q)
            new_pts.append(r)
        pts = np.array(new_pts)
    return pts

def detect_grid_pixel_size(gray_img):
    edges = cv2.Canny(gray_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is not None:
        lines = lines.reshape(-1, 4)
        x_coords = sorted(list(lines[:, 0]) + list(lines[:, 2]))
        diffs = np.diff(x_coords)
        valid_diffs = diffs[(diffs > 15) & (diffs < 200)]
        if len(valid_diffs) > 0:
            return float(np.median(valid_diffs))
    return None

# MOBİL KAMERA VE GALERİ SEÇİMİ
input_mode = st.radio("Fotoğraf Alma Yöntemi:", ["📷 Canlı Kamera İle Çek", "📂 Galeriden / Dosyadan Seç"], horizontal=True)

if input_mode == "📷 Canlı Kamera İle Çek":
    uploaded_file = st.camera_input("Kalıbın Fotoğrafını Çekin")
else:
    uploaded_file = st.file_uploader("Kalıp Fotoğrafını Yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_type = cv2.THRESH_BINARY_INV if not invert_thresh else cv2.THRESH_BINARY
    if auto_thresh:
        _, thresh = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(blurred, threshold_val, 255, thresh_type)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_thresh = cv2.morphologyEx(cleaned_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    m = margin_crop
    cleaned_thresh[:m, :] = 0
    cleaned_thresh[-m:, :] = 0
    cleaned_thresh[:, :m] = 0
    cleaned_thresh[:, -m:] = 0

    contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preview_img = image.copy()
    valid_contours = []

    filtered_contours = []
    min_area = (img_h * img_w) * 0.002
    max_area = (img_h * img_w) * 0.50

    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            if x > m and y > m and (x + w) < (img_w - m) and (y + h) < (img_h - m):
                filtered_contours.append(c)

    mm_per_pixel = 0.2
    if filtered_contours:
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
        main_contour = filtered_contours[0]
        x_b, y_b, w_b, h_b = cv2.boundingRect(main_contour)

        if scale_method == "Kalıbın Bilinen Bir Kenarı İle":
            if ref_type == "Kalıbın Toplam Genişliği (Yatay - X)":
                ref_pixel_len = w_b
                cv2.line(preview_img, (x_b, y_b - 15), (x_b + w_b, y_b - 15), (0, 242, 254), 4)
            elif ref_type == "Kalıbın Toplam Yüksekliği (Dikey - Y)":
                ref_pixel_len = h_b
                cv2.line(preview_img, (x_b - 15, y_b), (x_b - 15, y_b + h_b), (0, 242, 254), 4)
            else:
                ref_pixel_len = w_b
                cv2.line(preview_img, (x_b, y_b + h_b + 15), (x_b + w_b, y_b + h_b + 15), (0, 242, 254), 4)

            if ref_pixel_len > 0:
                mm_per_pixel = known_length_mm / ref_pixel_len
                st.info(f"📐 **NefeSoft Kalibrasyonu:** 1 Piksel = {mm_per_pixel:.4f} mm")
        else:
            grid_pixel_size = detect_grid_pixel_size(gray)
            if grid_pixel_size:
                mm_per_pixel = grid_unit_mm / grid_pixel_size
                st.info(f"📐 **Izgara Kalibrasyonu:** 1 Piksel = {mm_per_pixel:.4f} mm")

        target_contours = [main_contour] if only_largest else filtered_contours
        for c in target_contours:
            approx = cv2.approxPolyDP(c, smooth_factor, True)
            pts_raw = approx.reshape(-1, 2)
            smoothed_pts = chaikin_smooth(pts_raw, iterations=chaikin_iters)
            valid_contours.append(smoothed_pts)
            cv_pts = smoothed_pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.drawContours(preview_img, [cv_pts], -1, (0, 242, 254), 4)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="1. Orijinal Fotoğraf", use_container_width=True)
    with col2:
        st.image(cleaned_thresh, caption="2. İşlenmiş Maske", use_container_width=True)
    with col3:
        st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption="3. NefeSoft Vektör Tespiti", use_container_width=True)

    st.markdown("---")

    if len(valid_contours) > 0:
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()

        for c_pts in valid_contours:
            points = [(float(pt[0] * mm_per_pixel), float(-pt[1] * mm_per_pixel)) for pt in c_pts]
            if len(points) > 2:
                msp.add_lwpolyline(points, close=True)

        dxf_stream = io.StringIO()
        doc.write(dxf_stream)
        dxf_data = dxf_stream.getvalue()

        st.success("✅ Vektörel CAD Çizgileri 1:1 Hassasiyetle Hazırlandı!")
        st.download_button(
            label="💾 1:1 ÖLÇEKLİ DXF DOSYASINI İNDİR",
            data=dxf_data,
            file_name="NefeSoft_Kalip_Ciktisi.dxf",
            mime="application/dxf",
            use_container_width=True
        )

st.markdown("<br><hr><center style='color:#475569; font-size:12px;'>© 2026 NefeSoft® Otomasyon & Yazılım Teknolojileri. Tüm Hakları Saklıdır.</center>", unsafe_allow_html=True)