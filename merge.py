import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage import filters, feature, measure, segmentation, color, restoration
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import disk
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import signal
import io
import base64

# Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Pengolahan Citra Digital",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tambahkan identitas di bawah page title
st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h4 style="color: #666; margin: 0;">Dibuat oleh:</h4>
    <h3 style="color: #1e3a8a; margin: 5px 0;">Dhea Putri Ananda</h3>
    <p style="color: #888; margin: 0; font-size: 16px;">NIM: 231511009</p>
</div>
""", unsafe_allow_html=True)

# CSS untuk styling
st.markdown("""
<style>
.main-title {
    font-size: 3em;
    font-weight: bold;
    text-align: center;
    color: #1e3a8a;
    margin-bottom: 30px;
}
.sub-title {
    font-size: 1.5em;
    font-weight: bold;
    color: #3b82f6;
    margin-top: 20px;
    margin-bottom: 10px;
}
.info-box {
    background-color: #f0f9ff;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #3b82f6;
    margin: 10px 0;
}
.module-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 15px;
    color: white;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

def load_image(image_file):
    """Memuat dan mengonversi gambar"""
    img = Image.open(image_file)
    return np.array(img)

def rgb_analysis(image):
    """Analisis komponen RGB"""
    if len(image.shape) == 3:
        r_channel = image[:,:,0]
        g_channel = image[:,:,1]
        b_channel = image[:,:,2]
        
        # Histogram RGB
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Gambar asli
        axes[0,0].imshow(image)
        axes[0,0].set_title('Gambar Asli')
        axes[0,0].axis('off')
        
        # Histogram RGB
        axes[0,1].hist(r_channel.ravel(), bins=256, color='red', alpha=0.7, label='Red')
        axes[0,1].hist(g_channel.ravel(), bins=256, color='green', alpha=0.7, label='Green')
        axes[0,1].hist(b_channel.ravel(), bins=256, color='blue', alpha=0.7, label='Blue')
        axes[0,1].set_title('Histogram RGB')
        axes[0,1].legend()
        
        # Channel terpisah
        axes[1,0].imshow(r_channel, cmap='Reds')
        axes[1,0].set_title('Channel Merah')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(g_channel, cmap='Greens')
        axes[1,1].set_title('Channel Hijau')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        return fig
    else:
        st.warning("Gambar harus dalam format RGB untuk analisis RGB")
        return None

def face_detection(image):
    """Deteksi wajah menggunakan Haar Cascade"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Load classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Gambar kotak di sekitar wajah
    result = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return result, len(faces)

def add_noise(image, noise_type='gaussian'):
    """Menambahkan noise pada gambar"""
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 25, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        noisy = image.copy()
        # Salt noise
        salt = np.random.random(image.shape[:2]) < 0.05
        noisy[salt] = 255
        # Pepper noise
        pepper = np.random.random(image.shape[:2]) < 0.05
        noisy[pepper] = 0
    elif noise_type == 'speckle':
        noise = np.random.randn(*image.shape)
        noisy = image + image * noise * 0.1
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy

def denoise_image(image, method='gaussian'):
    """Menghilangkan noise dari gambar"""
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        return cv2.medianBlur(image, 5)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == 'nlm':
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

def contour_analysis(image):
    """Analisis kontur dan bentuk"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    # Threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Cari kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Gambar kontur
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    # Analisis bentuk
    shapes_info = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area > 100:  # Filter kontur kecil
            # Aproximasi kontur
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Identifikasi bentuk
            sides = len(approx)
            if sides == 3:
                shape = "Segitiga"
            elif sides == 4:
                shape = "Persegi/Persegi Panjang"
            elif sides > 4:
                shape = "Lingkaran/Elips"
            else:
                shape = "Bentuk Kompleks"
            
            shapes_info.append({
                'Bentuk': shape,
                'Luas': area,
                'Keliling': perimeter,
                'Jumlah Sisi': sides
            })
    
    return result, shapes_info

def image_compression(image, quality=50):
    """Kompresi gambar JPEG"""
    # Konversi ke PIL Image
    pil_img = Image.fromarray(image)
    
    # Simpan dengan kualitas tertentu
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    
    # Muat kembali
    buffer.seek(0)
    compressed_img = Image.open(buffer)
    
    return np.array(compressed_img)

def color_space_conversion(image):
    """Konversi ruang warna"""
    results = {}
    
    if len(image.shape) == 3:
        # HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        results['HSV'] = hsv
        
        # LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        results['LAB'] = lab
        
        # YUV
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        results['YUV'] = yuv
        
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        results['Grayscale'] = gray
    
    return results

def texture_analysis(image):
    """Analisis tekstur menggunakan GLCM"""
    # Konversi ke grayscale jika perlu
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Hitung GLCM
    distances = [1, 2, 3]
    angles = [0, 45, 90, 135]
    
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
    
    glcm = graycomatrix(gray, distances, np.radians(angles), levels=256, symmetric=True, normed=True)
    
    results = {}
    for prop in properties:
        results[prop] = graycoprops(glcm, prop)
    
    return results

# ============= MODUL 2: HISTOGRAM & OPERATIONS =============

def histogram_operations(image):
    """Operasi histogram lengkap"""
    # Konversi ke grayscale jika perlu
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Histogram asli
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Histogram equalization
    equalized = cv2.equalizeHist(gray)
    hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    hist_clahe = cv2.calcHist([clahe_img], [0], None, [256], [0, 256])
    
    # Histogram stretching
    min_val = np.min(gray)
    max_val = np.max(gray)
    stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    hist_stretched = cv2.calcHist([stretched], [0], None, [256], [0, 256])
    
    return {
        'original': {'image': gray, 'hist': hist},
        'equalized': {'image': equalized, 'hist': hist_eq},
        'clahe': {'image': clahe_img, 'hist': hist_clahe},
        'stretched': {'image': stretched, 'hist': hist_stretched}
    }

def advanced_histogram_analysis(image):
    """Analisis histogram lanjutan"""
    if len(image.shape) == 3:
        # Analisis per channel untuk RGB
        channels = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram RGB
        for i, (channel, color) in enumerate(zip(channels, colors)):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            axes[0, 0].plot(hist, color=color, alpha=0.7, label=channel)
        axes[0, 0].set_title('Histogram RGB')
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('Intensitas')
        axes[0, 0].set_ylabel('Frekuensi')
        
        # Histogram kumulatif
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        
        axes[0, 1].plot(cdf_normalized, color='blue')
        axes[0, 1].set_title('Histogram Kumulatif')
        axes[0, 1].set_xlabel('Intensitas')
        axes[0, 1].set_ylabel('Frekuensi Kumulatif')
        
        # Statistik histogram
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        median_val = np.median(gray)
        
        axes[1, 0].bar(['Mean', 'Std', 'Median'], [mean_val, std_val, median_val])
        axes[1, 0].set_title('Statistik Gambar')
        axes[1, 0].set_ylabel('Nilai')
        
        # Entropy
        hist_norm = hist / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
        
        axes[1, 1].text(0.5, 0.5, f'Entropy: {entropy:.2f}\nMean: {mean_val:.2f}\nStd: {std_val:.2f}', 
                       transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Informasi Citra')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    return None

# ============= MODUL 3: FILTERING & FOURIER TRANSFORM =============

def convolution_filters(image):
    """Berbagai jenis filter konvolusi"""
    # Konversi ke grayscale jika perlu
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Definisi kernel
    kernels = {
        'Sobel X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Sobel Y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'Laplacian': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'Edge Detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'Emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    }
    
    results = {}
    for name, kernel in kernels.items():
        filtered = cv2.filter2D(gray, -1, kernel)
        results[name] = np.clip(filtered, 0, 255).astype(np.uint8)
    
    return results

def fourier_transform_analysis(image):
    """Analisis Transformasi Fourier"""
    # Konversi ke grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # FFT
    f_transform = fft2(gray)
    f_shift = fftshift(f_transform)
    
    # Magnitude spectrum
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    # Phase spectrum
    phase_spectrum = np.angle(f_shift)
    
    # Low-pass filter
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Buat mask untuk low-pass filter
    mask_lp = np.zeros((rows, cols), np.uint8)
    r = 30
    cv2.circle(mask_lp, (ccol, crow), r, 1, -1)
    
    # Terapkan filter
    f_shift_lp = f_shift * mask_lp
    f_ishift_lp = ifftshift(f_shift_lp)
    img_lp = ifft2(f_ishift_lp)
    img_lp = np.abs(img_lp)
    
    # High-pass filter
    mask_hp = np.ones((rows, cols), np.uint8)
    cv2.circle(mask_hp, (ccol, crow), r, 0, -1)
    
    f_shift_hp = f_shift * mask_hp
    f_ishift_hp = ifftshift(f_shift_hp)
    img_hp = ifft2(f_ishift_hp)
    img_hp = np.abs(img_hp)
    
    return {
        'original': gray,
        'magnitude': magnitude_spectrum,
        'phase': phase_spectrum,
        'lowpass': img_lp,
        'highpass': img_hp
    }

def frequency_filtering(image, filter_type='lowpass', cutoff=30):
    """Filter frekuensi (lowpass/highpass/bandpass)"""
    # Konversi ke grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # FFT
    f_transform = fft2(gray)
    f_shift = fftshift(f_transform)
    
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Buat mask berdasarkan jenis filter
    if filter_type == 'lowpass':
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), cutoff, 1, -1)
    elif filter_type == 'highpass':
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), cutoff, 0, -1)
    elif filter_type == 'bandpass':
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), cutoff + 20, 1, -1)
        cv2.circle(mask, (ccol, crow), cutoff, 0, -1)
    
    # Terapkan filter
    f_shift_filtered = f_shift * mask
    f_ishift = ifftshift(f_shift_filtered)
    img_filtered = ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)
    
    return img_filtered.astype(np.uint8), mask

def zero_padding(image, pad_size):
    """Zero padding pada gambar"""
    if len(image.shape) == 3:
        padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    else:
        padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
    
    return padded

def display_grayscale_image(image, caption):
    """Fungsi khusus untuk menampilkan gambar grayscale"""
    if len(image.shape) == 2:
        # Konversi grayscale ke RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        st.image(image_rgb, caption=caption)
    else:
        st.image(image, caption=caption)

def compare_images(img1, img2):
    """Membandingkan dua gambar"""
    # Resize gambar ke ukuran yang sama
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    
    # Konversi ke grayscale untuk perhitungan
    gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2GRAY) if len(img1_resized.shape) == 3 else img1_resized
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY) if len(img2_resized.shape) == 3 else img2_resized
    
    # Hitung MSE
    mse = np.mean((gray1 - gray2) ** 2)
    
    # Hitung PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Hitung SSIM
    from skimage.metrics import structural_similarity as ssim
    ssim_value = ssim(gray1, gray2)
    
    # Hitung histogram correlation
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Difference image
    diff = cv2.absdiff(img1_resized, img2_resized)
    
    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_value,
        'Correlation': correlation,
        'Difference': diff
    }

def main():
    st.markdown('<h1 class="main-title">🖼️ Aplikasi Pengolahan Citra Digital</h1>', unsafe_allow_html=True)
    
    # Sidebar untuk navigasi
    st.sidebar.title("📋 Menu Navigasi")
    mode = st.sidebar.selectbox(
        "Pilih Mode Pemrosesan:",
        ["🔍 Pemrosesan 1 Citra", "⚖️ Pemrosesan 2 Citra", "ℹ️ Informasi Aplikasi"]
    )
    
    if mode == "🔍 Pemrosesan 1 Citra":
        st.markdown('<h2 class="sub-title">Pemrosesan Satu Citra</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Pilih gambar untuk diproses",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Format yang didukung: PNG, JPG, JPEG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            # Load gambar
            image = load_image(uploaded_file)
            
            # Tampilkan gambar asli
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="Gambar Asli", use_column_width=True)
                st.write(f"**Dimensi:** {image.shape}")
                st.write(f"**Ukuran File:** {uploaded_file.size} bytes")
            
            with col2:
                # 9 OPSI PEMROSESAN (7 + 2 MODUL BARU)
                processing_options = st.multiselect(
                    "Pilih jenis pemrosesan yang diinginkan:",
                    [
                        "📊 Analisis RGB",
                        "👤 Deteksi Wajah",
                        "🔊 Penambahan & Pengurangan Noise",
                        "📐 Analisis Kontur & Bentuk",
                        "🗜️ Kompresi Citra",
                        "🎨 Konversi Ruang Warna",
                        "🧩 Analisis Tekstur",
                        "📈 Histogram & Operations",  # MODUL 2 BARU
                        "🔬 Filtering & Fourier Transform"  # MODUL 3 BARU
                    ]
                )
            
            # Proses berdasarkan pilihan
            if "📊 Analisis RGB" in processing_options:
                st.markdown('<h3 class="sub-title">📊 Analisis RGB</h3>', unsafe_allow_html=True)
                fig = rgb_analysis(image)
                if fig:
                    st.pyplot(fig)
                    
                    # Statistik RGB
                    if len(image.shape) == 3:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rata-rata Merah", f"{np.mean(image[:,:,0]):.2f}")
                            st.metric("Std Merah", f"{np.std(image[:,:,0]):.2f}")
                        with col2:
                            st.metric("Rata-rata Hijau", f"{np.mean(image[:,:,1]):.2f}")
                            st.metric("Std Hijau", f"{np.std(image[:,:,1]):.2f}")
                        with col3:
                            st.metric("Rata-rata Biru", f"{np.mean(image[:,:,2]):.2f}")
                            st.metric("Std Biru", f"{np.std(image[:,:,2]):.2f}")
            
            if "👤 Deteksi Wajah" in processing_options:
                st.markdown('<h3 class="sub-title">👤 Deteksi Wajah</h3>', unsafe_allow_html=True)
                try:
                    face_result, face_count = face_detection(image)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(face_result, caption=f"Hasil Deteksi - {face_count} wajah ditemukan")
                    with col2:
                        st.info(f"**Jumlah wajah terdeteksi:** {face_count}")
                        if face_count > 0:
                            st.success("✅ Wajah berhasil dideteksi!")
                        else:
                            st.warning("⚠️ Tidak ada wajah yang terdeteksi")
                except Exception as e:
                    st.error(f"Error dalam deteksi wajah: {str(e)}")
            
            if "🔊 Penambahan & Pengurangan Noise" in processing_options:
                st.markdown('<h3 class="sub-title">🔊 Penambahan & Pengurangan Noise</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Penambahan Noise")
                    noise_type = st.selectbox("Jenis Noise:", ['gaussian', 'salt_pepper', 'speckle'])
                    noisy_image = add_noise(image, noise_type)
                    st.image(noisy_image, caption=f"Gambar dengan Noise {noise_type.title()}")
                
                with col2:
                    st.subheader("Pengurangan Noise")
                    denoise_method = st.selectbox("Metode Denoising:", ['gaussian', 'median', 'bilateral', 'nlm'])
                    denoised = denoise_image(noisy_image, denoise_method)
                    st.image(denoised, caption=f"Hasil Denoising {denoise_method.title()}")
                
                # Perbandingan kualitas
                mse_noise = np.mean((image.astype(float) - noisy_image.astype(float)) ** 2)
                mse_denoise = np.mean((image.astype(float) - denoised.astype(float)) ** 2)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MSE Noise", f"{mse_noise:.2f}")
                with col2:
                    st.metric("MSE Setelah Denoising", f"{mse_denoise:.2f}")
            
            if "📐 Analisis Kontur & Bentuk" in processing_options:
                st.markdown('<h3 class="sub-title">📐 Analisis Kontur & Bentuk</h3>', unsafe_allow_html=True)
                contour_result, shapes_info = contour_analysis(image)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(contour_result, caption="Deteksi Kontur")
                with col2:
                    st.subheader("Informasi Bentuk")
                    for i, shape in enumerate(shapes_info):
                        st.write(f"**Objek {i+1}:**")
                        st.write(f"- Bentuk: {shape['Bentuk']}")
                        st.write(f"- Luas: {shape['Luas']:.2f}")
                        st.write(f"- Keliling: {shape['Keliling']:.2f}")
                        st.write(f"- Jumlah Sisi: {shape['Jumlah Sisi']}")
                        st.write("---")
            
            if "🗜️ Kompresi Citra" in processing_options:
                st.markdown('<h3 class="sub-title">🗜️ Kompresi Citra</h3>', unsafe_allow_html=True)
                
                quality = st.slider("Kualitas Kompresi (%):", 10, 100, 50)
                compressed = image_compression(image, quality)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(image, caption="Gambar Asli")
                with col2:
                    st.image(compressed, caption=f"Kompresi {quality}%")
                with col3:
                    # Hitung ukuran file
                    original_size = uploaded_file.size
                    
                    # Estimate compressed size
                    pil_img = Image.fromarray(compressed)
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format='JPEG', quality=quality)
                    compressed_size = len(buffer.getvalue())
                    
                    compression_ratio = (1 - compressed_size/original_size) * 100
                    
                    st.metric("Ukuran Asli", f"{original_size} bytes")
                    st.metric("Ukuran Kompresi", f"{compressed_size} bytes")
                    st.metric("Rasio Kompresi", f"{compression_ratio:.1f}%")
            
            if "🎨 Konversi Ruang Warna" in processing_options:
                st.markdown('<h3 class="sub-title">🎨 Konversi Ruang Warna</h3>', unsafe_allow_html=True)
                color_results = color_space_conversion(image)
                
                if color_results:
                    # Tampilkan dalam grid
                    cols = st.columns(2)
                    for i, (space_name, converted_img) in enumerate(color_results.items()):
                        with cols[i % 2]:
                            if space_name == 'Grayscale':
                                display_grayscale_image(converted_img, f"Ruang Warna: {space_name}")
                            else:
                                st.image(converted_img, caption=f"Ruang Warna: {space_name}")
            
            if "🧩 Analisis Tekstur" in processing_options:
                st.markdown('<h3 class="sub-title">🧩 Analisis Tekstur</h3>', unsafe_allow_html=True)
                texture_results = texture_analysis(image)
                
                col1, col2 = st.columns(2)
                with col1:
                    # Tampilkan gambar grayscale
                    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
                    display_grayscale_image(gray_img, "Gambar Grayscale untuk Analisis")
                
                with col2:
                    st.subheader("Properti Tekstur GLCM")
                    for prop_name, values in texture_results.items():
                        mean_value = np.mean(values)
                        st.metric(f"{prop_name.title()}", f"{mean_value:.4f}")
                
                # Visualisasi hasil GLCM
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                properties = list(texture_results.keys())
                
                for i, prop in enumerate(properties):
                    ax = axes[i//2, i%2]
                    values = texture_results[prop]
                    im = ax.imshow(values, cmap='viridis')
                    ax.set_title(f'GLCM {prop.title()}')
                    plt.colorbar(im, ax=ax)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # ============= MODUL 2 BARU =============
            if "📈 Modul 2: Histogram & Operations" in processing_options:
                st.markdown("""
                <div class="module-box">
                <h3>📈 Modul 2: Histogram & Operations</h3>
                <p>Operasi aritmatika, logika, histogram, dan manipulasi citra lanjutan</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Sub-opsi untuk Modul 2
                module2_sub = st.selectbox(
                    "Pilih operasi Histogram & Operations:",
                    ["📊 Operasi Histogram", "📈 Analisis Histogram Lanjutan", "⚖️ Equalization & Specification"],
                    key="module2_sub"
                )
                
                if module2_sub == "📊 Operasi Histogram":
                    st.subheader("Operasi Histogram")
                    hist_results = histogram_operations(image)
                    
                    # Tampilkan hasil dalam grid 2x2
                    col1, col2 = st.columns(2)
                    row1_items = list(hist_results.items())[:2]
                    row2_items = list(hist_results.items())[2:]
                    
                    for i, (name, data) in enumerate(row1_items):
                        with col1 if i == 0 else col2:
                            st.write(f"**{name.title()}**")
                            display_grayscale_image(data['image'], f"Hasil {name}")
                    
                    for i, (name, data) in enumerate(row2_items):
                        with col1 if i == 0 else col2:
                            st.write(f"**{name.title()}**")
                            display_grayscale_image(data['image'], f"Hasil {name}")
                    
                    # Plot histogram perbandingan
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    axes = axes.ravel()
                    for i, (name, data) in enumerate(hist_results.items()):
                        axes[i].plot(data['hist'])
                        axes[i].set_title(f'Histogram {name.title()}')
                        axes[i].set_xlabel('Intensitas')
                        axes[i].set_ylabel('Frekuensi')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                elif module2_sub == "📈 Analisis Histogram Lanjutan":
                    st.subheader("Analisis Histogram Lanjutan")
                    fig = advanced_histogram_analysis(image)
                    if fig:
                        st.pyplot(fig)
                    
                    # Statistik tambahan
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
                    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist_norm = hist / hist.sum()
                    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Entropy", f"{entropy:.3f}")
                    with col2:
                        st.metric("Contrast", f"{np.std(gray):.2f}")
                    with col3:
                        st.metric("Brightness", f"{np.mean(gray):.2f}")
                    with col4:
                        st.metric("Dynamic Range", f"{np.max(gray) - np.min(gray)}")
            
            # ============= MODUL 3 BARU =============
            if "🔬 Modul 3: Filtering & Fourier Transform" in processing_options:
                st.markdown("""
                <div class="module-box">
                <h3>🔬 Modul 3: Filtering & Fourier Transform</h3>
                <p>Konvolusi, padding, filter frekuensi, transformasi fourier, dan noise reduction</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Sub-opsi untuk Modul 3
                module3_sub = st.selectbox(
                    "Pilih operasi Filtering & Fourier Transform:",
                    [
                        "🔧 Zero Padding", 
                        "🌊 Convolution Filters", 
                        "📻 Frequency Filtering", 
                        "🔬 Fourier Transform Analysis"
                    ],
                    key="module3_sub"
                )
                
                if module3_sub == "🔧 Zero Padding":
                    st.subheader("Zero Padding")
                    pad_size = st.slider("Ukuran Padding:", 10, 100, 30, key="padding_slider")
                    padded = zero_padding(image, pad_size)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Gambar Asli")
                        st.write(f"Ukuran asli: {image.shape}")
                    with col2:
                        st.image(padded, caption=f"Hasil Zero Padding ({pad_size}px)")
                        st.write(f"Ukuran setelah padding: {padded.shape}")
                
                elif module3_sub == "🌊 Convolution Filters":
                    st.subheader("Filter Konvolusi")
                    conv_results = convolution_filters(image)
                    
                    # Tampilkan hasil dalam grid 3x2
                    cols = st.columns(3)
                    for i, (filter_name, result) in enumerate(conv_results.items()):
                        with cols[i % 3]:
                            display_grayscale_image(result, f"Filter {filter_name}")
                
                elif module3_sub == "📻 Frequency Filtering":
                    st.subheader("Filter Frekuensi")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        filter_type = st.selectbox("Jenis Filter:", ['lowpass', 'highpass', 'bandpass'], key="freq_filter")
                    with col2:
                        cutoff = st.slider("Cutoff Frequency:", 10, 100, 30, key="freq_cutoff")
                    
                    filtered_img, mask = frequency_filtering(image, filter_type, cutoff)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        display_grayscale_image(mask * 255, f"Mask {filter_type.title()}")
                    with col2:
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
                        display_grayscale_image(gray, "Original")
                    with col3:
                        display_grayscale_image(filtered_img, f"Hasil {filter_type.title()}")
                
                elif module3_sub == "🔬 Fourier Transform Analysis":
                    st.subheader("Analisis Transformasi Fourier")
                    fourier_results = fourier_transform_analysis(image)
                    
                    # Tampilkan hasil
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        display_grayscale_image(fourier_results['original'], "Gambar Asli")
                        display_grayscale_image(fourier_results['lowpass'].astype(np.uint8), "Low-pass Filter")
                    
                    with col2:
                        # Untuk magnitude spectrum menggunakan matplotlib
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.imshow(fourier_results['magnitude'], cmap='gray')
                        ax.set_title('Magnitude Spectrum')
                        ax.axis('off')
                        st.pyplot(fig)
                        
                        display_grayscale_image(fourier_results['highpass'].astype(np.uint8), "High-pass Filter")
                    
                    # Informasi tambahan
                    st.subheader("Informasi Spektrum")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("DC Component", f"{np.abs(fourier_results['magnitude'][fourier_results['magnitude'].shape[0]//2, fourier_results['magnitude'].shape[1]//2]):.2f}")
                    with col2:
                        st.metric("Max Frequency", f"{np.max(fourier_results['magnitude']):.2f}")
                    with col3:
                        st.metric("Energy", f"{np.sum(fourier_results['magnitude']**2):.0f}")
    
    elif mode == "⚖️ Pemrosesan 2 Citra":
        st.markdown('<h2 class="sub-title">Pemrosesan Dua Citra</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gambar Pertama")
            uploaded_file1 = st.file_uploader(
                "Pilih gambar pertama",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key="img1"
            )
        
        with col2:
            st.subheader("Gambar Kedua")
            uploaded_file2 = st.file_uploader(
                "Pilih gambar kedua",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key="img2"
            )
        
        if uploaded_file1 is not None and uploaded_file2 is not None:
            image1 = load_image(uploaded_file1)
            image2 = load_image(uploaded_file2)
            
            # Tampilkan gambar
            col1, col2 = st.columns(2)
            with col1:
                st.image(image1, caption="Gambar 1", use_column_width=True)
            with col2:
                st.image(image2, caption="Gambar 2", use_column_width=True)
            
            # Pilihan operasi
            operation = st.selectbox(
                "Pilih operasi untuk kedua gambar:",
                [
                    "📊 Perbandingan Statistik",
                    "🔍 Analisis Kemiripan",
                    "➕ Operasi Aritmatika",
                    "🎭 Blending",
                    "📈 Perbandingan Histogram"
                ]
            )
            
            if operation == "📊 Perbandingan Statistik":
                st.markdown('<h3 class="sub-title">📊 Perbandingan Statistik</h3>', unsafe_allow_html=True)
                
                # Resize gambar untuk perbandingan
                h, w = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
                img1_resized = cv2.resize(image1, (w, h))
                img2_resized = cv2.resize(image2, (w, h))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Gambar 1")
                    st.write(f"Dimensi: {image1.shape}")
                    st.write(f"Rata-rata: {np.mean(img1_resized):.2f}")
                    st.write(f"Std Dev: {np.std(img1_resized):.2f}")
                    st.write(f"Min: {np.min(img1_resized)}")
                    st.write(f"Max: {np.max(img1_resized)}")
                
                with col2:
                    st.subheader("Gambar 2")
                    st.write(f"Dimensi: {image2.shape}")
                    st.write(f"Rata-rata: {np.mean(img2_resized):.2f}")
                    st.write(f"Std Dev: {np.std(img2_resized):.2f}")
                    st.write(f"Min: {np.min(img2_resized)}")
                    st.write(f"Max: {np.max(img2_resized)}")
                
                with col3:
                    st.subheader("Perbedaan")
                    diff_mean = abs(np.mean(img1_resized) - np.mean(img2_resized))
                    diff_std = abs(np.std(img1_resized) - np.std(img2_resized))
                    st.write(f"Selisih Rata-rata: {diff_mean:.2f}")
                    st.write(f"Selisih Std Dev: {diff_std:.2f}")
            
            elif operation == "🔍 Analisis Kemiripan":
                st.markdown('<h3 class="sub-title">🔍 Analisis Kemiripan</h3>', unsafe_allow_html=True)
                
                comparison_results = compare_images(image1, image2)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MSE (Mean Squared Error)", f"{comparison_results['MSE']:.2f}")
                    st.metric("PSNR (Peak Signal-to-Noise Ratio)", f"{comparison_results['PSNR']:.2f} dB")
                    
                with col2:
                    st.metric("SSIM (Structural Similarity)", f"{comparison_results['SSIM']:.4f}")
                    st.metric("Korelasi Histogram", f"{comparison_results['Correlation']:.4f}")
                
                # Tampilkan difference image
                st.subheader("Gambar Perbedaan")
                st.image(comparison_results['Difference'], caption="Perbedaan antara kedua gambar")
                
                # Interpretasi hasil
                st.subheader("Interpretasi Hasil")
                if comparison_results['SSIM'] > 0.9:
                    st.success("✅ Gambar sangat mirip")
                elif comparison_results['SSIM'] > 0.7:
                    st.info("ℹ️ Gambar cukup mirip")
                else:
                    st.warning("⚠️ Gambar berbeda signifikan")
            
            elif operation == "➕ Operasi Aritmatika":
                st.markdown('<h3 class="sub-title">➕ Operasi Aritmatika</h3>', unsafe_allow_html=True)
                
                # Resize gambar ke ukuran yang sama
                h, w = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
                img1_resized = cv2.resize(image1, (w, h))
                img2_resized = cv2.resize(image2, (w, h))
                
                arithmetic_op = st.selectbox(
                    "Pilih operasi aritmatika:",
                    ["Penjumlahan", "Pengurangan", "Perkalian", "Pembagian"]
                )
                
                if arithmetic_op == "Penjumlahan":
                    result = cv2.add(img1_resized, img2_resized)
                elif arithmetic_op == "Pengurangan":
                    result = cv2.subtract(img1_resized, img2_resized)
                elif arithmetic_op == "Perkalian":
                    result = cv2.multiply(img1_resized.astype(np.float32)/255, img2_resized.astype(np.float32)/255) * 255
                    result = result.astype(np.uint8)
                elif arithmetic_op == "Pembagian":
                    result = cv2.divide(img1_resized.astype(np.float32), img2_resized.astype(np.float32) + 1) * 255
                    result = result.astype(np.uint8)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(img1_resized, caption="Gambar 1")
                with col2:
                    st.image(img2_resized, caption="Gambar 2")
                with col3:
                    st.image(result, caption=f"Hasil {arithmetic_op}")
            
            elif operation == "🎭 Blending":
                st.markdown('<h3 class="sub-title">🎭 Blending</h3>', unsafe_allow_html=True)
                
                # Resize gambar ke ukuran yang sama
                h, w = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
                img1_resized = cv2.resize(image1, (w, h))
                img2_resized = cv2.resize(image2, (w, h))
                
                alpha = st.slider("Bobot Gambar 1 (Alpha):", 0.0, 1.0, 0.5, 0.1)
                beta = 1.0 - alpha
                
                blended = cv2.addWeighted(img1_resized, alpha, img2_resized, beta, 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(img1_resized, caption=f"Gambar 1 (α={alpha:.1f})")
                with col2:
                    st.image(img2_resized, caption=f"Gambar 2 (β={beta:.1f})")
                with col3:
                    st.image(blended, caption="Hasil Blending")
            
            elif operation == "📈 Perbandingan Histogram":
                st.markdown('<h3 class="sub-title">📈 Perbandingan Histogram</h3>', unsafe_allow_html=True)
                
                # Resize gambar ke ukuran yang sama
                h, w = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
                img1_resized = cv2.resize(image1, (w, h))
                img2_resized = cv2.resize(image2, (w, h))
                
                # Konversi ke grayscale untuk histogram
                gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2GRAY) if len(img1_resized.shape) == 3 else img1_resized
                gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY) if len(img2_resized.shape) == 3 else img2_resized
                
                # Plot histogram
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.hist(gray1.ravel(), bins=256, alpha=0.7, label='Gambar 1', color='blue')
                ax1.hist(gray2.ravel(), bins=256, alpha=0.7, label='Gambar 2', color='red')
                ax1.set_title('Overlay Histogram')
                ax1.set_xlabel('Intensitas Pixel')
                ax1.set_ylabel('Frekuensi')
                ax1.legend()
                
                # Histogram terpisah
                ax2.plot(cv2.calcHist([gray1], [0], None, [256], [0, 256]), color='blue', label='Gambar 1')
                ax2.plot(cv2.calcHist([gray2], [0], None, [256], [0, 256]), color='red', label='Gambar 2')
                ax2.set_title('Perbandingan Histogram')
                ax2.set_xlabel('Intensitas Pixel')
                ax2.set_ylabel('Frekuensi')
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Metrik perbandingan histogram
                hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
                
                correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
                intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
                bhattacharyya = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Korelasi", f"{correlation:.4f}")
                    st.metric("Chi-Square", f"{chi_square:.4f}")
                with col2:
                    st.metric("Interseksi", f"{intersection:.4f}")
                    st.metric("Bhattacharyya", f"{bhattacharyya:.4f}")
    
    else:  # Informasi Aplikasi
        st.markdown('<h2 class="sub-title">ℹ️ Informasi Aplikasi</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Tentang Aplikasi</h3>
        <p>Aplikasi Pengolahan Citra Digital yang lengkap dengan 9 modul pemrosesan untuk pembelajaran 
        komprehensif dalam bidang computer vision dan image processing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔧 9 Modul Pemrosesan Lengkap
        
        #### 🔍 Pemrosesan 1 Citra (9 Opsi):
        1. **📊 Analisis RGB**: Analisis komponen warna merah, hijau, dan biru
        2. **👤 Deteksi Wajah**: Deteksi wajah menggunakan Haar Cascade
        3. **🔊 Penambahan & Pengurangan Noise**: Gaussian, Salt & Pepper, Speckle
        4. **📐 Analisis Kontur & Bentuk**: Deteksi dan klasifikasi bentuk
        5. **🗜️ Kompresi Citra**: Kompresi JPEG dengan berbagai kualitas
        6. **🎨 Konversi Ruang Warna**: RGB, HSV, LAB, YUV, Grayscale
        7. **🧩 Analisis Tekstur**: Analisis tekstur menggunakan GLCM
        8. **📈 Modul 2: Histogram & Operations**: Operasi histogram, equalization, analisis lanjutan
        9. **🔬 Modul 3: Filtering & Fourier Transform**: Zero padding, konvolusi, frequency filtering, FFT
        
        #### ⚖️ Pemrosesan 2 Citra:
        - **📊 Perbandingan Statistik**: Membandingkan properti statistik
        - **🔍 Analisis Kemiripan**: MSE, PSNR, SSIM, dan korelasi
        - **➕ Operasi Aritmatika**: Penjumlahan, pengurangan, perkalian, pembagian
        - **🎭 Blending**: Pencampuran dua gambar dengan bobot tertentu
        - **📈 Perbandingan Histogram**: Analisis distribusi intensitas pixel
        """)
        
        st.markdown("""
        ### 🆕 Fitur Baru Modul 2 & 3
        
        #### 📈 Modul 2: Histogram & Operations
        - **Operasi Histogram**: Equalization, CLAHE, Stretching
        - **Analisis Histogram Lanjutan**: Entropy, statistik, CDF
        - **Equalization & Specification**: Penyamaan histogram
        
        #### 🔬 Modul 3: Filtering & Fourier Transform  
        - **Zero Padding**: Penambahan padding gambar
        - **Convolution Filters**: Sobel, Laplacian, Sharpen, Edge Detection
        - **Frequency Filtering**: Low-pass, High-pass, Band-pass
        - **Fourier Transform**: FFT analysis, magnitude/phase spectrum
        """)
        
        st.markdown("""
        ### 🛠️ Teknologi yang Digunakan
        - **Streamlit**: Framework untuk antarmuka web
        - **OpenCV**: Library untuk computer vision
        - **PIL/Pillow**: Pengolahan gambar
        - **NumPy**: Komputasi numerik
        - **Matplotlib/Seaborn**: Visualisasi data
        - **Scikit-image**: Algoritma pengolahan citra
        - **SciPy**: Transformasi Fourier dan filter digital
        """)
        
        st.success("✅ Aplikasi lengkap dengan 9 modul pemrosesan siap digunakan!")

if __name__ == "__main__":
    main()