import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import io
import base64
from skimage import feature, measure, filters, segmentation
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from scipy import ndimage
from scipy.stats import entropy
import pandas as pd
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Pengolahan Citra Digital Pro",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .module-header {
        font-size: 1.5rem;
        color: #2e8b57;
        font-weight: bold;
        margin: 1rem 0;
        padding: 0.5rem;
        background-color: #f0f8f0;
        border-left: 4px solid #2e8b57;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class ImageProcessor:
    def __init__(self):
        pass
    
    # ========== MODUL 1: ARRAY RGB DAN INFO DASAR ==========
    def get_rgb_info(self, image):
        """Mendapatkan informasi RGB dari citra"""
        if len(image.shape) == 3:
            b, g, r = cv2.split(image)
            rgb_array = cv2.merge([r, g, b])
        else:
            rgb_array = image
            
        info = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'size': image.size,
            'min_val': np.min(image),
            'max_val': np.max(image),
            'mean_val': np.mean(image),
            'std_val': np.std(image)
        }
        
        return rgb_array, info
    
    # ========== MODUL 4: FACE DETECTION + NOISE + SHARPENING ==========
    def detect_faces(self, image):
        """Deteksi wajah menggunakan Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade (menggunakan default OpenCV)
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            result_image = image.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(result_image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
            return result_image, len(faces)
        except:
            return image, 0
    
    def add_salt_pepper_noise(self, image, noise_ratio=0.05):
        """Menambahkan noise salt and pepper"""
        noisy = image.copy()
        h, w = image.shape[:2]
        
        # Salt noise (white pixels)
        num_salt = int(noise_ratio * h * w * 0.5)
        coords = [np.random.randint(0, i-1, num_salt) for i in [h, w]]
        if len(image.shape) == 3:
            noisy[coords[0], coords[1], :] = 255
        else:
            noisy[coords[0], coords[1]] = 255
            
        # Pepper noise (black pixels)
        num_pepper = int(noise_ratio * h * w * 0.5)
        coords = [np.random.randint(0, i-1, num_pepper) for i in [h, w]]
        if len(image.shape) == 3:
            noisy[coords[0], coords[1], :] = 0
        else:
            noisy[coords[0], coords[1]] = 0
            
        return noisy
    
    def remove_noise(self, image):
        """Menghilangkan noise menggunakan median filter"""
        if len(image.shape) == 3:
            return cv2.medianBlur(image, 5)
        else:
            return cv2.medianBlur(image, 5)
    
    def sharpen_image(self, image):
        """Menajamkan citra menggunakan unsharp masking"""
        # Kernel sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    # ========== MODUL 5: KONTUR DAN FITUR BENTUK ==========
    def process_contours(self, image):
        """Pemrosesan kontur dan ekstraksi fitur bentuk"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Deteksi tepi Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Mencari kontur
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Gambar kontur
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        
        # Ekstraksi fitur bentuk
        features = []
        for i, contour in enumerate(contours[:5]):  # Ambil 5 kontur terbesar
            if len(contour) > 5:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Moments
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx, cy = 0, 0
                    
                    features.append({
                        'contour_id': i,
                        'area': area,
                        'perimeter': perimeter,
                        'circularity': circularity,
                        'centroid_x': cx,
                        'centroid_y': cy
                    })
        
        return contour_image, edges, features
    
    def compute_projection_profile(self, image):
        """Menghitung proyeksi integral (horizontal dan vertikal)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Proyeksi horizontal (sum setiap baris)
        horizontal_projection = np.sum(gray, axis=1)
        
        # Proyeksi vertikal (sum setiap kolom)
        vertical_projection = np.sum(gray, axis=0)
        
        return horizontal_projection, vertical_projection
    
    # ========== MODUL 6: KOMPRESI CITRA ==========
    def compress_image_quality(self, image, quality_levels=[95, 75, 50, 25]):
        """Kompresi JPEG dengan berbagai level kualitas"""
        compressed_images = {}
        compression_ratios = {}
        
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Original size
        original_buffer = io.BytesIO()
        pil_image.save(original_buffer, format='PNG')
        original_size = len(original_buffer.getvalue())
        
        for quality in quality_levels:
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality)
            compressed_size = len(buffer.getvalue())
            
            # Load compressed image
            buffer.seek(0)
            compressed_pil = Image.open(buffer)
            compressed_array = np.array(compressed_pil)
            
            if len(image.shape) == 3 and len(compressed_array.shape) == 3:
                compressed_images[quality] = cv2.cvtColor(compressed_array, cv2.COLOR_RGB2BGR)
            else:
                compressed_images[quality] = compressed_array
                
            compression_ratios[quality] = original_size / compressed_size
        
        return compressed_images, compression_ratios
    
    def run_length_encoding(self, image):
        """Implementasi sederhana Run-Length Encoding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Flatten image
        flat = gray.flatten()
        
        # RLE encoding
        encoded = []
        current_pixel = flat[0]
        count = 1
        
        for pixel in flat[1:]:
            if pixel == current_pixel:
                count += 1
            else:
                encoded.append((current_pixel, count))
                current_pixel = pixel
                count = 1
        encoded.append((current_pixel, count))
        
        # Calculate compression ratio
        original_size = len(flat)
        compressed_size = len(encoded) * 2  # (pixel_value, count) pairs
        compression_ratio = original_size / compressed_size
        
        return encoded[:10], compression_ratio  # Return first 10 pairs as example
    
    # ========== MODUL 7: RUANG WARNA ==========
    def convert_color_spaces(self, image):
        """Konversi ke berbagai ruang warna"""
        color_spaces = {}
        
        if len(image.shape) == 3:
            # RGB (default)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_spaces['RGB'] = rgb
            
            # HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            color_spaces['HSV'] = hsv
            
            # LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            color_spaces['LAB'] = lab
            
            # YUV
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            color_spaces['YUV'] = yuv
            
            # YCrCb
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            color_spaces['YCrCb'] = ycrcb
        else:
            color_spaces['Grayscale'] = image
            
        return color_spaces
    
    # ========== ANALISIS TEKSTUR ==========
    def compute_glcm_features(self, image):
        """Menghitung fitur GLCM (Gray Level Co-occurrence Matrix)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Resize jika terlalu besar untuk efisiensi
        if gray.shape[0] > 256 or gray.shape[1] > 256:
            gray = cv2.resize(gray, (256, 256))
        
        # Quantize to reduce computation
        gray = (gray // 32) * 32  # Reduce to 8 levels
        
        # Compute GLCM
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(gray, distances, angles, levels=256, symmetric=True, normed=True)
        
        # Calculate features
        features = {
            'contrast': graycoprops(glcm, 'contrast')[0, 0],
            'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
            'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
            'energy': graycoprops(glcm, 'energy')[0, 0],
            'correlation': graycoprops(glcm, 'correlation')[0, 0],
            'ASM': graycoprops(glcm, 'ASM')[0, 0]
        }
        
        return features
    
    def compute_lbp_features(self, image):
        """Menghitung Local Binary Pattern"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Resize jika terlalu besar
        if gray.shape[0] > 256 or gray.shape[1] > 256:
            gray = cv2.resize(gray, (256, 256))
        
        # Compute LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate histogram
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        return lbp, hist
    
    def compute_gabor_features(self, image):
        """Menghitung fitur Gabor"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Resize jika terlalu besar
        if gray.shape[0] > 256 or gray.shape[1] > 256:
            gray = cv2.resize(gray, (256, 256))
        
        # Gabor filter parameters
        frequencies = [0.1, 0.3, 0.5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        gabor_responses = []
        gabor_features = {}
        
        for freq in frequencies:
            for angle in angles:
                filtered_real, filtered_imag = gabor(gray, frequency=freq, theta=angle)
                gabor_responses.append(filtered_real)
                
                # Calculate energy
                energy = np.sum(filtered_real**2)
                key = f"freq_{freq:.1f}_angle_{int(np.degrees(angle))}"
                gabor_features[key] = energy
        
        return gabor_responses[:4], gabor_features  # Return first 4 for display
    
    def compute_first_order_statistics(self, image):
        """Menghitung statistik orde pertama"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        features = {
            'mean': np.mean(gray),
            'variance': np.var(gray),
            'standard_deviation': np.std(gray),
            'skewness': self._skewness(gray),
            'kurtosis': self._kurtosis(gray),
            'entropy': self._entropy(gray)
        }
        
        return features
    
    def _skewness(self, image):
        """Menghitung skewness"""
        mean = np.mean(image)
        std = np.std(image)
        return np.mean(((image - mean) / std) ** 3)
    
    def _kurtosis(self, image):
        """Menghitung kurtosis"""
        mean = np.mean(image)
        std = np.std(image)
        return np.mean(((image - mean) / std) ** 4) - 3
    
    def _entropy(self, image):
        """Menghitung entropy"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist[hist > 0]  # Remove zeros
        prob = hist / np.sum(hist)
        return -np.sum(prob * np.log2(prob))

def main():
    st.markdown('<h1 class="main-header">🖼️ Pengolahan Citra Digital Profesional</h1>', unsafe_allow_html=True)
    
    # Sidebar untuk kontrol
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        st.markdown("---")
        
        # Upload gambar
        st.subheader("📁 Upload Gambar")
        uploaded_files = st.file_uploader(
            "Pilih 1-3 gambar",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload maksimal 3 gambar untuk diproses"
        )
        
        if len(uploaded_files) > 3:
            st.error("⚠️ Maksimal 3 gambar!")
            uploaded_files = uploaded_files[:3]
    
    # Main content
    if uploaded_files:
        # Load images
        images = []
        for file in uploaded_files:
            image = Image.open(file)
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            images.append(opencv_image)
        
        # Preview uploaded images
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📸 Gambar yang diupload:")
        cols = st.columns(len(images))
        for i, (img, col) in enumerate(zip(images, cols)):
            with col:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(rgb_img, caption=f"Gambar {i+1}", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tombol proses
        if st.button("🚀 OLAH CITRA", type="primary", use_container_width=True):
            processor = ImageProcessor()
            
            with st.spinner("🔄 Memproses citra... Harap tunggu!"):
                # Process semua gambar
                all_results = []
                for i, image in enumerate(images):
                    st.markdown(f"### 🖼️ Hasil Pengolahan Gambar {i+1}")
                    
                    # Create tabs untuk setiap modul
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        "📊 RGB & Info", "👤 Face & Noise", "🔍 Kontur & Bentuk", 
                        "📦 Kompresi", "🎨 Ruang Warna", "🔬 Tekstur"
                    ])
                    
                    # ========== TAB 1: RGB & INFO ==========
                    with tab1:
                        st.markdown('<div class="module-header">📊 Modul 1: Array RGB dan Informasi Citra</div>', unsafe_allow_html=True)
                        
                        rgb_array, info = processor.get_rgb_info(image)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("🖼️ Citra RGB")
                            st.image(cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB), use_column_width=True)
                        
                        with col2:
                            st.subheader("📋 Informasi Citra")
                            info_df = pd.DataFrame([
                                ["Dimensi", f"{info['shape']}"],
                                ["Tipe Data", info['dtype']],
                                ["Ukuran Total", f"{info['size']:,} piksel"],
                                ["Nilai Minimum", f"{info['min_val']:.2f}"],
                                ["Nilai Maksimum", f"{info['max_val']:.2f}"],
                                ["Rata-rata", f"{info['mean_val']:.2f}"],
                                ["Deviasi Standar", f"{info['std_val']:.2f}"]
                            ], columns=["Parameter", "Nilai"])
                            st.dataframe(info_df, use_container_width=True)
                        
                        # Histogram RGB
                        if len(image.shape) == 3:
                            st.subheader("📈 Histogram RGB")
                            fig, ax = plt.subplots(figsize=(10, 4))
                            colors = ['red', 'green', 'blue']
                            for i, color in enumerate(colors):
                                hist = cv2.calcHist([rgb_array], [i], None, [256], [0, 256])
                                ax.plot(hist, color=color, alpha=0.7, label=f'Channel {color.upper()}')
                            ax.set_xlabel('Intensitas Pixel')
                            ax.set_ylabel('Frekuensi')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close()
                    
                    # ========== TAB 2: FACE & NOISE ==========
                    with tab2:
                        st.markdown('<div class="module-header">👤 Modul 4: Deteksi Wajah, Noise, dan Sharpening</div>', unsafe_allow_html=True)
                        
                        # Face detection
                        face_result, num_faces = processor.detect_faces(image)
                        
                        # Noise processing
                        noisy_image = processor.add_salt_pepper_noise(image)
                        denoised_image = processor.remove_noise(noisy_image)
                        sharpened_image = processor.sharpen_image(image)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader(f"👤 Deteksi Wajah ({num_faces} wajah)")
                            st.image(cv2.cvtColor(face_result, cv2.COLOR_BGR2RGB), use_column_width=True)
                            
                            st.subheader("🔧 Citra dengan Noise")
                            st.image(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                        
                        with col2:
                            st.subheader("✨ Citra Denoised")
                            st.image(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                            
                            st.subheader("🔪 Citra Sharpened")
                            st.image(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    # ========== TAB 3: KONTUR & BENTUK ==========
                    with tab3:
                        st.markdown('<div class="module-header">🔍 Modul 5: Kontur dan Fitur Bentuk</div>', unsafe_allow_html=True)
                        
                        # Contour processing
                        contour_image, edges, contour_features = processor.process_contours(image)
                        h_proj, v_proj = processor.compute_projection_profile(image)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("🔍 Deteksi Kontur")
                            st.image(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                            
                            st.subheader("⚡ Deteksi Tepi Canny")
                            st.image(edges, use_column_width=True, clamp=True)
                        
                        with col2:
                            if contour_features:
                                st.subheader("📊 Fitur Bentuk Kontur")
                                features_df = pd.DataFrame(contour_features)
                                st.dataframe(features_df, use_container_width=True)
                            
                            # Projection profiles
                            st.subheader("📈 Proyeksi Integral")
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                            ax1.plot(h_proj)
                            ax1.set_title('Proyeksi Horizontal')
                            ax1.set_xlabel('Posisi Y')
                            ax1.set_ylabel('Intensitas')
                            
                            ax2.plot(v_proj)
                            ax2.set_title('Proyeksi Vertikal')
                            ax2.set_xlabel('Posisi X')
                            ax2.set_ylabel('Intensitas')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                    
                    # ========== TAB 4: KOMPRESI ==========
                    with tab4:
                        st.markdown('<div class="module-header">📦 Modul 6: Kompresi Citra</div>', unsafe_allow_html=True)
                        
                        # Compression
                        compressed_imgs, compression_ratios = processor.compress_image_quality(image)
                        rle_encoded, rle_ratio = processor.run_length_encoding(image)
                        
                        st.subheader("📊 Kompresi JPEG dengan Berbagai Kualitas")
                        
                        # Display compression results
                        quality_cols = st.columns(4)
                        for idx, (quality, img) in enumerate(compressed_imgs.items()):
                            with quality_cols[idx]:
                                st.write(f"**Kualitas {quality}%**")
                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, 
                                        use_column_width=True)
                                st.write(f"Rasio: {compression_ratios[quality]:.2f}x")
                        
                        # Compression analysis
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("📈 Analisis Kompresi")
                            compression_df = pd.DataFrame([
                                ["JPEG 95%", f"{compression_ratios[95]:.2f}x"],
                                ["JPEG 75%", f"{compression_ratios[75]:.2f}x"],
                                ["JPEG 50%", f"{compression_ratios[50]:.2f}x"],
                                ["JPEG 25%", f"{compression_ratios[25]:.2f}x"],
                                ["RLE (estimasi)", f"{rle_ratio:.2f}x"]
                            ], columns=["Metode", "Rasio Kompresi"])
                            st.dataframe(compression_df, use_container_width=True)
                        
                        with col2:
                            st.subheader("🔢 Contoh Run-Length Encoding")
                            rle_df = pd.DataFrame(rle_encoded, columns=["Nilai Pixel", "Jumlah"])
                            st.dataframe(rle_df.head(10), use_container_width=True)
                            st.caption(f"Rasio kompresi RLE: {rle_ratio:.2f}x")
                    
                    # ========== TAB 5: RUANG WARNA ==========
                    with tab5:
                        st.markdown('<div class="module-header">🎨 Modul 7: Transformasi Ruang Warna</div>', unsafe_allow_html=True)
                        
                        color_spaces = processor.convert_color_spaces(image)
                        
                        st.subheader("🌈 Berbagai Ruang Warna")
                        
                        # Display color spaces
                        space_names = list(color_spaces.keys())
                        num_spaces = len(space_names)
                        cols_per_row = 3
                        
                        for row in range(0, num_spaces, cols_per_row):
                            cols = st.columns(cols_per_row)
                            for col_idx in range(cols_per_row):
                                space_idx = row + col_idx
                                if space_idx < num_spaces:
                                    space_name = space_names[space_idx]
                                    space_img = color_spaces[space_name]
                                    
                                    with cols[col_idx]:
                                        st.write(f"**{space_name}**")
                                        if len(space_img.shape) == 3:
                                            # For color images, show RGB version
                                            if space_name == 'RGB':
                                                st.image(space_img, use_column_width=True)
                                            else:
                                                st.image(cv2.cvtColor(space_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                                        else:
                                            st.image(space_img, use_column_width=True, clamp=True)
                        
                        # Color space analysis
                        if len(image.shape) == 3:
                            st.subheader("📊 Analisis Channel Warna")
                            
                            selected_space = st.selectbox("Pilih ruang warna untuk analisis:", 
                                                        options=list(color_spaces.keys()))
                            
                            if selected_space in color_spaces and len(color_spaces[selected_space].shape) == 3:
                                space_img = color_spaces[selected_space]
                                
                                # Channel analysis
                                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                                channel_names = {
                                    'RGB': ['Red', 'Green', 'Blue'],
                                    'HSV': ['Hue', 'Saturation', 'Value'],
                                    'LAB': ['L*', 'a*', 'b*'],
                                    'YUV': ['Y', 'U', 'V'],
                                    'YCrCb': ['Y', 'Cr', 'Cb']
                                }
                                
                                names = channel_names.get(selected_space, ['Channel 0', 'Channel 1', 'Channel 2'])
                                
                                for ch in range(3):
                                    axes[ch].imshow(space_img[:,:,ch], cmap='gray')
                                    axes[ch].set_title(f'{names[ch]} Channel')
                                    axes[ch].axis('off')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                    
                    # ========== TAB 6: TEKSTUR ==========
                    with tab6:
                        st.markdown('<div class="module-header">🔬 Modul 7: Analisis Tekstur</div>', unsafe_allow_html=True)
                        
                        # Texture analysis
                        first_order = processor.compute_first_order_statistics(image)
                        glcm_features = processor.compute_glcm_features(image)
                        lbp_image, lbp_hist = processor.compute_lbp_features(image)
                        gabor_responses, gabor_features = processor.compute_gabor_features(image)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Statistik Orde Pertama")
                            first_order_df = pd.DataFrame([
                                ["Mean", f"{first_order['mean']:.2f}"],
                                ["Variance", f"{first_order['variance']:.2f}"],
                                ["Std Deviation", f"{first_order['standard_deviation']:.2f}"],
                                ["Skewness", f"{first_order['skewness']:.2f}"],
                                ["Kurtosis", f"{first_order['kurtosis']:.2f}"],
                                ["Entropy", f"{first_order['entropy']:.2f}"]
                            ], columns=["Parameter", "Nilai"])
                            st.dataframe(first_order_df, use_container_width=True)
                            
                            st.subheader("🔍 Fitur GLCM")
                            glcm_df = pd.DataFrame([
                                ["Contrast", f"{glcm_features['contrast']:.4f}"],
                                ["Dissimilarity", f"{glcm_features['dissimilarity']:.4f}"],
                                ["Homogeneity", f"{glcm_features['homogeneity']:.4f}"],
                                ["Energy", f"{glcm_features['energy']:.4f}"],
                                ["Correlation", f"{glcm_features['correlation']:.4f}"],
                                ["ASM", f"{glcm_features['ASM']:.4f}"]
                            ], columns=["Fitur", "Nilai"])
                            st.dataframe(glcm_df, use_container_width=True)
                        
                        with col2:
                            st.subheader("🎭 Local Binary Pattern")
                            st.image(lbp_image, use_column_width=True, clamp=True)
                            
                            # LBP Histogram
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.bar(range(len(lbp_hist)), lbp_hist)
                            ax.set_title('Histogram LBP')
                            ax.set_xlabel('LBP Value')
                            ax.set_ylabel('Frequency')
                            st.pyplot(fig)
                            plt.close()
                        
                        # Gabor filters
                        st.subheader("🌊 Respons Filter Gabor")
                        gabor_cols = st.columns(4)
                        for idx, gabor_resp in enumerate(gabor_responses):
                            with gabor_cols[idx]:
                                st.image(gabor_resp, use_column_width=True, clamp=True)
                                st.caption(f"Filter Gabor {idx+1}")
                        
                        # Gabor features
                        st.subheader("📈 Energi Filter Gabor")
                        gabor_df = pd.DataFrame(list(gabor_features.items()), 
                                              columns=["Filter", "Energi"])
                        st.dataframe(gabor_df, use_container_width=True)
                    
                    st.markdown("---")
                
                st.success("✅ Pengolahan citra selesai!")
                st.balloons()
    
    else:
        # Landing page
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>🎯 Aplikasi Pengolahan Citra Digital Komprehensif</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Aplikasi ini mengimplementasikan 6 modul pengolahan citra digital:
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin: 2rem 0; color: black;">
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
                    <h4>📊 Modul 1: Array RGB & Info</h4>
                    <p>Analisis array RGB dan informasi dasar citra</p>
                </div>
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #ff7f0e; color: black;">
                    <h4>👤 Modul 4: Face Detection & Noise</h4>
                    <p>Deteksi wajah, noise processing, dan sharpening</p>
                </div>
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2ca02c; color: black;">
                    <h4>🔍 Modul 5: Kontur & Bentuk</h4>
                    <p>Analisis kontur, deteksi tepi, dan fitur bentuk</p>
                </div>
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #d62728; color: black;">
                    <h4>📦 Modul 6: Kompresi Citra</h4>
                    <p>Kompresi JPEG dan Run-Length Encoding</p>
                </div>
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #9467bd; color: black;">
                    <h4>🎨 Modul 7: Ruang Warna</h4>
                    <p>Transformasi berbagai ruang warna</p>
                </div>
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #8c564b; color: black;">
                    <h4>🔬 Modul 7: Analisis Tekstur</h4>
                    <p>GLCM, LBP, Gabor, dan statistik tekstur</p>
                </div>
            </div>
            <p style="font-size: 1.1rem; color: #1f77b4; color: white;">
                🚀 <strong>Upload 1-3 gambar di sidebar untuk memulai!</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()