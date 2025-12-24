import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from fpdf import FPDF
import io

st.set_page_config(
    page_title="Skinlytics - AI Skin Detection",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Skinlytics")
st.markdown("### Deteksi Dini Penyakit Kulit Menggunakan Artificial Intelligence")
st.write("Upload gambar kulit yang ingin diperiksa, dan AI kami akan menganalisisnya.")
st.divider()

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('skin_model.keras')
    return model

def load_class_names():
    with open('class_names.txt', 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((160, 160))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

try:
    model = load_model()
    class_names = load_class_names()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def create_pdf(penyakit, keyakinan, img_file):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=24)
    pdf.cell(200, 20, txt="Laporan Hasil Diagnosa Skinlytics", ln=1, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Hasil Analisis AI:", ln=1, align='L')
    
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(200, 10, txt=f"Penyakit Terdeteksi: {penyakit}", ln=1, align='L')
    
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt=f"Tingkat Keyakinan: {keyakinan:.2f}%", ln=1, align='L')
    
    pdf.cell(200, 10, txt="", ln=1)
    pdf.cell(200, 10, txt="Saran: Segera konsultasikan hasil ini dengan dokter spesialis.", ln=1, align='L')
    
    img_path = "temp_img.jpg"
    img_file.save(img_path)
    pdf.image(img_path, x=10, y=80, w=100)
    
    return pdf.output(dest='S').encode('latin-1')


st.sidebar.write("Project by: Kelompok 3")
col1, col2 = st.columns(2)

with col1:
    st.header("📸 Upload Gambar")
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diupload', use_container_width=True)
        
        analyze_button = st.button("🔍 Deteksi Penyakit", type="primary")

        if analyze_button:
            with st.spinner('Sedang menganalisis pixel gambar...'):
                try:
                    processed_img = preprocess_image(image)
                    predictions = model.predict(processed_img)
                    score = predictions[0] 
                    
                    predicted_class = class_names[np.argmax(score)]
                    confidence = 100 * np.max(score)
                    
                    st.session_state['hasil_diagnosa'] = {
                        'penyakit': predicted_class,
                        'confidence': confidence,
                        'gambar': image,
                        'filename': uploaded_file.name 
                    }
                    st.success("Analisis Selesai!")
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

    else:
        if 'hasil_diagnosa' in st.session_state:
            del st.session_state['hasil_diagnosa']

with col2:
    st.header("📊 Hasil Analisis")
    st.markdown("<br>", unsafe_allow_html=True)
   
    if 'hasil_diagnosa' in st.session_state:
        result = st.session_state['hasil_diagnosa']
        
        st.subheader(f"Prediksi: {result['penyakit']}")
        st.progress(int(result['confidence']))
        st.caption(f"Tingkat Keyakinan AI: {result['confidence']:.2f}%")
        
        if result['confidence'] > 80:
            st.info("AI sangat yakin dengan hasil ini.")
        elif result['confidence'] > 50:
            st.warning("AI cukup yakin, tapi gambar mungkin kurang jelas.")
        else:
            st.error("AI ragu-ragu. Coba upload foto yang lebih jelas/dekat.")

        st.write("---")
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            pdf_bytes = create_pdf(result['penyakit'], result['confidence'], result['gambar'])
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name="Laporan_Skinlytics.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        with col_btn2:
            if st.button("💬 Tanya Dokter AI", type="primary", use_container_width=True):
                st.switch_page("pages/2_Konsultasi.py")
                
        st.write("---")
        st.warning("⚠️ **Disclaimer:** Hasil ini hanyalah prediksi komputer.")

    elif uploaded_file is None:
        st.info("👈 Silakan upload gambar di panel sebelah kiri untuk memulai.")