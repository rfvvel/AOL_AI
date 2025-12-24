import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Skin Disease AI - Home",
    page_icon="🏠",
    layout="wide"
)
st.title("🏥 Skin Disease Detection System")

st.divider()

st.header("👨‍💻 Meet the Team")
st.write("Tim Mahasiswa Computer Science - BINUS University")
st.sidebar.write("Project by: Kelompok 3")
col1, col2, col3 = st.columns(3)

with col1:
    try:
        img_max = Image.open("maxwell.jpg") 
        st.image(img_max, use_container_width=True)
    except:
        st.warning("Foto Maxwell belum ada")
    
    st.subheader("Maxwell Nathaniel P.")
    st.caption("Ketua Tim | NIM: 2802428053")
    st.write("Role: AI Development & Research")

with col2:
    try:
        img_raf = Image.open("rafael.jpg")
        st.image(img_raf, use_container_width=True)
    except:
        st.warning("Foto Rafael belum ada")

    st.subheader("Rafael Putra Wijaya")
    st.caption("Anggota | NIM: 2802436471")
    st.write("Role: Dataset & Preprocessing")

with col3:
    try:
        img_stev = Image.open("steven.jpg")
        st.image(img_stev, use_container_width=True)
    except:
        st.warning("Foto Steven belum ada")

    st.subheader("Steven Nathaniel")
    st.caption("Anggota | NIM: 2802436420")
    st.write("Role: System Architecture")

st.divider()

st.info("Ingin langsung mencoba deteksi penyakit?")
if st.button("🚀 Mulai Diagnosa Sekarang"):
    st.switch_page("pages/1_Diagnosa.py")
    