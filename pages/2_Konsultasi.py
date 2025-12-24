import streamlit as st
import google.generativeai as genai

st.set_page_config(
    page_title="Konsultasi Dokter AI",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Konsultasi Skinlytics")
st.caption("Tanyakan apapun tentang kesehatan kulit, gejala, atau hasil diagnosa kamu.")
st.sidebar.write("Project by: Kelompok 3")

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    st.error("Secrets tidak ditemukan. Pastikan file .streamlit/secrets.toml ada.")
    st.stop()

genai.configure(api_key=api_key)
try:
    model = genai.GenerativeModel('gemini-flash-latest')
except:
    model = genai.GenerativeModel('gemini-pro')

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya asisten AI Skinlytics. Ada yang bisa saya bantu terkait masalah kulit Anda?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ketik pertanyaanmu di sini..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            context_diagnosa = ""
            if 'hasil_diagnosa' in st.session_state:
                hasil = st.session_state['hasil_diagnosa']
                context_diagnosa = f"""
                [INFO PENTING: User ini BARU SAJA melakukan diagnosa kulit otomatis]
                - Penyakit terdeteksi: {hasil['penyakit']}
                - Tingkat Keyakinan AI: {hasil['confidence']:.2f}%
                
                Jika user bertanya "apa obatnya", "jelaskan penyakit saya", atau "apakah ini bahaya",
                kamu HARUS menjawab spesifik mengenai penyakit {hasil['penyakit']} tersebut.
                """

            system_instruction = f"""
            Kamu adalah asisten medis ahli dermatologi (kulit) untuk aplikasi bernama Skinlytics.
            
            {context_diagnosa}
            
            Tugasmu:
            1. Menjawab pertanyaan seputar penyakit kulit, perawatan wajah, dan kesehatan kulit.
            2. Jika user bertanya tentang penyakit lain (misal jantung/mobil), tolak dengan sopan.
            3. Selalu ingatkan bahwa jawabanmu bukan pengganti diagnosa dokter asli.
            4. Gunakan bahasa Indonesia yang ramah, empatik, dan mudah dipahami.
            
            Pertanyaan User: 
            """
            
            response = model.generate_content(system_instruction + prompt, stream=True)
            
            for chunk in response:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Terjadi kesalahan koneksi: {e}")