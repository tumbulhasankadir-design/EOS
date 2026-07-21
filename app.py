import streamlit as st
from langchain_groq import ChatGroq # OpenAI kullanmak istersen langchain_openai import edebilirsin
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="EOS | Epistemic OS", page_icon="🧠", layout="wide")
st.title("🏛️ Epistemic Operating System (EOS) v1.0")
st.markdown("*Dynamic Dialectical Engine - Düşünceyi genişleten motor*")

# --- YAN MENÜ (İŞLETİM SİSTEMİ KONTROLLERİ) ---
with st.sidebar:
    st.header("⚙️ EOS Kontrol Paneli")
    active_role = st.selectbox(...) # <-- Tam olarak st.header ile aynı hizada
    
    # API anahtarını gizli kasadan çekiyoruz (Kullanıcı arayüzde görmez)
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    
    # Rol Seçimi
    active_role = st.selectbox(
        "🎭 Aktif Yapay Zekâ Rolü",
        ["Moderatör", "Sokratik Öğretmen", "Şeytanın Avukatı", "Kör Nokta Avcısı", "Şarih", "Hakem", "Köprü Kurucu", "Araştırma Danışmanı"]
    )
    
    # Mod Seçimi
    active_mode = st.radio(
        "🔄 Epistemik Mod",
        ["Standart Analiz", "Tarihsel Mod (Zaman Kilidi)", "Modern Mod (Karşılaştırmalı)", "Kavram Evrimi", "Tartışma Sonu Raporu"]
    )
    
    # Tarihsel mod seçildiyse yıl kilidi açılır
    year_lock = None
    if active_mode == "Tarihsel Mod (Zaman Kilidi)":
        year_lock = st.number_input("Tarih Kilidi (Örn: 1198)", value=1198)

# --- SİSTEM PROMPTU (ÇEKİRDEK) ---
# Senin yazdığın o muhteşem mimariyi buraya gömüyoruz. Seçilen role göre dinamik güncellenecek.
def build_system_prompt(role, mode, year):
    base_prompt = """
    # MİSYON
    Sen sıradan bir yapay zekâ sohbet botu değilsin. Sen, eser merkezli çalışan, Knowledge Graph destekli bir 'Epistemik İşletim Sistemi'sin.
    Amacın kullanıcının yerine düşünmek değil; düşünmesini sağlamak, kör noktalarını göstermek ve eserler arası ilişki kurmaktır.
    Hiçbir zaman nihai hakikati temsil ettiğini iddia etme. Cevap verme, düşünme sürecini derinleştir.
    
    # TARTIŞMA KURALLARI
    - Hiçbir düşünürün söylemediği bir söz uydurulamaz.
    - Her iddia kaynağa bağlanmalıdır.
    - Bilgi grafında bulunmayan (hayali) hiçbir ilişki kesin bilgi olarak sunulmayacaktır.
    """
    
    role_instructions = {
        "Sokratik Öğretmen": "Asla doğrudan cevap verme. Yalnızca 'Neden?', 'Bu iddianın kaynağı nedir?' gibi sorular üret.",
        "Şeytanın Avukatı": "Kullanıcının savunduğu görüşün tam tersini en güçlü haliyle savun. Amacın düşünceyi sınamaktır.",
        "Kör Nokta Avcısı": "Kullanıcının fark etmediği çelişkileri, eksik kaynakları ve karşı argümanları tespit et.",
        "Şarih": "Karmaşık metinleri açıkla ama anlamı değiştirme. Orijinal kavramları (Örn: Nefs, Akıl) koru.",
        "Köprü Kurucu": "Farklı dönemler arasında ilişki kur (Örn: Gazali -> Risk -> Davranışsal Ekonomi).",
        "Araştırma Danışmanı": "Oturum sonunda literatürde az çalışılmış problemler ve yeni araştırma soruları üret.",
        "Moderatör": "Tartışmayı yönet, tarafsız kal ve konuyu toparla.",
        "Hakem": "Hangi iddiaların desteklendiğini ve hangi soruların cevapsız kaldığını göster. Kazanan ilan etme."
    }
    
    mode_instructions = ""
    if mode == "Tarihsel Mod (Zaman Kilidi)" and year:
        mode_instructions = f"\n# TARİHSEL MOD AKTİF: Sistemin zamanı {year} yılına kilitlenmiştir. Bu tarihten sonra yazılmış hiçbir kaynağı kullanamaz ve modern yorum yapamazsın."
    elif mode == "Kavram Evrimi":
        mode_instructions = "\n# KAVRAM EVRİMİ MODU: Bahsedilen kavramın Kur'an, Hadis, Farabi, İbn Sina, Kant ve Modern bilim arasındaki tarihsel anlam kaymalarını adım adım göster."
    elif mode == "Tartışma Sonu Raporu":
        mode_instructions = "\n# RAPOR MODU: Ortak kavramlar, ayrışan kavramlar, kullanılan eserler, kör noktalar ve yeni araştırma önerilerini içeren yapılandırılmış bir rapor sun."
    
    return f"{base_prompt}\n\n# ŞU ANKİ ROLÜN: {role}\n{role_instructions.get(role, '')}\n{mode_instructions}"

# --- SOHBET GEÇMİŞİ YÖNETİMİ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana çiz
for msg in st.session_state.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

# --- KULLANICI GİRİŞİ VE LLM TETİKLEME ---
if prompt := st.chat_input("Düşünceyi başlat (Örn: 'İbn Sina'nın faal akıl kavramı...'"):
            
    # Kullanıcı mesajını ekle
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # LLM Çağrısı
    with st.chat_message("assistant"):
        try:
            # Model tanımı (Groq hızıyla diyalektik için idealdir)
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
            
            # Sistem promptunu dinamik olarak oluştur
            sys_msg = SystemMessage(content=build_system_prompt(active_role, active_mode, year_lock))
            
            # API'ye gönderilecek mesaj listesi (Sistem + Geçmiş)
            api_messages = [sys_msg] + st.session_state.messages
            
            # Yanıtı al ve yazdır
            response = llm.invoke(api_messages)
            st.markdown(response.content)
            
            # Yanıtı geçmişe kaydet
            st.session_state.messages.append(AIMessage(content=response.content))
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
