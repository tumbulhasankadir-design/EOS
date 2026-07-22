import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
import io
from docx import Document

# LangChain Neo4j kütüphaneleri
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="EOS | Epistemic OS (GraphRAG)", page_icon="🧠", layout="wide")
st.title("🏛️ Epistemic Operating System (EOS) v1.0 - Knowledge Graph Aktif")
st.markdown("*Dynamic Dialectical Engine & GraphQ&A Active*")

# --- YAN MENÜ (İŞLETİM SİSTEMİ KONTROLLERİ) ---
with st.sidebar:
    st.header("⚙️ EOS Kontrol Paneli")
    
    # Güvenli kasadan anahtarları çekme
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    
    # Neo4j Bağlantı Kontrolü (Bağlandığı için direkt yeşil yanacak)
    neo4j_connected = False
    try:
        NEO4J_URI = st.secrets["NEO4J_URI"]
        NEO4J_USER = st.secrets["NEO4J_USER"]
        NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
        
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
        neo4j_connected = True
        st.sidebar.success("Knowledge Graph Bağlı 🕸️")
    except Exception as e:
        st.sidebar.warning(f"Graf bağlanamadı: {str(e)}")
    
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
    
    year_lock = None
    if active_mode == "Tarihsel Mod (Zaman Kilidi)":
        year_lock = st.number_input("Tarih Kilidi (Örn: 1198)", value=1198)

# --- SİSTEM PROMPTU (ÇEKİRDEK) ---
def build_system_prompt(role, mode, year):
    base_prompt = """
    # MİSYON
    Sen sıradan bir yapay zekâ sohbet botu değilsin. Sen, eser merkezli çalışan, Knowledge Graph destekli bir 'Epistemik İşletim Sistemi'sin.
    Amacın kullanıcının yerine düşünmek değil; düşünmesini sağlamak, kör noktalarını göstermek ve eserler arası ilişki kurmaktır.
    Hiçbir zaman nihai hakikati temsil ettiğini iddia etme. Cevap verme, düşünme sürecini derinleştir.
    """
    role_instructions = {
        "Sokratik Öğretmen": "Asla doğrudan cevap verme. Yalnızca 'Neden?', 'Bu iddianın kaynağı nedir?' gibi sorular üret.",
        "Şeytanın Avukatı": "Kullanıcının savunduğu görüşün tam tersini en güçlü haliyle savun.",
        "Kör Nokta Avcısı": "Kullanıcının fark etmediği çelişkileri, eksik kaynakları ve karşı argümanları tespit et.",
        "Şarih": "Karmaşık metinleri açıkla ama anlamı değiştirme. Orijinal kavramları koru.",
        "Köprü Kurucu": "Farklı dönemler arasında ilişki kur.",
        "Araştırma Danışmanı": "Oturum sonunda literatürde az çalışılmış problemler üret.",
        "Moderatör": "Tartışmayı yönet, tarafsız kal ve konuyu toparla.",
        "Hakem": "Hangi iddiaların desteklendiğini ve hangi soruların cevapsız kaldığını göster."
    }
    return f"{base_prompt}\n\n# ŞU ANKİ ROLÜN: {role}\n{role_instructions.get(role, '')}"

# --- SOHBET GEÇMİŞİ YÖNETİMİ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

# --- KULLANICI GİRİŞİ VE GRAF SORGULAMA ---
if prompt := st.chat_input("Düşünceyi başlat (Örn: 'İbn Sina'nın faal akıl kavramı...'"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        try:
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
            
            # --- GRAFTAN BİLGİ ÇEKME ---
            graph_context = ""
            if neo4j_connected:
                with st.spinner("Knowledge Graph taranıyor... 🕸️"):
                    try:
                        cypher_chain = GraphCypherQAChain.from_llm(
                            cypher_llm=llm,
                            qa_llm=llm,
                            graph=graph,
                            verbose=False,
                            return_direct=True
                        )
                        graph_result = cypher_chain.invoke({"query": prompt})["result"]
                        if graph_result:
                            graph_context = f"\n\n# KNOWLEDGE GRAPH DOĞRULANMIŞ İLİŞKİLERİ:\n{graph_result}"
                    except Exception:
                        pass
            
            # --- YANIT ÜRETME ---
            final_system_prompt = build_system_prompt(active_role, active_mode, year_lock) + graph_context
            sys_msg = SystemMessage(content=final_system_prompt)
            api_messages = [sys_msg] + st.session_state.messages
            
            with st.spinner("Epistemik motor düşünüyor... ⚙️"):
                response = llm.invoke(api_messages)
                
            st.markdown(response.content)
            st.session_state.messages.append(AIMessage(content=response.content))
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")

# --- WORD DOSYASI İNDİRME ÖZELLİĞİ ---
def generate_word_document(messages):
    doc = Document()
    doc.add_heading('Epistemik İşletim Sistemi - Raporu', 0)
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            doc.add_heading('Araştırmacı:', level=2)
            doc.add_paragraph(msg.content)
        elif isinstance(msg, AIMessage):
            doc.add_heading('Epistemik Motor:', level=2)
            doc.add_paragraph(msg.content)
        doc.add_paragraph('_' * 40)
            
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

if st.session_state.messages:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Dışa Aktar")
    word_data = generate_word_document(st.session_state.messages)
    st.sidebar.download_button(
        label="Tüm Diyaloğu Word Olarak İndir",
        data=word_data,
        file_name="epistemik_diyalog.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
