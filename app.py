import streamlit as st
import psycopg2
from pgvector.psycopg2 import register_vector
from PIL import Image
import os
import io
import re
import tempfile 
from datasets import load_dataset
from gradio_client import Client, handle_file
import torch
from transformers import AutoProcessor, AutoModel

CANDIDATE_LABELS = ["chair", "table", "dresser", "wardrobe", "bed", "sofa", "desk", "nightstand", "armchair"]
ìLABELS_ITA = {
    "chair": "Sedia", "table": "Tavolo", "dresser": "Cassettiera", 
    "wardrobe": "Armadio", "bed": "Letto", "sofa": "Divano", 
    "desk": "Scrivania", "nightstand": "Comodino", "armchair": "Poltrona"
}

HF_TOKEN = os.getenv("HF_TOKEN")


SPACE_ID = os.getenv("SPACE_ID")

st.set_page_config(layout="wide", page_title="Upcycling Search")

@st.cache_resource
def load_hf_dataset():
    return load_dataset("Arkan0ID/furniture-dataset", split="train")

dataset_originale = load_hf_dataset()

@st.cache_resource
def load_local_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model_id = "google/siglip2-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_id)
    
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    
    return processor, model, device

def estrai_embedding_locale(image_bytes):
    try:
        processor, model, device = load_local_model()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            image_features = outputs.pooler_output
        elif isinstance(outputs, torch.Tensor):
            image_features = outputs
        else:
            image_features = outputs[0] 
            
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        return image_features[0].cpu().tolist()
    except Exception as e:
        st.error(f"Errore modello locale: {e}")
        return None

def query_huggingface_api(image_bytes):
    try:
        hf_space_client = Client(SPACE_ID, HF_TOKEN)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        result = hf_space_client.predict(
            handle_file(tmp_path),
            fn_index=0
        )
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        return result
    except Exception as e:
        st.error(f"Errore remoto: {e}")
        return None

st.title("Upcycling Furniture Search")
usa_locale = st.toggle("Usa modello in locale (Docker)", value=True)

uploaded_file = st.file_uploader("Carica un mobile", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((448, 448)) 
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    img_bytes = buf.getvalue()

    st.image(img, width=300)

    if st.button("Trova Mobili Simili"):
        with st.spinner("L'intelligenza artificiale sta analizzando l'immagine..."):
            if usa_locale:
                vettore = estrai_embedding_locale(img_bytes)
            else:
                vettore = query_huggingface_api(img_bytes)
        
        if vettore is not None:
            if isinstance(vettore, list) and len(vettore) > 0:
                if isinstance(vettore[0], list):
                    vettore = vettore[0]
            
            try:
                conn = psycopg2.connect(
                    dbname=os.getenv("DB_NAME"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASS"),
                    host=os.getenv("DB_HOST"),
                    port=os.getenv("DB_PORT")
                )
                register_vector(conn)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT nome_file, materiali, (embedding <=> %s::vector) AS distance 
                    FROM progetti_restauro 
                    ORDER BY distance ASC 
                    LIMIT 5;
                """, (vettore,))
                
                risultati = cursor.fetchall()
                conn.close()

                st.subheader("Mobili simili trovati:")
                cols = st.columns(5)
                for idx, (nome_file, materiali, dist) in enumerate(risultati):
                    try:
                        id_img = int(nome_file) 
                        with cols[idx]:
                            st.image(dataset_originale[id_img]['image'], use_container_width=True)
                            st.caption(f"Distanza: {dist:.3f}")
                    except Exception as e:
                        st.write("Errore indice")
            except Exception as e:
                st.error(f"Errore Database: {e}")