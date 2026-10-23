import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Configuração da página do Streamlit
st.set_page_config(page_title="Detector de Objetos - Marketing", page_icon="🔍", layout="centered")

@st.cache_resource
def carregar_modelo():
    return YOLO('yolov8n.pt')

st.title("📊 Detector de Ativos de Marketing (YOLOv8)")
st.write("Identifique carros e celulares em imagens de campanhas de forma rápida.")

model = carregar_modelo()

# Upload de arquivo pelo usuário ou uso da imagem padrão images.jpg
uploaded_file = st.file_uploader("Escolha uma imagem...", type=['jpg', 'jpeg', 'png'])

image_path = 'images.jpg'
img = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    img = cv2.imread(image_path)
    if img is not None:
        st.info(f"Nenhum arquivo enviado. Utilizando a imagem padrão: `{image_path}`")
    else:
        st.warning(f"Certifique-se de colocar uma imagem chamada `{image_path}` na mesma pasta ou faça o upload acima.")

if img is not None:
    conf_thresh = st.slider("Limiar de Confiança", 0.1, 1.0, 0.4, 0.05)
    
    if st.button("Executar Detecção"):
        with st.spinner("Processando imagem..."):
            results = model(img, conf=conf_thresh, verbose=False)[0]
            
            target_classes = {2: 'Carro', 67: 'Celular'}
            img_copy = img.copy()
            
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id in target_classes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    label = f"{target_classes[cls_id]} {conf:.2f}"
                    
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_copy, label, (x1, max(y1 - 10, 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
            # Converte de BGR para RGB para o Streamlit exibir corretamente
            img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            
            st.image(img_rgb, caption="Resultado Processado", use_container_width=True)