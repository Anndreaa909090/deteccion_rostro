import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from io import BytesIO

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="Detección de Rostros con IA",
    page_icon="🙂",
    layout="wide"
)

st.title("🙂 Detección de Rostros con OpenCV")
st.markdown("---")

# ==================== CARGA DE CLASIFICADORES ====================
@st.cache_resource
def load_cascades():
    """Carga los clasificadores Haar Cascade con caché"""
    cascades = {}
    
    cascade_files = {
        'face': 'haarcascade_frontalface_default.xml',
        'face_alt': 'haarcascade_frontalface_alt2.xml',  # Alternativo más preciso
        'eye': 'haarcascade_eye.xml',
        'smile': 'haarcascade_smile.xml'
    }
    
    for name, file in cascade_files.items():
        if os.path.exists(file):
            cascade = cv2.CascadeClassifier(file)
            if not cascade.empty():
                cascades[name] = cascade
    
    return cascades

cascades = load_cascades()

if 'face' not in cascades:
    st.error("❌ No se encontró el clasificador de rostros")
    st.info("📥 Descarga el archivo desde: https://github.com/opencv/opencv/tree/master/data/haarcascades")
    st.stop()

# ==================== BARRA LATERAL - CONFIGURACIÓN ====================
st.sidebar.header("⚙️ Configuración")

# Selector de método de detección
st.sidebar.subheader("Método de Detección")
detection_method = st.sidebar.radio(
    "Selecciona el método:",
    ["Haar Cascade Mejorado", "Haar Cascade Original"],
    help="El método mejorado tiene mejor precisión y menos falsos positivos"
)

# Parámetros de detección
st.sidebar.subheader("Parámetros de Detección")

if detection_method == "Haar Cascade Mejorado":
    scale_factor = st.sidebar.slider(
        "Scale Factor",
        min_value=1.05,
        max_value=1.3,
        value=1.15,  # Valor más alto = menos falsos positivos
        step=0.01,
        help="Valores más altos reducen falsos positivos"
    )
    
    min_neighbors = st.sidebar.slider(
        "Min Neighbors",
        min_value=3,
        max_value=15,
        value=8,  # Valor más alto = más estricto
        help="Valores más altos reducen falsos positivos"
    )
else:
    scale_factor = st.sidebar.slider(
        "Scale Factor",
        min_value=1.01,
        max_value=1.5,
        value=1.1,
        step=0.01
    )
    
    min_neighbors = st.sidebar.slider(
        "Min Neighbors",
        min_value=1,
        max_value=10,
        value=5
    )

min_size = st.sidebar.slider(
    "Tamaño Mínimo (px)",
    min_value=30,
    max_value=200,
    value=80,  # Aumentado para evitar detectar objetos pequeños
    step=10,
    help="Tamaño mínimo del rostro a detectar"
)

# Filtros adicionales
st.sidebar.subheader("Filtros de Validación")
use_aspect_ratio = st.sidebar.checkbox(
    "Filtrar por proporciones", 
    value=True,
    help="Solo acepta detecciones con proporciones similares a un rostro"
)

use_size_filter = st.sidebar.checkbox(
    "Filtrar por tamaño relativo",
    value=True,
    help="Descarta detecciones muy pequeñas respecto a la imagen"
)

# Opciones de detección adicionales
st.sidebar.subheader("Detecciones Adicionales")
detect_eyes = st.sidebar.checkbox("Detectar Ojos 👁️", value=True, disabled='eye' not in cascades)
verify_with_eyes = st.sidebar.checkbox(
    "Verificar con ojos",
    value=True,
    help="Solo marca como rostro si detecta al menos un ojo"
)

detect_smile = st.sidebar.checkbox("Detectar Sonrisas 😊", value=False, disabled='smile' not in cascades)

# Opciones de visualización
st.sidebar.subheader("Visualización")
rectangle_color = st.sidebar.color_picker("Color del rectángulo", "#00FF00")
rectangle_thickness = st.sidebar.slider("Grosor del rectángulo", 1, 10, 3)
show_confidence = st.sidebar.checkbox("Mostrar métricas", value=True)
show_rejected = st.sidebar.checkbox("Mostrar detecciones rechazadas", value=False)

# ==================== FUNCIONES DE DETECCIÓN ====================
def hex_to_rgb(hex_color):
    """Convierte color hex a RGB para OpenCV"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb

def validate_face_region(x, y, w, h, img_width, img_height, gray_roi, cascades, params):
    """
    Valida si una región detectada es realmente un rostro
    Returns: (is_valid, reason)
    """
    use_ar, use_sf, verify_eyes = params
    
    # 1. Filtro de proporciones (rostros suelen ser ~1:1.3)
    if use_ar:
        aspect_ratio = w / h
        if aspect_ratio < 0.6 or aspect_ratio > 1.5:
            return False, "Proporción incorrecta"
    
    # 2. Filtro de tamaño relativo (debe ocupar al menos 2% de la imagen)
    if use_sf:
        face_area = w * h
        image_area = img_width * img_height
        relative_size = face_area / image_area
        
        if relative_size < 0.02:  # Menos del 2% de la imagen
            return False, "Demasiado pequeño"
    
    # 3. Verificación con detección de ojos
    if verify_eyes and 'eye' in cascades:
        eyes = cascades['eye'].detectMultiScale(gray_roi, 1.1, 5)
        if len(eyes) < 1:
            return False, "Sin ojos detectados"
    
    return True, "Válido"

@st.cache_data
def detect_faces(image_bytes, _cascades, params):
    """Detecta rostros con filtros mejorados"""
    # Convertir bytes a imagen
    image = Image.open(BytesIO(image_bytes))
    img = np.array(image.convert("RGB"))
    img_height, img_width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Mejorar contraste
    gray = cv2.equalizeHist(gray)
    
    # Extraer parámetros
    (scale_f, min_neigh, min_s, use_ar, use_sf, verify_eyes, 
     detect_e, detect_s, color, thickness, show_rej) = params
    
    # Detectar rostros
    faces = _cascades['face'].detectMultiScale(
        gray,
        scaleFactor=scale_f,
        minNeighbors=min_neigh,
        minSize=(min_s, min_s),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    results = {
        'faces': 0,
        'eyes': 0,
        'smiles': 0,
        'rejected': 0
    }
    
    # Validar y dibujar rostros
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Validar región
        is_valid, reason = validate_face_region(
            x, y, w, h, img_width, img_height, roi_gray, _cascades,
            (use_ar, use_sf, verify_eyes)
        )
        
        if is_valid:
            results['faces'] += 1
            
            # Dibujar rectángulo verde para rostros válidos
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(img, f'Rostro #{results["faces"]}', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Detectar ojos
            if detect_e and 'eye' in _cascades:
                eyes = _cascades['eye'].detectMultiScale(roi_gray, 1.1, 10)
                results['eyes'] += len(eyes)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(img, (x + ex, y + ey), 
                                (x + ex + ew, y + ey + eh), (255, 255, 0), 2)
            
            # Detectar sonrisas
            if detect_s and 'smile' in _cascades:
                smiles = _cascades['smile'].detectMultiScale(roi_gray, 1.8, 20)
                results['smiles'] += len(smiles)
                
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(img, (x + sx, y + sy), 
                                (x + sx + sw, y + sy + sh), (255, 0, 255), 2)
        else:
            results['rejected'] += 1
            
            # Mostrar detecciones rechazadas en rojo
            if show_rej:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, f'X: {reason}', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return Image.fromarray(img), results

def image_to_bytes(image):
    """Convierte imagen PIL a bytes para descarga"""
    buf = BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()

# ==================== INTERFAZ PRINCIPAL ====================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Cargar Imagen")
    uploaded_file = st.file_uploader(
        "Sube una imagen (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Formatos soportados: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="📷 Imagen Original", use_column_width=True)
        
        # Información de la imagen
        st.info(f"**Dimensiones:** {original_image.size[0]} x {original_image.size[1]} px")

with col2:
    st.subheader("🎯 Resultado")
    
    if uploaded_file:
        with st.spinner("🔍 Detectando rostros..."):
            # Preparar parámetros
            color = hex_to_rgb(rectangle_color)
            params = (
                scale_factor, min_neighbors, min_size,
                use_aspect_ratio, use_size_filter, verify_with_eyes if detect_eyes else False,
                detect_eyes, detect_smile, color, rectangle_thickness, show_rejected
            )
            
            # Procesar imagen
            image_bytes = uploaded_file.getvalue()
            result_img, results = detect_faces(image_bytes, cascades, params)
            
            st.image(result_img, caption="✅ Imagen Procesada", use_column_width=True)
            
            # Mostrar resultados
            if show_confidence:
                if results['faces'] > 0:
                    st.success(f"✅ **Rostros detectados:** {results['faces']}")
                else:
                    st.warning("⚠️ **No se detectaron rostros válidos**")
                
                if results['rejected'] > 0:
                    st.info(f"🚫 **Detecciones rechazadas:** {results['rejected']}")
                
                if detect_eyes and results['eyes'] > 0:
                    st.info(f"👁️ **Ojos detectados:** {results['eyes']}")
                
                if detect_smile and results['smiles'] > 0:
                    st.info(f"😊 **Sonrisas detectadas:** {results['smiles']}")
            
            # Botón de descarga
            if results['faces'] > 0:
                st.download_button(
                    label="💾 Descargar Imagen Procesada",
                    data=image_to_bytes(result_img),
                    file_name="rostros_detectados.png",
                    mime="image/png"
                )

# ==================== PIE DE PÁGINA ====================
st.markdown("---")

# Tips para mejorar detección
with st.expander("💡 Tips para mejorar la detección"):
    st.markdown("""
    **Si no detecta rostros:**
    - Reduce el valor de **Min Neighbors** (a 5-6)
    - Reduce el **Tamaño Mínimo** (a 50-60px)
    - Desactiva "Verificar con ojos"
    
    **Si detecta objetos falsos (aretes, cadenas):**
    - Aumenta **Min Neighbors** (a 8-10) ✅
    - Aumenta **Scale Factor** (a 1.15-1.2) ✅
    - Aumenta **Tamaño Mínimo** (a 80-100px) ✅
    - Activa "Filtrar por proporciones" ✅
    - Activa "Verificar con ojos" ✅
    
    **Para mejor precisión:**
    - Usa imágenes con buena iluminación
    - Rostros frontales funcionan mejor
    - Evita ángulos muy inclinados
    """)

with st.expander("ℹ️ Información sobre los parámetros"):
    st.markdown("""
    **Scale Factor:** Valores más altos (1.15-1.2) = menos falsos positivos pero puede perder rostros pequeños.
    
    **Min Neighbors:** Valores más altos (8-10) = detección más estricta, menos falsos positivos.
    
    **Tamaño Mínimo:** Valores más altos ignoran objetos pequeños como aretes o accesorios.
    
    **Filtrar por proporciones:** Descarta detecciones que no tienen forma de rostro (muy anchas o muy altas).
    
    **Verificar con ojos:** Solo acepta como rostro si detecta al menos un ojo dentro de la región.
    """)

st.markdown("""
<div style='text-align: center'>
    <p>🔧 Desarrollado con Streamlit & OpenCV | 
    <a href='https://github.com/opencv/opencv' target='_blank'>Documentación OpenCV</a></p>
</div>
""", unsafe_allow_html=True)