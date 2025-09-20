import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
import random

# -------------------------------
# Custom CSS for Professional Styling
# -------------------------------
st.markdown("""
    <style>
        /* Main purple theme */
        .main-header {
            font-size: 3rem;
            color: #6a0dad;
            font-weight: 700;
            margin-bottom: 0;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #8a2be2;
            font-weight: 400;
            margin-top: 0;
        }
        .agrinova-badge {
            background-color: #6a0dad;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: inline-block;
        }
        .stButton>button {
            background-color: #6a0dad;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #8a2be2;
            color: white;
        }
        .consult-button {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-weight: 700;
            font-size: 1.1rem;
            margin: 2rem auto;
            display: block;
            width: 80%;
            text-align: center;
        }
        .consult-button:hover {
            background-color: #218838;
            color: white;
        }
        .prediction-card {
            background-color: #f8f5ff;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 5px solid #6a0dad;
            margin-top: 1rem;
        }
        .confidence-bar {
            background-color: #e6e6fa;
            height: 1.5rem;
            border-radius: 0.3rem;
            margin: 0.5rem 0;
            overflow: hidden;
        }
        .confidence-fill {
            background-color: #6a0dad;
            height: 100%;
            border-radius: 0.3rem;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 0.5rem;
            color: white;
            font-weight: 500;
            font-size: 0.8rem;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #6a0dad;
            font-size: 0.9rem;
        }
        /* Progress circle styling */
        .progress-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px 0;
        }
        .progress-circle {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: conic-gradient(#6a0dad 0% var(--progress), #f0f0f0 var(--progress) 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }
        .progress-circle::before {
            content: '';
            position: absolute;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: white;
        }
        .progress-text {
            position: relative;
            font-weight: bold;
            color: #6a0dad;
        }
        .symptom-item {
            margin-bottom: 8px;
            padding-left: 15px;
            position: relative;
        }
        .symptom-item:before {
            content: "•";
            position: absolute;
            left: 0;
            color: #6a0dad;
        }
        .disease-name {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .disease-name-red {
            color: #dc3545; /* Red for diseases */
        }
        .disease-name-green {
            color: #28a745; /* Green for healthy */
        }
        .causative-agent {
            font-size: 1.1rem;
            color: #6a0dad;
            margin-bottom: 15px;
            font-weight: 600;
            background-color: #f0e6ff;
            padding: 8px 12px;
            border-radius: 5px;
            border-left: 4px solid #6a0dad;
        }
        .confidence {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c662d;
        }
        /* Hide unnecessary Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 1. Load Local Model (from repo)
# -------------------------------
MODEL_PATH = "Maize_model.keras"  # exact filename in repo

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# -------------------------------
# 2. Class Names and Disease Information
# -------------------------------

CLASS_NAMES = [
    "Blight",
    "Common Rust",
    "Gray Leaf Spot",
    "Healthy"
]

# Disease information database
disease_info = {
    "Blight": {
        "causative_agent": "Bacterium *Pantoea stewartii* (Stewart's wilt / blight in maize)",
        "symptoms": [
            "Long, water-soaked lesions that turn gray or brown",
            "Leaves may appear wilted or scorched",
            "In severe cases, stunted plant growth and yield loss"
        ],
        "prevention": [
            "Plant resistant maize varieties",
            "Control insect vectors (e.g., corn flea beetles)",
            "Rotate crops with non-hosts to break disease cycle"
        ],
        "confidence": 0.92
    },
    "Common Rust": {
        "causative_agent": "Fungus *Puccinia sorghi*",
        "symptoms": [
            "Reddish-brown powdery pustules on both sides of leaves",
            "Heavy infections cause yellowing and premature leaf death",
            "Reduced photosynthesis leading to poor grain fill"
        ],
        "prevention": [
            "Use resistant maize hybrids",
            "Apply fungicides if disease pressure is high",
            "Practice crop rotation and good field sanitation"
        ],
        "confidence": 0.88
    },
    "Gray Leaf Spot": {
        "causative_agent": "Fungus *Cercospora zeae-maydis*",
        "symptoms": [
            "Rectangular gray to tan lesions running parallel to veins",
            "Leaves may appear blighted under heavy infection",
            "Can cause significant yield reduction in humid conditions"
        ],
        "prevention": [
            "Plant tolerant or resistant varieties",
            "Use crop rotation with non-cereal crops",
            "Apply fungicides during critical growth stages if needed"
        ],
        "confidence": 0.85
    },
    "Healthy": {
        "causative_agent": "None",
        "symptoms": [
            "Leaves remain green and photosynthetically active",
            "No visible lesions, pustules, or wilting"
        ],
        "prevention": [
            "Maintain good agronomic practices",
            "Regular scouting to catch early infections",
            "Use certified, disease-free seed"
        ],
        "confidence": 0.97
    }
}

# -------------------------------
# 3. Enhanced Streamlit Interface
# -------------------------------

st.set_page_config(
    page_title="AgriNova.ai - Maize Disease Detection",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Section
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="main-header">AgriNova.ai</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Intelligent Crop Disease Detection</h2>', unsafe_allow_html=True)
    
with col2:
    st.markdown('<div class="agrinova-badge">Maize Disease Classifier</div>', unsafe_allow_html=True)

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Main content
if model is None:
    st.error("Model loading failed. Please check if the model file exists in the repository.")
    st.info("""
    **Troubleshooting steps:**
    - Verify the model file 'Maize_model.keras' is in the root directory
    - Check the file is not corrupted
    - Refresh the page to try again
    """)
else:
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Maize Leaf Image")
        st.markdown("Upload a clear image of a maize leaf for disease analysis")
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # Show uploaded image with a border
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Maize Leaf", use_container_width=True)
            
            if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
                # Simulate analysis with variable time (5-15 seconds)
                analysis_time = random.randint(5, 15)
                progress_placeholder = st.empty()
                
                # Show progress circle
                for i in range(analysis_time):
                    progress_percent = (i + 1) / analysis_time * 100
                    progress_placeholder.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-circle" style="--progress: {progress_percent}%">
                            <div class="progress-text">{int(progress_percent)}%</div>
                        </div>
                    </div>
                    <p style="text-align: center;">Analyzing... {i+1}/{analysis_time} seconds</p>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
                
                # Clear progress indicator
                progress_placeholder.empty()
                
                # Preprocess image and run prediction
                img = image.resize((224, 224))
                img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

                # Run prediction
                try:
                    preds = model.predict(img_array)
                    confidence = np.max(preds) * 100
                    predicted_class = CLASS_NAMES[np.argmax(preds)]
                    
                    # Store results in session state
                    st.session_state.prediction = {
                        "class": predicted_class,
                        "confidence": confidence,
                        "all_predictions": preds[0]
                    }
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    with col2:
        if uploaded_file and "prediction" in st.session_state:
            pred = st.session_state.prediction
            
            st.markdown("### 📊 Analysis Results")
            
            # Results card
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
            # Get disease data
            disease_data = disease_info[pred["class"]]
            
            # Display disease name with color coding
            if pred["class"] == "Healthy":
                st.markdown(f'<div class="disease-name disease-name-green">{pred["class"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="disease-name disease-name-red">{pred["class"]}</div>', unsafe_allow_html=True)
            
            # Display causative agent with enhanced visibility
            st.markdown(f'<div class="causative-agent">Causative Agent: {disease_data["causative_agent"]}</div>', unsafe_allow_html=True)
            
            # Display confidence
            st.markdown(f'<div class="confidence">Confidence: {pred["confidence"]:.2f}%</div>', unsafe_allow_html=True)
            
            # Display symptoms
            st.markdown("**Typical Symptoms:**")
            for symptom in disease_data["symptoms"]:
                st.markdown(f'<div class="symptom-item">{symptom}</div>', unsafe_allow_html=True)
            
            # Display prevention measures
            st.markdown("**Prevention/Management:**")
            for measure in disease_data["prevention"]:
                st.markdown(f'<div class="symptom-item">{measure}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.info("👈 Upload an image and click 'Analyze' to get results")
    
    # Add "Consult an Expert" button below the two columns
    if uploaded_file and "prediction" in st.session_state:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f'<a href="tel:+2348136626696" style="text-decoration: none;">'
                f'<button class="consult-button">📞 Speak to an Expert: +2348136626696</button>'
                f'</a>',
                unsafe_allow_html=True
            )
            
    # Add some information about the system
    st.markdown("---")
    st.markdown("### ℹ️ About This System")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("**Technology**")
        st.markdown("- Deep Learning AI")
        st.markdown("- Computer Vision")
        st.markdown("- TensorFlow Backend")
        
    with info_col2:
        st.markdown("**Capabilities**")
        st.markdown("- 4 Disease Classifications")
        st.markdown("- Real-time Analysis")
        st.markdown("- Confidence Scoring")
        
    with info_col3:
        st.markdown("**Benefits**")
        st.markdown("- Early Disease Detection")
        st.markdown("- Reduced Crop Loss")
        st.markdown("- Increased Yield")

# Footer
st.markdown("---")
st.markdown('<div class="footer">AgriNova.ai • AI-Powered Agricultural Solutions • © 2025</div>', unsafe_allow_html=True)
