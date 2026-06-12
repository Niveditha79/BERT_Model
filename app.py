# app.py
import os
import logging

# Mute system warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Page Config
st.set_page_config(
    page_title="Standard BERT Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

# 2. Custom CSS (Sage Green & Forest Emerald Theme)
st.markdown("""
    <style>
        /* Muted Sage Green Background */
        .stApp {
            background-color: #f0fdf4;
        }
        
        /* Forest Emerald Gradient Title */
        .gradient-heading {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #15803d 0%, #0d9488 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            padding-bottom: 5px;
        }
        
        .title-subtitle {
            font-size: 1.05rem;
            color: #166534; /* Darker Green Subtext */
            margin-bottom: 20px;
        }
        
        .title-divider {
            height: 4px;
            background: linear-gradient(135deg, #15803d 0%, #0d9488 100%);
            border-radius: 2px;
            margin-bottom: 30px;
        }
        
        /* Modern Column Headers */
        .col-header-1 {
            color: #15803d; /* Emerald */
            font-weight: 700;
            font-size: 1.25rem;
            margin-bottom: 15px;
            border-bottom: 2px solid #bbf7d0;
            padding-bottom: 8px;
        }
        .col-header-2 {
            color: #0f766e; /* Teal */
            font-weight: 700;
            font-size: 1.25rem;
            margin-bottom: 15px;
            border-bottom: 2px solid #99f6e4;
            padding-bottom: 8px;
        }
        .col-header-3 {
            color: #115e59; /* Deep Teal */
            font-weight: 700;
            font-size: 1.25rem;
            margin-bottom: 15px;
            border-bottom: 2px solid #99f6e4;
            padding-bottom: 8px;
        }

        /* White Dashboard Card with Sage Borders */
        .prediction-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 25px 15px;
            box-shadow: 0 4px 6px -1px rgba(22, 101, 52, 0.05);
            border: 1px solid #bbf7d0;
            border-top: 5px solid #15803d;
            text-align: center;
            margin-bottom: 20px;
        }
        .prediction-label {
            font-size: 0.8rem;
            color: #166534;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .prediction-class {
            font-size: 1.8rem;
            font-weight: 800;
            color: #14532d; /* Dark Forest Green */
            margin: 10px 0;
        }
        .prediction-confidence {
            font-size: 1.1rem;
            color: #0d9488;
            font-weight: 700;
        }

        /* Sidebar Badge Styling in Sage Green Tones */
        .sidebar-badge {
            background-color: #dcfce7;
            color: #166534;
            padding: 10px 14px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 10px;
            display: inline-block;
            width: 100%;
            border: 1px solid #bbf7d0;
            border-left: 4px solid #15803d;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Class Mapping
CLASS_MAPPING = {
    'negative': '🔴 Negative Sentiment',
    'positive': '🟢 Positive Sentiment'
}
CLASS_NAMES = list(CLASS_MAPPING.keys())

# Preset Movie Reviews
PRESETS = {
    "Write custom review...": "",
    "Preset Positive Sample": "An outstanding performance from the entire cast. The screenplay is beautiful and the pacing is excellent.",
    "Preset Negative Sample": "This was a complete waste of time. The script is poorly structured and the characters are dull."
}

# 4. Smart Model Loader
@st.cache_resource
def load_bert():
    local_model_path = 'bert_model'
    local_tokenizer_path = 'bert_tokenizer'
    hub_model_name = "textattack/bert-base-uncased-SST-2"
    
    # If the local folders exist, use them
    if os.path.exists(local_model_path) and os.path.exists(local_tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    else:
        # Otherwise, fetch directly from Hugging Face Hub (ideal for Streamlit Cloud)
        tokenizer = AutoTokenizer.from_pretrained(hub_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(hub_model_name)
        
    return model, tokenizer

model, tokenizer = load_bert()

# 5. Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 style='color: #15803d; font-weight:800; margin-top:0;'>BERT Console</h2>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='color: #166534; font-weight: 600;'>Target Categories:</p>", unsafe_allow_html=True)
    for category in CLASS_MAPPING.values():
        st.markdown(f'<div class="sidebar-badge">{category}</div>', unsafe_allow_html=True)
    st.divider()
    st.caption("Engine: Standard BERT-Base")

# 6. Main Dashboard Layout
st.markdown('<h1 class="gradient-heading">Standard BERT Classifier</h1>', unsafe_allow_html=True)
st.markdown('<div class="title-subtitle">Contextual sequence parsing utilizing standard BERT-Base (110M params) sequence classification pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="title-divider"></div>', unsafe_allow_html=True)

if model is None or tokenizer is None:
    st.error("🚨 **BERT initialization failed.** Unable to load the model from local paths or the Hugging Face Hub.")
else:
    # Restructured 3-Column Interface
    col1, col2, col3 = st.columns([1, 1.5, 1.25], gap="large")

    with col1:
        st.markdown('<div class="col-header-1">1. Choose Template</div>', unsafe_allow_html=True)
        preset_choice = st.selectbox(
            "Select preloaded textual patterns:",
            options=list(PRESETS.keys())
        )
        default_text = PRESETS[preset_choice] if PRESETS[preset_choice] != "" else ""

    with col2:
        st.markdown('<div class="col-header-2">2. Document Input</div>', unsafe_allow_html=True)
        user_input = st.text_area(
            "Edit or type document text sequence:",
            value=default_text,
            height=160,
            placeholder="Type your review here..."
        )

    with col3:
        st.markdown('<div class="col-header-3">3. Standard BERT Outputs</div>', unsafe_allow_html=True)
        
        if not user_input.strip():
            st.markdown("""
                <div style="background-color: #ffffff; border: 1px dashed #bbf7d0; border-radius: 8px; padding: 25px; text-align: center;">
                    <p style="color: #166534; font-size: 0.9rem; margin: 0;">Provide active text sequences in column 2 to display live inference charts.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Processing standard BERT transformer attention layers..."):
                # Run prediction
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1).numpy()[0]
                
                best_match_idx = np.argmax(probs)
                label_name = CLASS_NAMES[best_match_idx]
                display_label = CLASS_MAPPING[label_name]
                confidence_score = probs[best_match_idx] * 100
                
                # Metric Card
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="prediction-label">Evaluated Sentiment</div>
                        <div class="prediction-class">{display_label.upper()}</div>
                        <div class="prediction-confidence">Certainty Index: {confidence_score:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Progress Indicators
                for i, score in enumerate(probs):
                    class_tag = CLASS_MAPPING[CLASS_NAMES[i]]
                    col_txt, col_pb = st.columns([1, 2])
                    with col_txt:
                        if i == best_match_idx:
                            st.markdown(f"<span style='color: #15803d; font-weight: 800; font-size: 0.85rem;'>{class_tag}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color: #166534; font-weight: 500; font-size: 0.85rem;'>{class_tag}</span>", unsafe_allow_html=True)
                    with col_pb:
                        if i == best_match_idx:
                            st.progress(float(score), text=f"Match — {score*100:.1f}%")
                        else:
                            st.progress(float(score), text=f"{score*100:.1f}%")