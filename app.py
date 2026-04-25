import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="MedClassify", page_icon="🧠", layout="wide")

# -----------------------------
# CSS (Professional UI)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #9CA3AF;
    margin-bottom: 25px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #16a34a, #22c55e);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    border: none;
}

/* Input */
textarea {
    border-radius: 10px !important;
    border: 1px solid #374151 !important;
    background-color: #1F2937 !important;
    color: white !important;
}

/* Result box */
.result-box {
    background: linear-gradient(90deg, #064e3b, #16a34a);
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 0px 12px rgba(34,197,94,0.4);
    font-size: 20px;
    font-weight: 600;
}

/* Graph labels */
.graph-label {
    margin-top: 8px;
    margin-bottom: 2px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "biobert_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

label_map = {
    0: "Diagnostic",
    1: "Factoid",
    2: "Preventive",
    3: "Prognostic"
}

# -----------------------------
# Prediction Function
# -----------------------------
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()

    return label_map[pred], probs

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="title">🧠 MedClassify</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Medical Question Type Classification<br>Classifies medical questions into Diagnostic, Factoid, Preventive, Prognostic</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1.6, 1])

# LEFT
with col1:
    st.markdown("### 💡 Example questions:")
    st.write("""
    • What are the symptoms of diabetes?  
    • How is cancer diagnosed?  
    • How can I prevent heart disease?  
    • What is the prognosis of COVID-19?
    """)

    user_input = st.text_area("✍️ Enter your medical question:", height=120)
    analyze = st.button("🔍 Analyze Question")

# RIGHT (Graph)
with col2:
    st.markdown("### 📈 Model Confidence")

    graph_container = st.container()

st.markdown("---")

# -----------------------------
# Prediction
# -----------------------------
if analyze:
    if len(user_input.strip()) < 10:
        st.warning("⚠️ Please enter a meaningful medical question.")
    else:
        with st.spinner("Analyzing medical question..."):
            pred_label, probs = predict(user_input)

        confidence = float(torch.max(probs))

        # -------- GRAPH (RIGHT SIDE FIXED) --------
        with graph_container:
            for i, label in enumerate(label_map.values()):
                value = float(probs[i])

                # Highlight predicted class
                if label == pred_label:
                    st.markdown(f"<div class='graph-label'><b>{label} (Predicted)</b> — {value:.2f}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='graph-label'>{label} — {value:.2f}</div>", unsafe_allow_html=True)

                st.progress(value)

        # -------- RESULT --------
        st.markdown("### 📊 Result")

        st.markdown(f"""
        <div class="result-box">
        🏷️ Prediction: {pred_label}
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence)
        st.write(f"**Confidence:** {confidence:.2f}")
        st.caption("Higher confidence indicates stronger model certainty.")

# -----------------------------
# ABOUT
# -----------------------------
st.markdown("---")
st.markdown("### 📘 About")

st.write("""
MedClassify is a Natural Language Processing (NLP) project that classifies medical questions into:
- Diagnostic  
- Factoid  
- Preventive  
- Prognostic  

The system is trained on labeled medical datasets using transformer-based models.
""")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("<div style='text-align:center;'>© 2026 Malleshwari | Medical Question Type Classification System</div>", unsafe_allow_html=True)