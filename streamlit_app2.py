"""
LoRA Fine-Tune Demo - Windows Fixed
output_merged/ folder | Classic Streamlit
"""

import streamlit as st
import torch
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Kills tokenizer warning

# Suppress torch distributed warning
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

st.set_page_config(page_title="LoRA Demo", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.main {margin-top: -20px;}
h1 {color: #1f77b4; font-family: 'Arial Black';}
</style>
""", unsafe_allow_html=True)

st.title("🤖 LoRA Fine-Tune Demo")
st.markdown("*FLAN-T5 + Trial 21 | 41.84% ROUGE-1*")

# Model loading with error handling
@st.cache_resource
def load_model():
    model_paths = ["./output_merged", "./output_merged/", "output_merged"]
    
    for path in model_paths:
        if os.path.exists(path) and os.path.isdir(path):
            try:
                st.info(f"🔍 Loading from: `{path}`")
                
                tokenizer = AutoTokenizer.from_pretrained(path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                model.eval()
                
                st.success("✅ Model loaded!")
                return model, tokenizer, path
            except Exception as e:
                st.warning(f"❌ Path `{path}` failed: {str(e)[:100]}")
                continue
    
    st.error("❌ No valid model found. Check `output_merged/` folder")
    return None, None, None

model, tokenizer, model_path = load_model()
if model is None:
    st.stop()

# Sidebar info
with st.sidebar:
    st.header("📊 Model Info")
    st.metric("ROUGE-1", "41.84%")
    st.metric("ROUGE-L", "33.22%")
    st.info(f"**Loaded:** `{model_path}`")
    st.code("r=128, alpha=64\nbs=32, lr=5.58e-4")

# Main app
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Generate Summary")
    
    prompt = st.text_area(
        "Conversation:",
        height=120,
        placeholder="Person1: Upgrade your system?\nPerson2: Not sure where to start..."
    )
    
    col_len, col_beam = st.columns(2)
    with col_len:
        max_length = st.slider("Max tokens", 30, 150, 80)
    with col_beam:
        num_beams = st.slider("Beams", 1, 6, 4)
    
    if st.button("**Generate** 🚀", type="primary"):
        if prompt and tokenizer:
            try:
                inputs = tokenizer(
                    f"Summarize:\n{prompt}",
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        num_beams=num_beams,
                        early_stopping=True,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.success("**Summary:**")
                st.markdown(f"**{summary}**")
                
            except Exception as e:
                st.error(f"Generation error: {e}")

with col2:
    st.header("🔥 Quick Tests")
    
    examples = {
        "Traffic": "Person1: Late again? Person2: Traffic jam!",
        "Office": "Person1: No more instant messaging. Person2: Understood.",
        "Upgrade": "Person1: Upgrade RAM? Person2: How much?"
    }
    
    ex = st.radio("Pick example:", examples.keys())
    if ex:
        st.text_area("", examples[ex], height=100, key="ex")

st.markdown("---")
st.markdown("*Venkat's LoRA | Lightning.AI L40S 46GB*")