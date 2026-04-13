
**📌 Project Title Dialogue Summarization Using Large Language Models (FLAN‑T5)**  

📖 Overview : This project demonstrates how Generative AI and Large Language Models (LLMs) can be applied to the task of dialogue summarization. The focus is on understanding how prompt design and inference strategies influence model performance, rather than model training. 

Using the FLAN‑T5 model from Hugging Face and the DialogSum dataset, the project walks through:

    Summarization without prompt engineering
    Instruction‑based prompting (zero‑shot)
    One‑shot and few‑shot in‑context learning
    Effects of generation configuration parameters

🎯 Objectives

    Understand how LLMs interpret natural language prompts
    Compare zero‑shot, one‑shot, and few‑shot inference
    Analyze the role of prompt templates in guiding model behavior
    Experiment with generation parameters such as temperature and sampling
    Gain practical experience with LLM‑based NLP workflows

🧠 Model & Dataset

    Model: FLAN‑T5 (google/flan‑t5‑base)
    Dataset: DialogSum (10,000+ human‑annotated dialogues and summaries)

🧪 Key Experiments

    Baseline inference: Model behavior without task instructions
    Instruction prompting: Explicit summarization prompts
    FLAN prompt templates: Task‑optimized prompts
    One‑shot learning: Single example provided in context
    Few‑shot learning: Multiple examples for improved task understanding
    Generation tuning: Effects of max_new_tokens, temperature, top‑k, and top‑p

🛠️ Tech Stack

    Python
    PyTorch
    Hugging Face Transformers & Datasets
    FLAN‑T5
    Jupyter Notebook

✅ Key Takeaways

    Prompt engineering significantly improves LLM performance
    One‑shot and few‑shot learning enable effective in‑context learning
    Increasing shots beyond a threshold yields diminishing returns
    Generation parameters strongly affect output quality and creativity
    Prompt design is often more impactful than model changes for many NLP tasks

🚀 Future Improvements

    Fine‑tuning FLAN‑T5 on dialogue‑specific data
    Automatic evaluation using ROUGE or BERTScore
    Extending to domain‑specific dialogues (customer support, banking, healthcare)
    Comparing performance with newer LLMs

📌 Use Cases

    Customer support summarization
    Meeting and conversation summaries
    Chatbot conversation analytics
    Business intelligence and reporting


📊 ROUGE Metrics (Evaluation for Summarization)

ROUGE = Recall‑Oriented Understudy for Gisting Evaluation
It measures how much overlap exists between a generated summary and a reference (ground truth) summary.
🔹 ROUGE‑1

Definition:
Measures overlap of unigrams (single words) between generated and reference summaries.

Formula (Recall version):
ROUGE-1=Number of overlapping unigramsTotal unigrams in reference
ROUGE-1=Total unigrams in referenceNumber of overlapping unigrams​



ROUGE‑2

Definition:
Measures overlap of bigrams (two consecutive words).

Formula (Recall version):
ROUGE-2=Number of overlapping bigramsTotal bigrams in reference
ROUGE-2=Total bigrams in referenceNumber of overlapping bigrams​


ROUGE‑L

Definition:
Based on Longest Common Subsequence (LCS) between generated and reference text.

LCS = longest sequence of words appearing in both texts in the same order (not necessarily consecutive).

Formula (Recall version):
ROUGE-L=Length of LCSLength of reference
ROUGE-L=Length of referenceLength of LCS​


Intuition

    BLEU asks:(Precision-Oriented)
    “Did you generate exactly what I expected?”

    ROUGE asks: (Recall-Oriented)
    “Did you cover the important parts?”

## FLAN-T5 247M base (https://arxiv.org/pdf/2210.11416)
 FLAN‑T5 is an encoder–decoder (seq2seq) model 


 ### Load Dataset and LLM
 Dataset : dialuge summairize 

Zero-Shot Baseline (Original FLAN-T5):


Full Fine-Tuning (Reference Baseline):

LoRa_optuna_optimizations

LoRa_Optimizations: 


Qualitative Evaluation (Human-Readable):



Quantitative ROUGE Comparison:

streamlit.py for inference 





