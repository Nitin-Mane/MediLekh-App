# ✅ MediLekh Streamlit App – Fixed Multi-PDF RAG with QA, Summary & Stable UI

import streamlit as st
import pdfplumber
import google.generativeai as genai
import re
from PIL import Image
from datetime import datetime
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ====== Gemini & Vector Store Setup ======
genai.configure(api_key="AIzaSyCxjaXRhwGJIWK0wWExjMRo6Mn-dSGFr4k")
embedding_model = HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

# ====== Functions ======
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def save_chunks_to_vectorstore(chunks):
    docs = [Document(page_content=c) for c in chunks]
    return FAISS.from_documents(docs, embedding_model)

def retrieve_from_db(db, query, k=3):
    docs = db.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

def highlight_categories(text):
    highlight_map = {
        "Diagnosis": ["diabetes", "pneumonia", "stroke", "infection", "hypertension", "asthma"],
        "Treatment": ["tablet", "therapy", "medication", "injection", "antibiotic", "chemotherapy"],
        "Vitals": ["blood pressure", "temperature", "oxygen", "heart rate", "respiration"],
        "Medications": ["metformin", "insulin", "paracetamol", "atorvastatin"],
        "Procedures": ["ECG", "MRI", "CT scan", "colonoscopy", "surgery"],
        "Allergies": ["penicillin", "NSAIDs", "shellfish", "latex"],
        "Lab Findings": ["hemoglobin", "creatinine", "glucose", "WBC", "platelets"],
        "Administrative": ["admission", "discharge", "follow-up", "ward", "insurance"]
    }
    colors = {
        "Diagnosis": "#FFD6E0",
        "Treatment": "#D6F5D6",
        "Vitals": "#FFFACD",
        "Medications": "#E0FFFF",
        "Procedures": "#D8BFD8",
        "Allergies": "#FFC0CB",
        "Lab Findings": "#F0E68C",
        "Administrative": "#E6E6FA"
    }
    for cat, terms in highlight_map.items():
        for term in terms:
            text = re.sub(rf"\b({term})\b", 
                          f"<span style='background-color:{colors[cat]}; padding:2px;'><b>\\1</b></span>", 
                          text, flags=re.IGNORECASE)
    return text

def generate_summary(text, report_type):
    prompt = f"""
You are a clinical summarization expert. The following is a collection of medical reports belonging to a single patient. 
Generate a medically structured summary under headings like Diagnosis, Treatment, Vitals, and Follow-up.

Report Type: {report_type}
Patient Record:
{text}
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def question_text(context, question):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
Use the EMR context to answer clinically:

{context}

Question: {question}
"""
    response = model.generate_content(prompt)
    return response.text

def generate_auto_questions(report_type):
    base = {
        "General Checkup": ["What are the vitals?", "Any lifestyle advice?", "Was follow-up suggested?"],
        "Lab Results": ["What tests were abnormal?", "Any signs of infection?"],
        "Radiology Report": ["What findings are shown?", "Is surgery recommended?"],
        "Discharge Summary": ["Final diagnosis?", "Medications prescribed?", "Follow-up instructions?"],
        "Other": ["What condition is described?", "Any interventions or prescriptions?"]
    }
    extra = ["Summarize diagnosis", "List treatments", "Vitals summary", "Is surgery mentioned?", "Chronic illness noted?", "Follow-up documented?"]
    return base.get(report_type, base["Other"]) + extra

# ====== Streamlit App ======
def main():
    st.set_page_config(page_title="MediLekh – Clinical Assistant")
    st.title("🧠 MediLekh: Medical Assistant App")

    date = st.date_input("📅 Session Date", value=datetime.today())
    report_type = st.selectbox("📄 Report Type", ["General Checkup", "Lab Results", "Radiology Report", "Discharge Summary", "Other"])
    notes = st.text_area("🗒️ Session Notes")
    pdf_files = st.file_uploader("📎 Upload Medical PDFs", type="pdf", accept_multiple_files=True)

    if pdf_files:
        all_text, all_chunks = "", []
        for pdf in pdf_files:
            text = extract_text_from_pdf(pdf)
            all_text += text + "\n"
            all_chunks.extend(chunk_text(text))

        db = save_chunks_to_vectorstore(all_chunks)
        st.markdown("### 📘 Report Preview")
        preview = highlight_categories(all_text[:2000])
        st.markdown(f"<div style='border:1px solid #ccc; padding:10px;'>{preview}</div>", unsafe_allow_html=True)

        # Session state to keep summary and questions after interaction
        if "summary" not in st.session_state:
            st.session_state.summary = ""
        if "custom_question" not in st.session_state:
            st.session_state.custom_question = ""
        if "custom_answer" not in st.session_state:
            st.session_state.custom_answer = ""

        if st.button("📋 Generate Medical Summary"):
            st.session_state.summary = generate_summary(all_text, report_type)

        if st.session_state.summary:
            st.markdown("#### 📝 Clinical Summary")
            st.write(st.session_state.summary)

            st.markdown("---")
            st.markdown("### 💬 Ask a Question (User Input)")
            st.session_state.custom_question = st.text_input("Ask a clinical question:", value=st.session_state.custom_question)
            if st.button("🧠 Get Answer") and st.session_state.custom_question.strip():
                context = retrieve_from_db(db, st.session_state.custom_question)
                st.session_state.custom_answer = question_text(context, st.session_state.custom_question)

            if st.session_state.custom_answer:
                st.markdown(f"**Q:** {st.session_state.custom_question}")
                st.markdown(f"**A:** {st.session_state.custom_answer}")

            st.markdown("---")
            st.markdown("### 💡 Recommended Clinical Questions")
            for q in generate_auto_questions(report_type):
                if st.button(f"🔎 {q}", key=q):
                    context = retrieve_from_db(db, q)
                    response = question_text(context, q)
                    st.markdown(f"**Q:** {q}")
                    st.markdown(f"**A:** {response}")
                    st.markdown("---")

            st.subheader("📦 Session Summary")
            st.markdown(f"""
            - **Date**: {date.strftime('%Y-%m-%d')}
            - **Files**: {len(pdf_files)}
            - **Type**: {report_type}
            - **Notes**: {notes if notes else 'N/A'}
            """)

if __name__ == "__main__":
    main()
