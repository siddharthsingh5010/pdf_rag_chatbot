### Description ###
# This is a simple Streamlit ChatBot app which is powered by OpenAI Model along with LangChain, RAG
# In this user can upload any text PDF file and can ask chatbot questions regarding content of PDF in natural language
####

# Importing Libraries
import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import base64
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Defining LLM Model and Splitter
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
text_splitter = RecursiveCharacterTextSplitter()

st.markdown(
    """
    <style>
        :root {
            --cyan: #22d3ee;
            --violet: #8b5cf6;
            --panel: rgba(15, 23, 42, 0.82);
            --muted: #94a3b8;
        }
        .stApp {
            color: #e2e8f0;
            background-color: #050816;
            background-image:
                radial-gradient(circle at 15% 0%, rgba(34, 211, 238, .12), transparent 32rem),
                radial-gradient(circle at 90% 10%, rgba(139, 92, 246, .14), transparent 28rem),
                linear-gradient(rgba(34, 211, 238, .025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(34, 211, 238, .025) 1px, transparent 1px);
            background-size: auto, auto, 32px 32px, 32px 32px;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] {
            max-width: 760px;
            padding-top: 5rem;
        }
        .eyebrow {
            color: var(--cyan);
            font-family: monospace;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .16em;
            margin-bottom: .8rem;
            text-transform: uppercase;
        }
        .profile-wrap {
            display: inline-flex;
            margin-bottom: 1.5rem;
            padding: 3px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--cyan), var(--violet));
            box-shadow: 0 0 30px rgba(34, 211, 238, .16);
        }
        .profile-photo {
            width: 132px;
            height: 132px;
            border: 4px solid #080d1c;
            border-radius: 50%;
            display: block;
            object-fit: cover;
            object-position: center;
        }
        .hero-title {
            font-size: clamp(2.8rem, 8vw, 4.8rem);
            font-weight: 800;
            letter-spacing: -.055em;
            line-height: 1;
            margin: 0 0 1.25rem;
            background: linear-gradient(100deg, #f8fafc 15%, var(--cyan) 55%, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #a9b6ca;
            font-size: 1.05rem;
            line-height: 1.75;
            margin-bottom: 1.35rem;
            max-width: 680px;
        }
        .subtitle .intro-title {
            color: #f1f5f9;
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0 0 .8rem;
        }
        .subtitle p { margin: 0 0 1rem; }
        .chips { display: flex; flex-wrap: wrap; gap: .55rem; margin-bottom: 3rem; }
        .chip {
            background: rgba(34, 211, 238, .07);
            border: 1px solid rgba(34, 211, 238, .2);
            border-radius: 999px;
            color: #b6eff7;
            font-family: monospace;
            font-size: .76rem;
            padding: .35rem .7rem;
        }
        .section-label {
            color: #f1f5f9;
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0 0 .85rem;
        }
        .contact-section { margin-top: 3rem; }
        .hire-message {
            background: linear-gradient(135deg, rgba(34, 211, 238, .09), rgba(139, 92, 246, .09));
            border: 1px solid rgba(34, 211, 238, .3);
            border-radius: 12px;
            color: #cbd5e1;
            line-height: 1.7;
            margin-top: .8rem;
            padding: 1.2rem;
        }
        .hire-message a { color: var(--cyan) !important; font-weight: 700; }
        .contact-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .75rem;
        }
        .contact-card {
            background: var(--panel);
            border: 1px solid rgba(148, 163, 184, .18);
            border-radius: 12px;
            color: #cbd5e1 !important;
            padding: 1rem;
            text-decoration: none !important;
            transition: all .2s ease;
            overflow-wrap: anywhere;
        }
        .contact-card:hover {
            border-color: var(--cyan);
            color: white !important;
            transform: translateY(-2px);
        }
        .contact-type {
            color: var(--cyan);
            display: block;
            font-family: monospace;
            font-size: .7rem;
            letter-spacing: .1em;
            margin-bottom: .35rem;
            text-transform: uppercase;
        }
        @media (max-width: 600px) {
            .contact-grid { grid-template-columns: 1fr; }
        }
        .stLinkButton > a {
            background: var(--panel);
            border: 1px solid rgba(148, 163, 184, .18);
            border-radius: 12px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, .18);
            color: #e2e8f0;
            font-size: 1rem;
            font-weight: 600;
            min-height: 4rem;
            justify-content: flex-start;
            padding-left: 1.25rem;
            transition: all .2s ease;
        }
        .stLinkButton > a:hover {
            background: rgba(17, 35, 57, .96);
            border-color: var(--cyan);
            box-shadow: 0 0 28px rgba(34, 211, 238, .12);
            color: white;
            transform: translateY(-2px);
        }
        .stButton { margin-top: 1rem; }
        .stButton > button {
            background: transparent;
            border-color: rgba(148, 163, 184, .2);
            color: var(--muted);
        }
        .stButton > button[data-testid="stBaseButton-primary"] {
            background: var(--panel);
            border: 1px solid rgba(148, 163, 184, .18);
            border-radius: 12px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, .18);
            color: #e2e8f0;
            font-size: 1rem;
            font-weight: 600;
            min-height: 4rem;
            transition: all .2s ease;
        }
        .stButton > button[data-testid="stBaseButton-primary"]:hover {
            background: rgba(17, 35, 57, .96);
            border-color: var(--cyan);
            box-shadow: 0 0 28px rgba(34, 211, 238, .12);
            color: white;
            transform: translateY(-2px);
        }
        /* Delete button */
        .delete-btn-wrap .stButton > button {
            background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
            border: 1px solid #ef4444 !important;
            border-radius: 12px !important;
            box-shadow: 0 0 20px rgba(239, 68, 68, .35) !important;
            color: #fff !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
            letter-spacing: .03em !important;
            min-height: 3rem !important;
            margin-top: 2rem !important;
            transition: all .2s ease !important;
            width: 100% !important;
        }
        .delete-btn-wrap .stButton > button:hover {
            background: linear-gradient(135deg, #ef4444, #dc2626) !important;
            box-shadow: 0 0 36px rgba(239, 68, 68, .55) !important;
            transform: translateY(-2px) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

photo_data = base64.b64encode(
    Path(__file__).with_name("icon.png").read_bytes()
).decode("ascii")

st.markdown(
    f'<div class="profile-wrap"><img class="profile-photo" '
    f'src="data:image/png;base64,{photo_data}" alt="AAAAA"></div>'
    '<div class="eyebrow">Ask Questions from PDF Document</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h1 class="hero-title">AI Powered PDF Document Chatbot</h1>'
    '<div class="subtitle">'
    '<p class="intro-title">Upload a PDF Document</p>'
    '<p>and let the chatbot answer any of your question directly '
    'from the context provided in the document'
    'using Retrieval Augmented Generation (RAG).'
    '</p>'
    '</div>', unsafe_allow_html=True
)

# Track saved file paths across reruns
if "saved_files" not in st.session_state:
    st.session_state.saved_files = []

# File upload widget
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

# Display information about the uploaded file
if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    # Saving file to disk
    save_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    # Track the saved path (avoid duplicates)
    if save_path not in st.session_state.saved_files:
        st.session_state.saved_files.append(save_path)

    # Show chat box after file upload
    st.markdown("Enter your Question below:")
    user_input = st.text_input("", placeholder="Type your message here...")

    # Build RAG pipeline
    loader = PyPDFLoader(save_path)
    document = loader.load()
    documents = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {question}""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain: retrieve → format → prompt → LLM → parse
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_input != "":
        answer = rag_chain.invoke(user_input)
        st.text_area("AI : ", value=answer, height=200)

    # ── Delete button ────────────────────────────────────────────────────────
    st.markdown('<div class="delete-btn-wrap">', unsafe_allow_html=True)
    if st.button("🗑️ Delete Uploaded File(s)", key="delete_files"):
        deleted = []
        for path in st.session_state.saved_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    deleted.append(os.path.basename(path))
            except Exception as e:
                st.error(f"Could not delete {os.path.basename(path)}: {e}")
        st.session_state.saved_files = []
        if deleted:
            names = ", ".join(deleted)
            st.success(f"✅ Successfully deleted: {names}")
        else:
            st.info("No files found on disk to delete.")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.write("No file uploaded yet.")
