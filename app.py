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
from dotenv import load_dotenv
load_dotenv()

# Defining LLM Model and Splitter
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
text_splitter = RecursiveCharacterTextSplitter()

# Title of the app
st.title("🤖 AI Powered ChatBot : Ask Questions from PDF Documents🤖")

st.markdown(
    "[💻 View Source on GitHub](https://github.com/siddharthsingh5010/pdf_rag_chatbot)", 
    unsafe_allow_html=True
)
st.markdown(
    "[Owner : Siddharth Singh](https://www.nomadicsid.com)", 
    unsafe_allow_html=True
)

# File upload widget
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

# Display information about the uploaded file
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    st.write(f"Uploaded file: {uploaded_file.name}")
    st.write(f"File type: {uploaded_file.type}")

    # Saving file to disk
    save_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    # Show chat box after file upload
    st.subheader("LLM and RAG Powered ChatBot")
    user_input = st.text_input("You: ", "Type your message here...")

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

    if user_input != 'Type your message here...':
        answer = rag_chain.invoke(user_input)
        st.text_area("AI : ", value=answer, height=200)
else:
    st.write("No file uploaded yet.")
