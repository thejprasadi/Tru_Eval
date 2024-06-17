import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
# import os
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langsmith import Client
import os
import shutil
import stat, time
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "LLM Eval with Trulens"



template = """Answer the question based only on the following context:
{context}
If you don't know the answer, just say out of scope, don't try to make up an answer.
Question: {question}
"""

persist_directory = "data_docs"
prompt=ChatPromptTemplate.from_template(template)
model=ChatOpenAI(model_name="gpt-4-turbo-preview",temperature=0)
output_parser=StrOutputParser()
embeddings = OpenAIEmbeddings()


def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal"""
    os.chmod(path, stat.S_IWRITE)
    try:
        func(path)
    except Exception as e:
        # Wait for a short delay before retrying
        time.sleep(0.1)
        try:
            func(path)
        except Exception as e:
            print(f'Failed to delete {path}. Reason: {e}')

def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, onerror=remove_readonly)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def read_pdf(docs_raw):
    st.toast('Starting process project...', icon='🙂')
    docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(docs_raw_text)
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
    
    vectordb.persist()
    st.toast('Project processed successfully...', icon='😁')

st.set_page_config(
    page_title="Evaluate with Trulens",
    page_icon="👨‍💻",
    layout="wide",
    initial_sidebar_state="collapsed"
    
)


st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Evaluation of Contextual Search")

st.subheader("Upload Document",divider=False)

with st.form('qa_form'):
    uploaded_file = st.file_uploader("Choose a PDF file")
    submitted_btn = st.form_submit_button("Process Document", use_container_width=True, type="secondary")

if submitted_btn:
    if uploaded_file is not None:
        try:
            temp_file="./temp.pdf"
            with open(temp_file,"wb") as file:
                file.write(uploaded_file.getvalue())
            
            loader= PyPDFLoader(temp_file)
            docs_raw = loader.load()
            read_pdf(docs_raw)
            
        except Exception as e:
            st.error(e)
    
    
        
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button('Evaluation Metrics', key='backend_button1', type="primary", use_container_width=True, help="Click for Evaluation Metrics"):
        st.switch_page("pages/2_Q&A with Document.py")

with col2:
    if st.button('Batch Test Evaluation', key='backend_button2', type="primary", use_container_width=True, help="Click for Batch Test Evaluation"):
        st.switch_page("pages/3_Evaluation with Trulens.py")
        
with col3:
    if st.button('Evaluate with Custom Metrics', key='backend_button3', type="primary", use_container_width=True, help="Click for Evaluate with Custom Metrics"):
        st.switch_page("pages/4_Evaluate with Custom Metrics.py")


    

    
    

    

