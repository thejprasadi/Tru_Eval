
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
from trulens_eval.feedback.provider import OpenAI
from trulens_eval.utils.generated import re_0_10_rating
from trulens_eval.feedback import prompts
from trulens_eval import Feedback, Select, Tru, TruChain
from trulens_eval.app import App

import os

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
embeddings = OpenAIEmbeddings()
model=ChatOpenAI(model_name="gpt-4-turbo-preview",temperature=0)
output_parser=StrOutputParser()
def format_docs(docs):
    format_D="\n\n".join([d.page_content for d in docs])
    return format_D

db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":10})
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
    )
context = App.select_context(chain)

st.set_page_config(
    page_title="Evaluate with Trulens",
    page_icon="👨‍💻",
    layout="wide",
    initial_sidebar_state="collapsed"
    
)

from trulens_eval.feedback.provider import OpenAI
from trulens_eval.utils.generated import re_0_10_rating
from typing import Optional, Dict, Tuple

class Custom_FeedBack(OpenAI):
    def check_cstom_metric(self, answer: Optional[str] = None, question: Optional[str] = None, context: Optional[str] = None) -> float:
        """
        Custom feedback function to evaluate RAG using custom metric.

        Args:
            answer (str): The generated answer.
            question (str): The question being asked.
            context (str): Context used to generate the answer.

        Returns:
            float: A value between 0 and 1. 0 being "not related to the professional_prompt" and 1 being "related to the professional_prompt".
        """
        global prompt

        if answer is not None and question is not None and context is not None:
            professional_prompt = "prompt: {}\n" \
                                  "where 0 is not at all related and 10 is extremely related: \n\n" \
                                  "Answer: {}\n" \
                                  "Question: {}\n" \
                                  "Context: {}\n".format(prompt, answer, question, context)
        elif answer is not None and question is not None and context is None:
            professional_prompt = "prompt: {}\n" \
                                  "where 0 is not at all related and 10 is extremely related: \n\n" \
                                  "Answer: {}\n" \
                                  "Question: {}\n".format(prompt, answer, question)
        elif answer is not None and question is None and context is not None:
            professional_prompt = "prompt: {}\n" \
                                  "where 0 is not at all related and 10 is extremely related: \n\n" \
                                  "Answer: {}\n" \
                                  "Context: {}\n".format(prompt, answer, context)
        elif answer is None and question is not None and context is not None:
            professional_prompt = "prompt: {}\n" \
                                  "where 0 is not at all related and 10 is extremely related: \n\n" \
                                  "Question: {}\n" \
                                  "Context: {}\n".format(prompt, question, context)
        elif answer is not None and question is None and context is None:
            professional_prompt = "prompt: {}\n" \
                                  "where 0 is not at all related and 10 is extremely related: \n\n" \
                                  "Answer: {}\n".format(prompt, answer)
        elif answer is None and question is not None and context is None:
            professional_prompt = "prompt: {}\n" \
                                  "where 0 is not at all related and 10 is extremely related: \n\n" \
                                  "Question: {}\n".format(prompt, question)
        elif answer is None and question is None and context is not None:
            professional_prompt = "prompt: {}\n" \
                                  "where 0 is not at all related and 10 is extremely related: \n\n" \
                                  "Context: {}\n".format(prompt, context)
        else:
            professional_prompt = "prompt: {}\n" \
                                  "where 0 is not at all related and 10 is extremely related: \n\n" \
                                  "No answer, question, or context provided.\n".format(prompt)

        return self.generate_score_and_reasons(system_prompt=professional_prompt)

standalone = Custom_FeedBack()


def assign_variables(ans, ques, cont):
    # Simply return the provided values
    return ans, ques, cont
prompt="initial"
from trulens_eval.app import App
context = App.select_context(chain)

def manage_variable(ans, ques, cont, promptMain, promptSub):
    returned_ans, returned_ques, returned_cont = ans,ques,cont
    global prompt
    prompt = promptSub
    global context

    # Check and define f_custom_function based on variable values
    if returned_ans is not None and returned_ques is not None and returned_cont is not None:
        f_custom_function = (
            Feedback(standalone.check_cstom_metric)
            .on(answer=Select.RecordOutput)
            .on(question=Select.RecordInput)
            .on(context)
        )
    elif returned_ans is None and returned_ques is None and returned_cont is not None:
        f_custom_function = (
            Feedback(standalone.check_cstom_metric)
            .on(context)
        )
    elif returned_ans is None and returned_ques is not None and returned_cont is None:
        f_custom_function = (
            Feedback(standalone.check_cstom_metric)
            .on(question=Select.RecordInput)
        )
    elif returned_ans is not None and returned_ques is None and returned_cont is None:
        f_custom_function = (
            Feedback(standalone.check_cstom_metric)
            .on(answer=Select.RecordOutput)
        )
    elif returned_ans is None and returned_ques is not None and returned_cont is not None:
        f_custom_function = (
            Feedback(standalone.check_cstom_metric)
            .on(question=Select.RecordInput)
            .on(context)
        )
    elif returned_ans is not None and returned_ques is None and returned_cont is not None:
        f_custom_function = (
            Feedback(standalone.check_cstom_metric)
            .on(answer=Select.RecordOutput)
            .on(context)
        )
        
    elif returned_ans is not None and returned_ques is not None and returned_cont is None:
        f_custom_function = (
            Feedback(standalone.check_cstom_metric)
            .on(answer=Select.RecordOutput)
            .on(question=Select.RecordInput)
        )
        
    tru_recorder = TruChain(chain,
    app_id='C',
    feedbacks=[f_custom_function])

    with tru_recorder as recording:
        llm_response = chain.invoke(promptMain)


    tru=Tru()
    records, feedback = tru.get_records_and_feedback(app_ids=[])

    rec = recording.get()
    
    return rec

    # for feedback, feedback_result in rec.wait_for_feedback_results().items():
    #     print(feedback.name, feedback_result.result)






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

if st.button('Homepage', key='backend_button', type="primary", use_container_width=True, help="Go to Homepage"):
    st.switch_page("1_Homepage.py")

st.title("Evaluate with Custom Metrics")

ans = None
ques = None
cont = None
promptSubCheck = None
promptSub = ""
    
st.subheader("Check the Custom Metrics",divider=False)
answer = st.checkbox("Answer")
question = st.checkbox("Question")
context = st.checkbox("Context")
promptSubCheck = st.checkbox("Prompt")
if promptSubCheck:
    st.text_input("Prompt",placeholder='Please Enter the Prompt', key = 'givenPrompt')
mainPrompt = st.text_input("RAG Questions",placeholder='Please Enter the Prompt', key = 'mainPrompt')

submitted_btn = st.button("Evaluate with Custom Metrics", use_container_width=True, type="secondary")


if submitted_btn: 
    if answer:
        ans = 'ok'
    if question:    
        ques = 'ok'
    if context:
        cont = 'ok'
    if promptSubCheck:
        promptSub = st.session_state.givenPrompt
        
    promptMain = st.session_state.mainPrompt
        
        
    rec = manage_variable(ans, ques, cont, promptMain, promptSub)
    
    
    for feedback, feedback_result in rec.wait_for_feedback_results().items():
        st.write(feedback.name, feedback_result.result)
        
    # st.write("Answer: ", ans)
    # st.write("Question: ", ques)
    # st.write("Context: ", cont)
    # st.write("Sub Prompt: ", promptSub)
    # st.write("Main Prompt: ", prompt)
    
    

st.write("")
st.write("")
st.write("") 
        