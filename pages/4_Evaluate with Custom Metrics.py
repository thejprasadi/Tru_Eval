
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

from trulens_eval.feedback.provider import OpenAI
from trulens_eval.utils.generated import re_0_10_rating
from typing import Optional, Dict, Tuple
from trulens_eval.feedback import prompts

class Custom_FeedBack(OpenAI):
    def custom_metric_score(self, answer: Optional[str] = None, question: Optional[str] = None, context: Optional[any] = None) -> Tuple[float, Dict]:
      """
      Tweaked version of context relevance, extending AzureOpenAI provider.
      A function that completes a template to check the relevance of the statement to the question.
      Scoring guidelines for scores 5-8 are removed to push the LLM to more extreme scores.
      Also uses chain of thought methodology and emits the reasons.

      Args:
          question (str): A question being asked.
          context (str): A statement to the question.

      Returns:
          float: A value between 0 and 1. 0 being "not relevant" and 1 being "relevant".
      """
      global prompt

      if answer is not None and question is not None and context is not None:
          professional_prompt = "prompt: {}\n" \
                                "where 0 is not at all related and 10 is extremely related: \n\n" \
                                "Answer: {}\n" \
                                "Question: {}\n" \
                                "Context: {}\n" \
                                "{}\n" .format(prompt, answer, question, context, prompts.COT_REASONS_TEMPLATE)
      elif answer is not None and question is not None and context is None:
          professional_prompt = "prompt: {}\n" \
                                "where 0 is not at all related and 10 is extremely related: \n\n" \
                                "Answer: {}\n" \
                                "Question: {}\n" \
                                "{}\n" .format(prompt, answer, question, prompts.COT_REASONS_TEMPLATE)
      elif answer is not None and question is None and context is not None:
          professional_prompt = "prompt: {}\n" \
                                "where 0 is not at all related and 10 is extremely related: \n\n" \
                                "Answer: {}\n" \
                                "Context: {}\n" \
                                "{}\n" .format(prompt, answer, context, prompts.COT_REASONS_TEMPLATE)
      elif answer is None and question is not None and context is not None:
          professional_prompt = "prompt: {}\n" \
                                "where 0 is not at all related and 10 is extremely related: \n\n" \
                                "Question: {}\n" \
                                "Context: {}\n" \
                                "{}\n" .format(prompt, question, context, prompts.COT_REASONS_TEMPLATE)
      elif answer is not None and question is None and context is None:
          professional_prompt = "prompt: {}\n" \
                                "where 0 is not at all related and 10 is extremely related: \n\n" \
                                "Answer: {}\n" \
                                "{}\n" .format(prompt, answer, prompts.COT_REASONS_TEMPLATE)
      elif answer is None and question is not None and context is None:
          professional_prompt = "prompt: {}\n" \
                                "where 0 is not at all related and 10 is extremely related: \n\n" \
                                "Question: {}\n" \
                                "{}\n" .format(prompt, question, prompts.COT_REASONS_TEMPLATE)
      elif answer is None and question is None and context is not None:
          professional_prompt = "prompt: {}\n" \
                                "where 0 is not at all related and 10 is extremely related: \n\n" \
                                "Context: {}\n" \
                                "{}\n" .format(prompt, context, prompts.COT_REASONS_TEMPLATE)
      else:
          professional_prompt = "prompt: {}\n" \
                                "where 0 is not at all related and 10 is extremely related: \n\n" \
                                "No answer, question, or context provided.\n" \
                                "{}\n" .format(prompt, prompts.COT_REASONS_TEMPLATE)

      system_prompt = prompts.CONTEXT_RELEVANCE_SYSTEM.replace("- STATEMENT that is RELEVANT to most of the QUESTION should get a score of 0 - 10. where 0 is not at all RELEVANCE and 10 is extremely RELEVANCE.\n\n", "")
      user_prompt = professional_prompt
      # user_prompt = user_prompt.replace( "RELEVANCE:", prompts.COT_REASONS_TEMPLATE)
      # print("----------------")
      # print("aaa : ",prompts.CONTEXT_RELEVANCE_USER)
      # print("bbb : ",question)
      # print("ccc : ",context)
      # print("ddd : ",prompts.COT_REASONS_TEMPLATE)
      # print("eee : ",professional_prompt)
      # print("User Prompt : ",user_prompt)
      # print("----------------")
      return self.generate_score_and_reasons(system_prompt, user_prompt)


standalone = Custom_FeedBack()


def assign_variables(ans, ques, cont):
    # Simply return the provided values
    return ans, ques, cont
prompt="initial"
from trulens_eval.app import App


def manage_variable(ans, ques, cont, promptMain, promptSub):
    returned_ans, returned_ques, returned_cont = ans,ques,cont
    global prompt
    prompt = promptSub
    context = App.select_context(chain)

    # Check and define f_custom_function based on variable values
    if returned_ans is not None and returned_ques is not None and returned_cont is not None:
        f_custom_function = (
            Feedback(standalone.custom_metric_score)
            .on(answer=Select.RecordOutput)
            .on(question=Select.RecordInput)
            .on(context)
        )
    elif returned_ans is None and returned_ques is None and returned_cont is not None:
        f_custom_function = (
            Feedback(standalone.custom_metric_score)
            .on(context)
        )
    elif returned_ans is None and returned_ques is not None and returned_cont is None:
        f_custom_function = (
            Feedback(standalone.custom_metric_score)
            .on(question=Select.RecordInput)
        )
    elif returned_ans is not None and returned_ques is None and returned_cont is None:
        f_custom_function = (
            Feedback(standalone.custom_metric_score)
            .on(answer=Select.RecordOutput)
        )
    elif returned_ans is None and returned_ques is not None and returned_cont is not None:
        f_custom_function = (
            Feedback(standalone.custom_metric_score)
            .on(question=Select.RecordInput)
            .on(context)
        )
    elif returned_ans is not None and returned_ques is None and returned_cont is not None:
        f_custom_function = (
            Feedback(standalone.custom_metric_score)
            .on(answer=Select.RecordOutput)
            .on(context)
        )
        
    elif returned_ans is not None and returned_ques is not None and returned_cont is None:
        f_custom_function = (
            Feedback(standalone.custom_metric_score)
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
    
st.subheader("Custom Metric Score",divider=False)
mainPrompt = st.text_input("RAG Question",placeholder='Please Enter the Question', key = 'mainPrompt')
cols = st.columns(3)  # Adjust the number of columns if needed

# Define the checkboxes in each column
with cols[0]:
    answer = st.checkbox("Answer")
with cols[1]:
    question = st.checkbox("Question")
with cols[2]:
    context = st.checkbox("Context")

# answer = st.checkbox("Answer")
# question = st.checkbox("Question")
# context = st.checkbox("Context")
#promptSubCheck = st.checkbox("Prompt")
#mainPrompt = st.text_input("RAG Question",placeholder='Please Enter the Question', key = 'mainPrompt')
promptSubCheck=st.text_input("Prompt",placeholder='Please Enter the Custom Defined Prompt', key = 'givenPrompt')
# if promptSubCheck:
#     st.text_input("Prompt",placeholder='Please Enter the Custom DefinedPrompt', key = 'givenPrompt')

 
submitted_btn = st.button("Evaluate with Custom Metrics", use_container_width=True, type="secondary")


if submitted_btn: 
    if answer:
        ans = 'ok'
    if question:    
        ques = 'ok'
    if context:
        cont = 'ok'
    
    promptSub = st.session_state.givenPrompt
        
    promptMain = st.session_state.mainPrompt
        
        
    rec = manage_variable(ans, ques, cont, promptMain, promptSub)
    st.markdown(rec.main_output)
    st.write("")
    st.write("")
    
    
    for feedback, feedback_result in rec.wait_for_feedback_results().items():
        meta=feedback_result.calls[0].meta
        main_meta=meta['reason']
        # main_reason=meta['reason']
        
        
            
        if feedback.name == "custom_metric_score":
            st.write("Custom Metric Score")
            st.text(f"Custom Metric Score: {feedback_result.result}")
            st.markdown(f"Reason: {main_meta}")
            st.divider()
        
    # st.write("Answer: ", ans)
    # st.write("Question: ", ques)
    # st.write("Context: ", cont)
    # st.write("Sub Prompt: ", promptSub)
    # st.write("Main Prompt: ", prompt)
    
    

st.write("")
st.write("")
st.write("") 
        
