import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ['groq_api_key'] = os.getenv("groq_chatbot")
os.environ['langchain_api_key'] = os.getenv('langchain')

prompt = ChatPromptTemplate.from_messages([
    ('system','You are a Helpful Real Estate Agent. Who categorises a plot description into various categories.'),
    ('user','Plot Description:{question}')
])

def generate_response(question,engine,temperature,max_token):
    llm = ChatGroq(model = engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question':question})
    return answer


st.title('Real Estate chatbot')
engine = st.sidebar.selectbox('Select Model',['gemma2-9b-it','distil-whisper-large-v3-en','llama-3.1-70b-versatile','whisper-large-v3'])
temperature = st.sidebar.slider(label='Temperature',min_value=0.0,max_value=1.0,value=0.7)
max_token = st.sidebar.slider(label='Max Tokens',min_value=100,max_value=500,value=250)

st.write('Please provide your plot description')
user_input = st.text_area(label='You:',height=250)

if user_input:
    response = generate_response(user_input,engine,temperature,max_token)
    st.write(response)
    
else:
    st.write('Provide description')

