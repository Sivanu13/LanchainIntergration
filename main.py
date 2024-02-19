import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# OpenAI API key and document
OPENAI_API_KEY = "Documents/text.txt"
DOCUMENT = "Documents/Google News.txt"

def generate_response(query_text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) # Split documents into chunks
    texts = text_splitter.create_documents([DOCUMENT])
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # Select embeddings
    db = Chroma.from_documents(texts, embeddings) # Create a vectorstore from documents
    retriever = db.as_retriever() # Create retriever interface
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type='stuff', retriever=retriever) # Create QA chain
    return qa.run(query_text)

# Page title
st.set_page_config(page_title=' Ask the Doc App')
st.title('Analyze the news')

# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.')

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not query_text)
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(query_text)
            result.append(response)

if len(result):
    st.info(response)
