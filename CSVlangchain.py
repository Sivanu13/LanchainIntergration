import streamlit as st
import csv
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from collections import namedtuple

# OpenAI API key and CSV file
OPENAI_API_KEY = ""
CSV_FILE = "Documents/Google News.csv"

Document = namedtuple('Document', ['page_content', 'metadata'])

def load_documents_from_csv(csv_file):
    documents = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            documents.append(Document(' '.join(row), {}))
    return documents

def generate_response(query_text):
    documents = load_documents_from_csv(CSV_FILE)
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma.from_documents(documents, embeddings)  # Pass documents
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type='stuff', retriever=retriever)
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
