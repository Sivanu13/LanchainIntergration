import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Hardcoded OpenAI API key and document
HARDCODED_OPENAI_API_KEY = ""
HARDCODED_DOCUMENT = "Documents/text.txt"

def generate_response(query_text):
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents([HARDCODED_DOCUMENT])
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=HARDCODED_OPENAI_API_KEY)
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=HARDCODED_OPENAI_API_KEY), chain_type='stuff', retriever=retriever)
    return qa.run(query_text)

# Page title
st.set_page_config(page_title=' Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

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
