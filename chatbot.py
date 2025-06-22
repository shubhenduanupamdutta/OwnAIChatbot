import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Upload pdf files
st.header("My AI Chatbot")

with st.sidebar:
    st.title("Your documents")
    file = st.file_uploader("Upload a pdf file and start asking questions", type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)

    # Break it into changes
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=1000, chunk_overlap=150, length_function=len
    )

    chunks = text_splitter.split_text(text)
    st.write(chunks)
