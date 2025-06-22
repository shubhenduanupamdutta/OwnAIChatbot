import streamlit as st
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pypdf import PdfReader

load_dotenv()

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
    # st.write(chunks)

    # Generating Embeddings
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = OpenAIEmbeddings()

    # Creating Vector Store = FAISS
    vector_stores = FAISS.from_texts(chunks, embeddings)

    # Get user questions
    user_question = st.text_input("Type your question here")

    # Do similarity search
    if user_question:
        match = vector_stores.similarity_search(user_question)
        # st.write(match)

        # Define llm
        llm = ChatOpenAI(
            temperature=0, max_completion_tokens=1000, model="gpt-3.5-turbo"
        )

        chain = load_qa_chain(llm)
        response = chain.run(input_questions=match, question=user_question)
        st.write(response)
