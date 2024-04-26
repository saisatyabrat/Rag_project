import streamlit as st
import google.generativeai as genai

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from IPython.display import Markdown as md
from langchain_community.document_loaders import PyPDFLoader

#api key reading
f = open("key\.api.txt")
api_key = f.read()

# titel and sunb titel
st.title("Wel come to my RAG application")
st.subheader("a RAG System on “Leave No Context Behind” Paper")

# user input
user_input = st.text_area("Enter your query here :-")

# model defining
chat_model = ChatGoogleGenerativeAI(
    google_api_key = api_key,
    model = "gemini-1.5-pro-latest",
    convert_system_message_to_human=True)

# chat template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content= "you are a helpfull assistant,you have to take user question and answer relavant information about that query "),
    HumanMessagePromptTemplate.from_template("""Question: {question}
Answer:""")
])

#out-put parser
output_parsers = StrOutputParser()

#loading the pdf data
loader = PyPDFLoader("2404.07143v1.pdf")
pages = loader.load_and_split()

data = loader.load()


# chunking docoment

text_split = NLTKTextSplitter(chunk_size=500,chunk_overlap=100)
chunks = text_split.split_documents(data)


# creating chunk embadings

embading_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key,model = "models/embedding-001")

# store chunk in vector

db = Chroma.from_documents(chunks,embading_model,persist_directory= "./chroma_db")
db.persist()

# connecting with chroma
db_connection = Chroma(persist_directory= "./chroma_db",embedding_function = embading_model)

# retriving object from chroma db

retriver = db_connection.as_retriever(search_kwargs= {"k":5})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content = "you are a helpfull assistant,you have to take user question and answer relavant information about that query"),
    HumanMessagePromptTemplate.from_template("""Context:
    {context}

    Question:
    {question}

    Answer:""")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context":retriver | format_docs, "question": RunnablePassthrough()} 
    | chat_template
    | chat_model
    | output_parsers
)

if st.button("Generate Response"):
    st.balloons()
    response = rag_chain.invoke(user_input)
    st.write(response)
