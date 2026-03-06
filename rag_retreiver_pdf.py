# Install required libraries
# pip install langchain langchain-community openai chromadb pypdf

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Step 1: Load the PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# Step 2: Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

# Step 3: Convert chunks into embeddings
embeddings = OpenAIEmbeddings()

# Step 4: Store embeddings in Vector Database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./db"
)

# Step 5: Create Retriever
retriever = vectorstore.as_retriever()

# Step 6: Load LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Step 7: Create Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Step 8: Ask Question
query = "What is this PDF about?"

response = qa_chain.run(query)

print(response)