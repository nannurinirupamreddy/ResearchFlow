from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_files(path: str, file_type: str, loader_type=PyPDFLoader):
    loader = DirectoryLoader(path=f"{path}", glob=f"**/*.{file_type}", loader_cls=loader_type)

    docs = loader.load()

    return docs

pdf_docs = load_files(path="./data/", file_type="pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(pdf_docs)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Loaded {len(pdf_docs)} pages")
print(f"Created {len(chunks)} chunks")
print("Documents stored in Chroma.")