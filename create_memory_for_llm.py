from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# step 1: load raw PDF(s)
DATA_PATH="data/"
def load_pdf_files(data):
    loader  = DirectoryLoader(data,
        glob='*.pdf',
        loader_cls=PyPDFLoader)

    documents=loader.load()
    return documents

documents= load_pdf_files(data=DATA_PATH)
print("Length of PDF pages:", len(documents))

# step 2: Create chunks
def create_chunks(extracted_data):
    text_spiltter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_spiltter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print("Length of Text Chunks", len(text_chunks))

# step 3: Create  vector Embaeddings
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model


embedding_model=get_embedding_model()


# step 4: store embadding in FAISS
          
