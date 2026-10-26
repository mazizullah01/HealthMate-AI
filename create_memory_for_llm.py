from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
# step 3: Create  vector Embaeddings
# step 4: store embadding in FAISS
          