import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# step 1: Setup LLM(Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
huggingface_repo_id="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEmbeddings(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_Kwargs={"token":HF_TOKEN,
                      "max_lenght":"512"
                      }
    )
    return llm

# step 2: Connect LLM with FAISS and Create chain