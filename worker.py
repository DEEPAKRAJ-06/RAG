import os
import torch
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# ✅ Set a writable cache directory for both Hugging Face Hub and Transformers
# ✅ Use `/tmp/huggingface_cache/` instead of `./huggingface_cache`
os.environ["HF_HOME"] = "/tmp/huggingface_cache"

# Check for GPU availability
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_pipeline = None
embeddings = None


def init_llm():
    global llm_pipeline, embeddings

    # Ensure API key is set in Hugging Face Spaces
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in environment variables.")

    model_id = model_id = "tiiuae/falcon-rw-1b"  # Falcon-1B model

    hf_pipeline = pipeline("text-generation", model=model_id, device=DEVICE)
    llm_pipeline = HuggingFacePipeline(pipeline=hf_pipeline)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )


def process_document(document_path):
    global conversation_retrieval_chain

    # Ensure LLM and embeddings are initialized
    if not llm_pipeline or not embeddings:
        init_llm()

    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    
    # Load or create ChromaDB
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=persist_directory)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 6})
    
    conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_pipeline, retriever=retriever
    )


def process_prompt(prompt):
    global conversation_retrieval_chain, chat_history

    if not conversation_retrieval_chain:
        return "No document has been processed yet. Please upload a PDF first."

    output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    answer = output["answer"]
    
    chat_history.append((prompt, answer))
    
    return answer
