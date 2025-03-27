from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_vector_retriever():
    # Load the document from hr_policy_doc.txt
    loader = TextLoader("hr_policy_doc.txt")
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Create embeddings using a HuggingFace model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a vector store with FAISS
    vector_store = FAISS.from_documents(texts, embeddings)

    # Return the retriever (optionally with search parameters)
    return vector_store.as_retriever(search_kwargs={"k": 3})