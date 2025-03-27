from langchain.chains import RetrievalQA
from langchain_community.llms.ollama import Ollama
from vector import get_vector_retriever

class RAGChatbot:
    def __init__(self):
        self.retriever = get_vector_retriever()
        self.llm = Ollama(model="gpt2")
        self.qa_chain = RetrievalQA(llm=self.llm, retriever=self.retriever)

    def get_response(self, query: str):
        return self.qa_chain.run(query)
