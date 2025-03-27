from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from vector import get_vector_retriever

class RAGChatbot:
    def __init__(self):
        # Initialize the retriever
        self.retriever = get_vector_retriever()
        
        # Initialize the LLM (updated to OllamaLLM)
        self.llm = OllamaLLM(model="llama3.2")
        
        # Define the prompt template for the QA task
        prompt = ChatPromptTemplate.from_template(
            "Based on the following context, answer the question:\n\n"
            "Context:\n{context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        )
        
        # Create a chain to combine documents using the LLM and prompt
        combine_docs_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        
        # Create the retrieval chain with the retriever and document chain
        self.qa_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=combine_docs_chain
        )

    def get_response(self, query: str):
        # Invoke the chain with the query and extract the answer
        result = self.qa_chain.invoke({"input": query})
        return result["answer"]  # The answer is typically under the "answer" key