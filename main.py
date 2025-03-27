from fastapi import FastAPI, Query
from rag import RAGChatbot

# Initialize FastAPI app
app = FastAPI()
chatbot = RAGChatbot()

@app.get("/ask")
def ask_question(query: str = Query(..., description="User question about HR policies")):
    response = chatbot.get_response(query)
    return {"answer": response}