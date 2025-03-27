# RAG-Based HR Policy Chatbot

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built using **FastAPI** and **LangChain**. It answers questions related to HR policies by retrieving relevant documents and generating responses using an LLM (Large Language Model).

## Project Structure

```
├── main.py          # FastAPI application entry point
├── rag.py           # RAG-based chatbot implementation
├── vector.py        # FAISS vector store for document retrieval
├── hr_policy_doc.txt # HR policy document (knowledge base)
└── README.md        # Project documentation
```

## Requirements

Ensure you have **Python 3.8+** installed, then install the dependencies:

```bash
pip install fastapi uvicorn langchain langchain-ollama langchain-huggingface faiss-cpu
```

## How It Works

1. **Document Processing:** The HR policy document (`hr_policy_doc.txt`) is loaded, split into smaller chunks, and embedded using `sentence-transformers/all-MiniLM-L6-v2`.
2. **Vector Storage:** The embeddings are stored using **FAISS**, allowing efficient similarity-based retrieval.
3. **Retrieval-Augmented Generation (RAG):** When a user asks a question, the most relevant document chunks are retrieved and passed to the LLM (`llama3.2`) for response generation.
4. **FastAPI Endpoint:** Users can query the chatbot via the `/ask` API endpoint.

## API Usage

Run the FastAPI application with:

```bash
uvicorn main:app --reload
```

### Available Endpoint

#### `GET /ask`

**Description:** Asks a question about HR policies.

**Query Parameter:**
- `query` (string): The user's question about HR policies.

**Example Request:**

```bash
curl -X 'GET' 'http://127.0.0.1:8000/ask?query=What is the leave policy?' -H 'accept: application/json'
```

**Example Response:**
```json
{
  "answer": "Employees are entitled to 20 days of paid leave per year.",
  "CODE": "200"
}
```

## Notes
- Ensure that `hr_policy_doc.txt` exists in the project directory.
- The chatbot uses the `llama3.2` model; ensure it is available locally or specify an alternative model.
- Modify `vector.py` to use different document sources if needed.

## Future Enhancements
- Integrate a more advanced LLM for improved responses.
- Implement a frontend interface for user-friendly interactions.
- Add multi-document support for broader HR knowledge coverage.



