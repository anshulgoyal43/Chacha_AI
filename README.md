# Chacha AI

Chacha AI is a Streamlit-based conversational assistant that leverages Google Gemini and LangChain for Retrieval-Augmented Generation (RAG) over custom PDF knowledge bases. The assistant responds in English, mimicking the style of a user's uncle from Haryana.

## Features
- **PDF Knowledge Base Upload:** Upload up to 5 PDF files (max 2MB each) to build a searchable knowledge base.
- **Text Extraction:** Extracts text from PDFs using `pdfplumber`.
- **Vector Store:** Stores document embeddings in-memory for fast similarity search.
- **Conversational RAG:** Answers user queries using context from uploaded documents, or general knowledge if context is missing.
- **Gemini Integration:** Uses Google Gemini (via LangChain) for chat and embeddings.
- **Streamlit UI:** Simple, wide-layout chat interface with file upload sidebar.

## Setup

### Prerequisites
- Python 3.8+
- Google API Key (for Gemini)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/anshulgoyal43/Chacha_AI.git
   cd Chacha_AI
   ```
2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your `.env` file (see example below).

### .env Example
```
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
GOOGLE_API_KEY=your_google_api_key
```

## Usage
1. Start the Streamlit app:
   ```bash
   streamlit run main.py
   ```
2. Open the app in your browser (usually at `http://localhost:8501`).
3. Use the sidebar to upload PDF files to the knowledge base.
4. Chat with Chacha AI in the main window.

## File Structure
- `main.py` — Main Streamlit app
- `requirements.txt` — Python dependencies
- `docs/` — Uploaded PDF files
- `.env` — Environment variables (API keys)

## Technologies Used
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [Google Gemini](https://ai.google.dev/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)

## License
MIT License

## Author
[Anshul Goyal](https://github.com/anshulgoyal43)
