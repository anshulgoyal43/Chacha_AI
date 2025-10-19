import pdfplumber
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os 

load_dotenv()
st.set_page_config(layout="wide")

def upload(text):
    splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    result = vector_store.add_texts(splits)
    # st.write("Add texts result:", result)
    # st.write("Vector store size after add:", len(vector_store.store))
    return 

google_api_key = st.secrets["GOOGLE_API_KEY"]

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  
    google_api_key=google_api_key
)
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = InMemoryVectorStore(embeddings)
vector_store = st.session_state["vector_store"]

with st.sidebar:
    def extract(file_path):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    
    DOCS_DIR = os.path.abspath("./docs")
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)


    st.subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload files to the Knowledge Base (max 10 files):", accept_multiple_files=True, type=["pdf"])
        submitted = st.form_submit_button("Upload")
        if uploaded_files and submitted:
            if len(uploaded_files) > 5:
                st.error("You can upload a maximum of 5 files at a time.")
            else:
                with st.spinner("Processing files..."):
                    for uploaded_file in uploaded_files:
                        if uploaded_file.size > 2 * 1024 * 1024:
                            st.error(f"File {uploaded_file.name} exceeds the 2MB size limit.")
                            break
                        file_path = os.path.join(DOCS_DIR, uploaded_file.name)
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.read())
                        
                        # Extract text from the saved PDF
                        extracted_text = extract(file_path)
                        st.info(f"Extracted text from {uploaded_file.name} (preview):\n{extracted_text[:500]}")
                        if extracted_text.strip():
                            upload(extracted_text)  # Function to upload text to vector store
                            st.success(f"File {uploaded_file.name} uploaded successfully!")
                            # Display a preview of the extracted text (first 500 chars)
                            
                        else:
                            st.warning(f"No text uploaded from {uploaded_file.name}")
st.subheader("Chat with your AI Assistant, Chacha AI!")


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like user's uncle from haryana in english. "
            "Use the following context from the knowledge base to answer questions. "
            "If the context is relevant, use it in your answer. If not relevant or empty, "
            "answer based on your general knowledge.\n\n"
            "Context: {context}\n\n"
            "Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    # Get the last user message for retrieval
    last_message = state["messages"][-1]
    
    # Retrieve relevant documents from vector store
    context = ""
    if isinstance(last_message, HumanMessage):
        try:
            # print("Last message is: ", last_message.content)
            docs = vector_store.similarity_search(last_message.content, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            print(context)
            # print("Number of docs retrieved: ", len(docs))
        except Exception as e:
            # If retrieval fails or vector store is empty, continue without context
            context = "No relevant documents found in knowledge base."
            print("Error during retrieval: ", e)
    # Invoke prompt with context
    # print("Context is: ",context)
    prompt = prompt_template.invoke({"messages": state["messages"], "context": context})
    response = model.invoke(prompt)
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

if "memory" not in st.session_state:
    st.session_state["memory"] = MemorySaver()
memory = st.session_state["memory"]
app = workflow.compile(checkpointer=memory)

# --- 💬 Conversation history ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

user_input = st.chat_input("You:")
if user_input:
    st.session_state["messages"].append(HumanMessage(user_input))
    config = {"configurable": {"thread_id": "chat123"}}
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_text = ""
        response_placeholder = st.empty()
        # Use the RAG workflow app to stream the response
        for chunk, metadata in app.stream(
            {"messages": st.session_state["messages"]},
            config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage):
                response_text += chunk.content
                response_placeholder.markdown(response_text)

    st.session_state["messages"].append(AIMessage(response_text))

# st.write("Vector store size:", len(vector_store.store))
