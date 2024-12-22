from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from data.document_loader import get_documents
from transformers import pipeline

# Load documents
docs = get_documents()

# Embedding setup
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Chroma vector store
db = Chroma.from_documents(docs, embedding_function)

# Define retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Define Hugging Face pipeline and LLM
local_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")
llm = HuggingFacePipeline(pipeline=local_pipeline)

# Other setup like memory and chain remains unchanged...


# Memory and prompt template
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
custom_template = """
You are an AI trained to answer questions based on a provided knowledge base. 
Given the following conversation history and a follow-up question, provide the most accurate and context-aware response.

Conversation History:
{chat_history}

Follow-up Question: {question}
Response:
"""
condense_prompt = PromptTemplate.from_template(custom_template)

# Conversational chain
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory, condense_question_prompt=condense_prompt
)

def process_query(question: str) -> str:
    """Processes the user query and returns an AI-generated answer."""
    response = conversational_chain({"question": question})
    return response.get("answer", "No relevant information found.")
