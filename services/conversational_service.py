from transformers import pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from data.document_loader import get_documents
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Load documents
docs = get_documents()
if not docs:
    raise ValueError("No documents loaded.")
else:
    print(f"Loaded {len(docs)} documents.")

try:
    zero_shot_classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
    logger.info("Zero-shot classification pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load zero-shot-classification pipeline: {e}")
    raise

try:
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Smaller embedding model
    logger.info("Embedding function initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize embedding function: {e}")
    raise

# Create Chroma vector store
try:
    db = Chroma.from_documents(docs, embedding_function, persist_directory="chroma_db")  # Use persistent directory
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    logger.info("Chroma vector store initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Chroma vector store: {e}")
    raise

# Define retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# test retriever
retrieved_docs = retriever.get_relevant_documents("What are the best options for saving money?")
print(retrieved_docs)

# tokenizer = AutoTokenizer.from_pretrained("cuneytkaya/fintech-chatbot-t5")
# model = AutoModelForSeq2SeqLM.from_pretrained("cuneytkaya/fintech-chatbot-t5")
# Define Hugging Face pipeline and LLM
try:
    local_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=local_pipeline)
    logger.info("Text generation pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load text generation pipeline: {e}")
    raise

# Memory and prompt template
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
custom_template = """
You are an AI assistant with access to a financial knowledge base. Based on the conversation history and the following question, provide a clear, detailed, and accurate response.

Conversation History:
{chat_history}

Follow-up Question: {question}

Provide a detailed response:
"""
condense_prompt = PromptTemplate.from_template(custom_template)

# Conversational chain
try:
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_prompt,
    )
    logger.info("Conversational chain initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize conversational chain: {e}")
    raise


# Define utility functions
def get_intent(question: str) -> str:
    """
    Predicts the user's intent using zero-shot classification.
    """
    try:
        result = zero_shot_classifier(question, ["category", "definition", "unknown"])
        return result["labels"][0]  # Get the highest confidence label
    except Exception as e:
        logger.error(f"Error in zero-shot classification: {e}")
        return "unknown"


def process_query(question: str) -> str:
    """
    Processes the query dynamically based on inferred intent and retrieved context.
    """
    try:
        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(question)
        if not retrieved_docs:
            return "I couldn't find relevant information in the knowledge base."

        # Combine the context from the retrieved documents
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # If the retrieved context doesn't match a specific pattern, use a default answer
        if question.lower() in ["who are you?", "what are you?", "introduce yourself"]:
            return "I am a financial knowledge assistant here to help you with your queries."

        # Prepare input for the model
        input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        # Generate response
        response = local_pipeline(input_text, max_new_tokens=100)
        generated_text = response[0]["generated_text"]

        # If the generated text seems out of context, fallback to default answer
        if not generated_text or "personal loan" in generated_text.lower():  # Example check
            return "I'm here to assist you with financial queries. Please ask a relevant question."

        return generated_text

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return "An error occurred while processing your query."


# Example queries for debugging
if __name__ == "__main__":
    questions = [
        "What are the types of accounts?",
        "What is a savings account?",
        "Explain current account.",
        "Tell me about accounts."
    ]
    for question in questions:
        print(f"Q: {question}")
        print(f"A: {process_query(question)}")
        print("-" * 50)