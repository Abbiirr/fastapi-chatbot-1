from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader


def get_documents():
    loader = TextLoader("banking_products.txt")
    raw_docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = [
        Document(page_content=chunk.strip(), metadata={"category": "saving" if "save" in chunk.lower() else "general"})
        for doc in raw_docs
        for chunk in text_splitter.split_text(doc.page_content)
    ]
    return documents
