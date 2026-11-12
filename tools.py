import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".txt",
    ".html",
    ".md",
}


def load_documents(docs_directory: Path):
    """Load supported documents from the project's document directory."""
    if not docs_directory.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {docs_directory}")

    documents = []
    for file_path in sorted(docs_directory.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".txt": TextLoader,
            ".html": UnstructuredHTMLLoader,
            ".md": UnstructuredMarkdownLoader,
        }
        documents.extend(loaders[file_path.suffix.lower()](str(file_path)).load())

    if not documents:
        raise ValueError(f"No supported documents found in {docs_directory}")
    return documents


def create_retrieval_chain_for_documents(documents):
    text_splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    ).split_documents(documents)
    if not text_splits:
        raise ValueError("The documents did not contain any text to index.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(text_splits, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to the .env file.")

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context below.
If the answer is not contained in the context, say "I don't know."

Context:
{context}

Question:
{input}
"""
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


def answer_question(question: str, docs_directory: Path | None = None) -> str:
    load_dotenv()
    directory = docs_directory or Path(__file__).parent / "docs"
    chain = create_retrieval_chain_for_documents(load_documents(directory))
    response = chain.invoke({"input": question})
    return response["answer"]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    question = os.getenv(
        "QUESTION",
        "What is the main experience in the document?",
    )
    answer = answer_question(question)
    print("\nAnswer:\n", textwrap.fill(answer, width=80))


if __name__ == "__main__":
    main()
