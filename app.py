import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from tools import (
    SUPPORTED_EXTENSIONS,
    create_retrieval_chain_for_documents,
    load_documents,
)


ROOT_DIR = Path(__file__).parent
DOCS_DIR = ROOT_DIR / "docs"

load_dotenv(ROOT_DIR / ".env")

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📚",
    layout="centered",
)


@st.cache_resource(show_spinner="Indexing documents and loading the assistant...")
def get_retrieval_chain(docs_directory: str):
    documents = load_documents(Path(docs_directory))
    return create_retrieval_chain_for_documents(documents)


def display_source_documents(context: list) -> None:
    if not context:
        return

    with st.expander("Sources used"):
        seen_sources = set()
        for document in context:
            source = document.metadata.get("source", "Unknown document")
            page = document.metadata.get("page")
            source_label = Path(source).name
            if page is not None:
                source_label += f" (page {page + 1})"

            if source_label in seen_sources:
                continue
            seen_sources.add(source_label)
            st.markdown(f"- `{source_label}`")


def main() -> None:
    st.title("📚 Document Assistant")
    st.write("Ask questions about the documents in the `docs` folder.")

    with st.sidebar:
        st.header("Settings")
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        st.caption(f"Model: `{model_name}`")
        st.subheader("Uploaded documents")
        document_files = sorted(
            file_path
            for file_path in DOCS_DIR.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ) if DOCS_DIR.is_dir() else []
        if document_files:
            for file_path in document_files:
                st.markdown(f"- `{file_path.name}`")
        else:
            st.caption("No supported documents found.")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        if st.button("Rebuild document index"):
            get_retrieval_chain.clear()
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                display_source_documents(message.get("context", []))

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            chain = get_retrieval_chain(str(DOCS_DIR))
            with st.spinner("Searching the documents..."):
                response = chain.invoke({"input": question})
            answer = response["answer"]
            context = response.get("context", [])
            st.markdown(answer)
            display_source_documents(context)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "context": context,
                }
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            message = str(error)
            st.error(message)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {message}", "context": []}
            )


if __name__ == "__main__":
    main()
