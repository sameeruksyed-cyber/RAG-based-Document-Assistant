import os
from dotenv import load_dotenv
from langchain.document_loaders import unstructuredMarkdownLoader, UnstructuredFileLoader, PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredHTMLLoader, UnstructuredEPubLoader
load_dotenv()

#load documents from pdf, docx, pptx, txt, csv, xlsx, html, md, epub, and other formats

#define the directory where the documents are stored
docs_directory = '.docs'
documents = []

file_path = [os.path.join(docs_directory, file) for file in os.listdir(docs_directory) if os.path.isfile(os.path.join(docs_directory, file))]

