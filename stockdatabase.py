
import os
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import time

pdf_folder_path = r"C:\Users\dewakar\Documents\angular project\some-investment-books-master"

documents = []
start = time.time()

for file in os.listdir(pdf_folder_path):
    if file.endswith(".pdf"):
        try:
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            print({file})
        except Exception as e:
            print(f"Skipping '{file}' due to error: {e}")

end = time.time()
print(f" Time taken: {end - start:.2f} seconds for 100 chunks")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)


from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  



texts = [doc.page_content for doc in chunked_documents]
start = time.time()
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=False,
    convert_to_numpy=True, 
)

end = time.time()
print(f" Time taken: {end - start:.2f} seconds for 100 chunks")

client = chromadb.PersistentClient(path=r"C:\Users\dewakar\Documents\angular project\project sample\database")
collection = client.get_or_create_collection(name="my_collection")

collection.add(
    documents= embeddings 
)