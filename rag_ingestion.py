import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    # data loading
    print("Loading Data .....")
    loader = UnstructuredLoader(
        file_path="./mediumblog.txt", chunking_strategy="basic", max_characters=1000000
    )
    document = loader.load()
    print(document)

    # splitting documents
    print("Splitting Documents ....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    # Embeddings
    embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))

    # Store embeddings in Vector DB Pinecone
    print("Storing Embeddings in Pine Cone ....")
    PineconeVectorStore.from_documents(
        documents=texts, embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )

    print("Finished storing embeddings in PineCone ....")
