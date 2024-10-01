# take an article, split it to chunks and store it
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

if __name__ == "__main__":
    load_dotenv()
    print("Ingesting...")
    
    loader = TextLoader("/Users/sanjeev/Projects/udemy/ice_breaker/mediumblog1.txt")
    
    document = loader.load() #will load txt file to a langchain document
    
    print("Splitting....")

    #create chunks from the document
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) 
    texts = text_splitter.split_documents(document)
    
    #create embedding
    embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))

    #store embeddings in pinecone
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    
    print("finish")
