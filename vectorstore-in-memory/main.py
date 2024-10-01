import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

if __name__ == "__main__":
    pdf_path = "/Users/sanjeev/Projects/udemy/ice_breaker/textbook.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    question="what is the problem with theory"
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()

    # vectorstore = FAISS.from_documents(docs, embeddings) #store vectors in ram instead of pinecone server
    # vectorstore.save_local("faiss_index_react2") #store in hard disk instead of ram

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings
                                       ,allow_dangerous_deserialization=True)
    

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke(
        {
            "input" : question
        }
    )

    print(res["answer"])

