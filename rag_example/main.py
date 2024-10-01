from dotenv import load_dotenv  
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

if __name__ == "__main__":
    print("retrieving")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    #What happens if we dont use RAG. ie dont query vector store for context

    query = "what is pinecone in machine learning?"
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    """
    A pinecone refers to a type of neural network accelerator that is designed
    to provide high-performance, low-power processing for machine learning applications. 
    Pinecones are typically used in edge computing devices, such as smartphones and IoT 
    devices, to enable on-device machine learning inference without relying on cloud-based
    servers. They are optimized for processing tasks such as image recognition, 
    natural language processing, and other AI algorithms.
    """

    #create vector store
    vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

    #this will be the prompt which we will be sending to the llm
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    """
        MAGIC stuff. will learn more later.
        Basically, this will take the query, get embeddings, get relevant embeddings 
        and pass everything to llm and get a response
    """
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})

    print(result['answer'])



