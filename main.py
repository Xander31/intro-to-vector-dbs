import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from openai import embeddings

load_dotenv()

if __name__ == "__main__":
    print("Retrieving...")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    query = "What is Pinecone in ML?"
    chain = PromptTemplate.from_template(template=query) | llm
    #result = chain.invoke(input={})
    #print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings
    )

    #Retrieval Prompt
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    #Create a chain for passing a list of Documents to a model.
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)

    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result_retrieval = retrieval_chain.invoke(input={"input": query})
    print(result_retrieval["answer"])
