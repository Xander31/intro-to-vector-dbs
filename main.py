import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from openai import embeddings
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs (docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

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

    #2nd implementation - LCEL (Langchain Expression Language)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as conscise as possible.
    Always say "thanks for asking" at the end of the answer.
    
    {context}
    
    Question: {question}
    
    Helpful Answer:
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(input=query)
    print(res.content)