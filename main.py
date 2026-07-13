import os
from dotenv import load_dotenv
from operator import itemgetter

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

# ==========================================================================
# RAG IMPLEMENTATION: Without LCEL (Simple Function-Based Approach)
# ==========================================================================
print("Initializing Components .......")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    
    Provide a detailed answer:"""
)


def invoke_openai(model, prompt_template, prompt_input):
    summary_promt_template = PromptTemplate(
        input_variables=["information"], template=prompt_template
    )

    # openAI llm
    llm = ChatOpenAI(temperature=0, model=model)

    chain = summary_promt_template | llm

    response = chain.invoke(input={"information": prompt_input})
    return response


def invoke_ollama(model, prompt_template, prompt_input):
    summary_promt_template = PromptTemplate(
        input_variables=["information"], template=prompt_template
    )

    # local llm using ollama
    llm = ChatOllama(temperature=0, model=model)

    chain = summary_promt_template | llm

    response = chain.invoke(input={"information": prompt_input})
    return response


def format_docs(docs):
    """Format retrieved documents into single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the promt with context and questions
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return Content
    return response.content


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


def main():

    # ===================================================================
    # Invoke LLMs from OpenAI and Ollama
    # ===================================================================

    #     print("Hello world from langchain!")
    #     information = """
    #     Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman, known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion.

    # Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada. He received bachelor's degrees from the University of Pennsylvania in 1997 before moving to California, United States, to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.

    # In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI. In 2022, he acquired the social network Twitter, implementing significant changes and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017.

    # Musk was the largest donor in the 2024 U.S. presidential election, and is a supporter of global far-right figures, causes, and political parties. In early 2025, he served as senior advisor to United States president Donald Trump and as the de facto head of DOGE. After a public feud with Trump, Musk left the Trump administration and announced he was creating his own political party, the America Party.

    # Musk's political activities, views, and statements have made him a polarizing figure, especially following the COVID-19 pandemic. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation and promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE.
    #     """

    #     summary_template = """
    #     given the information {information} about a person I want you to create:
    #     1. A short summary
    #     2. two interesting facts about them
    #     """

    #     openai_response = invoke_openai("gpt-5", summary_template, information)
    #     print(openai_response.content)

    #     ollama_response = invoke_ollama("gemma3:270m", summary_template, information)
    #     print(ollama_response.content)

    # ================================================
    # RAG Retrival
    # ================================================
    print("Retrieving ...")

    # Query
    query = "What is Pinecone in machine learning?"

    # =========================================================================
    # Option 0: Raw invocation without RAG (for testing output without RAG)
    # =========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer")
    print(result_raw.content)

    # =========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: WITHOUT LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer")
    print(result_without_lcel)

    # =========================================================================
    # Option 2: Use implementation WITH LCEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer")
    print(result_with_lcel)


if __name__ == "__main__":
    main()
