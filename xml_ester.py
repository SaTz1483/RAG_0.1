
import os
import xmltodict
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
import xmltodict
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.callbacks.manager import CallbackManager
file_path = "D:\\Carbon Emissions\\new_implementation\\final_result.txt"

def process_text_file_and_calculate_emissions(file_path, model="llama3.1", emission_factor=0.233, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Process a single text file with chain of thought, extract relevant components, and calculate carbon emissions.

    Parameters:
        file_path (str): The path to the text file to process.
        model (str): The LLM model to use for carbon emission calculation.
        emission_factor (float): The emission factor to use (default is 0.233 kg CO2e per kWh).

    Returns:
        chain, vectorstore, summary, run_chain_with_query (function to query the LLM).
    """

    # Step 1: Ensure the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Step 2: Load the text file
    try:
        with open(file_path, "r") as file:
            file_content = file.read()
        print(f"Text file successfully loaded. Number of characters: {len(file_content)}")
    except Exception as e:
        print(f"Error loading text file: {e}")
        return None

    # Step 3: Initialize a text splitter to split large files into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(file_content)

    # Convert each chunk into a LangChain Document object
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Step 4: Initialize HuggingFace embeddings
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print("HuggingFace embeddings loaded successfully.")
    except Exception as e:
        print(f"Error loading HuggingFace embeddings: {e}")
        return None

    # Step 5: Create a Chroma vector store
    persist_directory = "D:\\Carbon Emissions\\new_implementation\\vector_store\\text_files"
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        print("Chroma vectorstore successfully created.")
    except Exception as e:
        print(f"Error creating Chroma vectorstore: {e}")
        return None

    # Step 6: Initialize a retriever to get relevant content based on the user's query
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant chunks
        print("Retriever created successfully.")
    except Exception as e:
        print(f"Error creating retriever: {e}")
        return None

    # Step 7: Load the Ollama LLM
    try:
        llm = Ollama(model=model, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        print(f"Ollama model loaded: {llm.model}")
    except Exception as e:
        print(f"Error loading Ollama model: {e}")
        return None

    # Step 8: Create the chain of thought to calculate carbon emissions
    thought_prompt_template = """
    You are a helpful assistant for calculating carbon emissions based on the extracted data from the text file. Follow this format:

    Thought: I will start by extracting the relevant components (such as energy consumption, operational time, etc.) from the text file data.
    Action: Extract components and values like power consumption, operational time, and emission factor.
    Observation: The extracted relevant content is: {retrieved_content}.

    Thought: Now I will calculate carbon emissions using the formula: carbon_emissions = energy_consumption × operational_time × emission_factor.
    Action: Perform the calculation using the provided values and the emission factor .
    Observation: The calculated carbon emissions are: 

    Final Answer: Based on the extracted components and the provided emission factor, the total carbon emissions are {carbon_emissions} kg CO2e.
    """

    try:
        thought_prompt = PromptTemplate(
            input_variables=["retrieved_content", "emission_factor", "carbon_emissions"],
            template=thought_prompt_template
        )
        chain = LLMChain(llm=llm, prompt=thought_prompt)
        print("Chain of thought for carbon emissions calculation created successfully.")
    except Exception as e:
        print(f"Error creating chain of thought: {e}")
        return None

    # Step 9: Retrieve relevant content based on the user's query and calculate emissions
    def run_chain_with_query(query):
        try:
            relevant_docs = retriever.get_relevant_documents(query)
            if not relevant_docs:
                print("No relevant documents retrieved from the text file.")
                return None, None, None, ""
            retrieved_content = " ".join([doc.page_content for doc in relevant_docs])
            print(f"Retrieved content (first 500 chars): {retrieved_content[:500]}...")

            # Calculate carbon emissions
            try:
                carbon_emissions = "Calculated using extracted components"
                result = chain.run(
                    retrieved_content=retrieved_content,
                    emission_factor=emission_factor,
                    carbon_emissions=carbon_emissions
                )
                return result
            except Exception as e:
                print(f"Error during carbon emission calculation: {e}")
                return None
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return None, None, None, None

    return chain, vectorstore, run_chain_with_query


res = process_text_file_and_calculate_emissions(file_path)
print(res)