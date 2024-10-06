import pandas as pd
from langchain_community.document_loaders import CSVLoader
import os
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
import pandas as pd
import xmltodict
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_community.document_loaders import CSVLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

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



def process_csv_with_chain_of_thought(file_path, model="llama3.1", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Process CSV file with chain of thought and return the QA chain and a function to run the query."""
    
    # Step 1: Load the CSV file using pandas
    try:
        df = pd.read_csv(file_path)
        print(f"CSV file loaded successfully. It contains {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # Step 2: Load the CSV content into LangChain documents using CSVLoader
    try:
        loader = CSVLoader(file_path)
        documents = loader.load()
        print(f"CSVLoader successfully loaded the documents. Number of documents: {len(documents)}")
    except Exception as e:
        print(f"Error using CSVLoader: {e}")
        return None

    # Step 3: Summarize the rows and columns for the chain of thought
    summary = f"The CSV file contains {df.shape[0]} rows and {df.shape[1]} columns. The column names are: {', '.join(df.columns)}."
    print(f"Summary: {summary}")

    # Step 4: Initialize HuggingFace embeddings
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print("HuggingFace embeddings loaded successfully.")
    except Exception as e:
        print(f"Error loading HuggingFace embeddings: {e}")
        return None

    # Step 5: Create a Chroma vector store
    persist_directory = "D:\\Carbon Emissions\\new_implementation\\vector_store\\csvam"
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

    # Step 8: Create the chain of thought with a prompt
    thought_prompt_template = """
    You are a helpful assistant for processing CSV files. Follow this format:

    Thought: I will start by summarizing the CSV data.
    Action: Summarize the number of rows and columns.
    Observation: {summary}

    Thought: Now I need to retrieve relevant content from the CSV file.
    Action: Retrieve relevant content based on the query.
    Observation: Retrieved relevant content is: {retrieved_content}.

    Final Answer: I have summarized the data and provided relevant content based on the user's query.

    Context: {context}
    User: {query}
    AI: {answer}
    """
    
    try:
        thought_prompt = PromptTemplate(
            input_variables=["context", "summary", "query", "retrieved_content", "answer"],
            template=thought_prompt_template
        )
        chain = LLMChain(llm=llm, prompt=thought_prompt)
        print("Chain of thought created successfully.")
    except Exception as e:
        print(f"Error creating chain of thought: {e}")
        return None

    # Step 9: Retrieve relevant content based on the user's query
    def run_chain_with_query(query):
        try:
            relevant_docs = retriever.get_relevant_documents(query)
            if not relevant_docs:
                print("No relevant documents retrieved from the CSV.")
                return None
            retrieved_content = " ".join([doc.page_content for doc in relevant_docs])
            print(f"Retrieved content (first 500 chars): {retrieved_content[:500]}...")
            return chain, vectorstore, summary, retrieved_content
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return None, None, None, None

    return chain, vectorstore, summary, run_chain_with_query


def process_pdf_with_chain_of_thought(file_path, model="llama3.1", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Process a single PDF file with chain of thought, return a QA chain, and prepare for user queries."""

    # Step 1: Ensure the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Step 2: Load the PDF file using PyPDFLoader
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"PDFLoader successfully loaded the documents. Number of pages: {len(documents)}")
    except Exception as e:
        print(f"Error loading PDF file: {e}")
        return None

    # Step 3: Initialize HuggingFace embeddings
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print("HuggingFace embeddings loaded successfully.")
    except Exception as e:
        print(f"Error loading HuggingFace embeddings: {e}")
        return None

    # Step 4: Create a Chroma vector store
    persist_directory = "D:\\Carbon Emissions\\new_implementation\\vector_store\\pdf"
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

    # Step 5: Initialize a retriever to get relevant content based on the user's query
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant chunks
        print("Retriever created successfully.")
    except Exception as e:
        print(f"Error creating retriever: {e}")
        return None

    # Step 6: Load the Ollama LLM
    try:
        llm = Ollama(model=model, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        print(f"Ollama model loaded: {llm.model}")
    except Exception as e:
        print(f"Error loading Ollama model: {e}")
        return None

    # Step 7: Create the chain of thought with a prompt
    thought_prompt_template = """
    You are a helpful assistant for processing PDF files. Follow this format:

    Thought: I will start by summarizing the PDF content.
    Action: Summarize the number of pages and key information.
    Observation: {summary}

    Thought: Now I need to retrieve relevant content from the PDF based on the user's query.
    Action: Retrieve relevant content based on the query.
    Observation: Retrieved relevant content is: {retrieved_content}.

    Final Answer: I have summarized the content and provided relevant information based on the user's query.

    Context: {context}
    User: {query}
    AI: {answer}
    """
    
    try:
        thought_prompt = PromptTemplate(
            input_variables=["context", "summary", "query", "retrieved_content", "answer"],
            template=thought_prompt_template
        )
        chain = LLMChain(llm=llm, prompt=thought_prompt)
        print("Chain of thought created successfully.")
    except Exception as e:
        print(f"Error creating chain of thought: {e}")
        return None

    # Step 8: Summarize the PDF for the chain of thought
    summary = f"The PDF file contains {len(documents)} pages of data. Key topics and sections include summaries and detailed analysis."

    # Step 9: Retrieve relevant content based on the user's query
    def run_chain_with_query(query):
        try:
            relevant_docs = retriever.get_relevant_documents(query)
            if not relevant_docs:
                print("No relevant documents retrieved from the PDF.")
                return None, None, None, ""
            retrieved_content = " ".join([doc.page_content for doc in relevant_docs])
            print(f"Retrieved content (first 500 chars): {retrieved_content[:500]}...")
            return chain, vectorstore, summary, retrieved_content
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return None, None, None, None

    return chain, vectorstore, summary, run_chain_with_query    

    # Load the CSV file using pandas and LangChain CSVLoader
   



    

def process_xml_with(file_path, model="llama3.1", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Process XML file with chain of thought and return the QA chain and a function to run the query."""
    
    # Ensure the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Load and parse the XML file
    try:
        with open(file_path, 'r') as xml_file:
            xml_content = xml_file.read()
        xml_dict = xmltodict.parse(xml_content)
        xml_text = str(xml_dict)  # Convert the parsed XML to a string for text processing
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        return None

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_text(xml_text)

    # Convert the text splits into LangChain Documents
    documents = [Document(page_content=split) for split in all_splits]

    # Initialize embeddings for the XML processing
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None

    # Store the embeddings in Chroma vector store
    persist_directory = "D:\\Carbon Emissions\\chatbot_project\\chroma_langchain_db\\vec_store\\xmloan"
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

    # Load the vector store (optional if persisting between sessions)
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embedding_model
        )
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

    print("Loaded XML content into Chroma vector store.")

    # Initialize Ollama model for the custom QA chain
    try:
        llm = Ollama(
            model=model,
            verbose=True,
            callbacks=[StreamingStdOutCallbackHandler()]  # Replacing deprecated callback_manager
        )
    except Exception as e:
        print(f"Error loading Ollama model: {e}")
        return None

    # Chain of Thought Template for LLMChain
    thought_prompt_template = """
    You are a helpful assistant for processing XML files. Based on the provided context, follow this format:

    Thought: I will review the XML data within the context provided.
    Action: Summarize the key elements and values from the XML data.
    Observation: Retrieved relevant content is: {retrieved_content}.

    Final Answer: Based on the user's query and the given context, here is a summary of the XML file.
    """

    try:
        thought_prompt = PromptTemplate(
            input_variables=["retrieved_content", "query", "context"],
            template=thought_prompt_template
        )
        chain = LLMChain(llm=llm, prompt=thought_prompt)
        print("LLMChain for XML processing created successfully.")
    except Exception as e:
        print(f"Error creating LLMChain: {e}")
        return None

    # Retrieve relevant content based on the user's query
    def run_chain_with_query(query):
        try:
            relevant_docs = vectorstore.as_retriever().get_relevant_documents(query)
            if not relevant_docs:
                print("No relevant documents retrieved from the XML.")
                return None
            retrieved_content = " ".join([doc.page_content for doc in relevant_docs])
            print(f"Retrieved content (first 500 chars): {retrieved_content[:500]}...")
            return chain, vectorstore, None, retrieved_content  # No explicit summary for XML, so returning None
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return None, None, None, None

    # Return the chain, vectorstore, and a function to run the query
    return chain, vectorstore, None, run_chain_with_query

