import os
import xmltodict
import re
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManager

file_path = "D:\\Carbon Emissions\\new_implementation\\final_result.txt"

def extract_components_for_emissions(file_path, model="llama3.1", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Extract necessary components from a text file for carbon emissions calculation.

    Parameters:
        file_path (str): The path to the text file to process.
        model (str): The LLM model to use for extraction.

    Returns:
        list: A list of dictionaries containing extracted components for carbon emissions calculation in a standardized format.
    """
    # Step 1: Ensure the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Step 2: Load the text file
    with open(file_path, "r") as file:
        file_content = file.read()

    # Step 3: Split the text content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    text_chunks = text_splitter.split_text(file_content)

    # Step 4: Create Document objects for the text chunks
    documents = [Document(page_content=chunk) for chunk in text_chunks]

    # Step 5: Embed the text chunks
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = Chroma.from_documents(documents, embedding)

    # Step 6: Set up the LLM model
    llm = Ollama(model=model, callback_manager=CallbackManager([]))

    # Step 7: Define the chain of thought prompt to extract necessary components
    extraction_prompt_template = """
    You are an expert in analyzing energy data for carbon emissions calculation.
    Your task is to read the provided content like a human expert and extract the following components:
    - Component Name: The name or identifier of the component.
    - Energy Consumption (kWh): The amount of energy consumed by a component in kilowatt-hours per month.
    - Power Usage (W): The power usage of a component in watts.
    - Operational Hours (h): If not directly available, calculate as `Energy Consumption (kWh) / Power Usage (W)`.
    - Carbon Emission Factor: The amount of CO2 emissions produced per unit of energy consumed.

    If the component is an LED or Fluorescent light and the power usage is not provided, assume it to be 15W for LED lights.

    Given the content below:
    {retrieved_content}
    Extract these components carefully and provide the values in the following format:
    Component Name: <value>
    Energy Consumption (kWh): <value>
    Power Usage (W): <value>
    Operational Hours (h): <value>
    Carbon Emission Factor: <value>
    """

    try:
        extraction_prompt = PromptTemplate(
            input_variables=["retrieved_content"],
            template=extraction_prompt_template
        )
        chain = LLMChain(llm=llm, prompt=extraction_prompt)
    except Exception as e:
        raise RuntimeError(f"Error creating extraction chain: {e}")

    # Step 8: Retrieve relevant content and extract components
    def extract_components(query):
        try:
            relevant_docs = vectorstore.similarity_search(query)
            if not relevant_docs:
                raise ValueError("No relevant documents retrieved from the text file.")
            retrieved_content = " ".join([doc.page_content for doc in relevant_docs])

            # Run the chain to extract components
            extracted_results = chain.run(retrieved_content=retrieved_content)

            # Split results into multiple components if applicable
            extracted_components_list = re.split(r'\n\s*\n', extracted_results)
            extracted_components = []
            for result in extracted_components_list:
                # Parse the extracted result into a standardized format
                components = {
                    "Component Name": re.search(r"Component Name: (.+)", result).group(1).strip() if re.search(r"Component Name: (.+)", result) else "Unknown",
                    "Energy Consumption (kWh)": float(re.search(r"Energy Consumption \(kWh\): ([0-9\.]+)", result).group(1)) if re.search(r"Energy Consumption \(kWh\): ([0-9\.]+)", result) else 0,
                    "Power Usage (W)": float(re.search(r"Power Usage \(W\): ([0-9\.]+)", result).group(1)) if re.search(r"Power Usage \(W\): ([0-9\.]+)", result) else 0,
                    "Operational Hours (h)": float(re.search(r"Operational Hours \(h\): ([0-9\.]+)", result).group(1)) if re.search(r"Operational Hours \(h\): ([0-9\.]+)", result) else 0,
                    "Carbon Emission Factor": float(re.search(r"Carbon Emission Factor: ([0-9\.]+)", result).group(1)) if re.search(r"Carbon Emission Factor: ([0-9\.]+)", result) else 0
                }
                extracted_components.append(components)

            return extracted_components
        except Exception as e:
            raise RuntimeError(f"Error extracting components: {e}")

    # Extract components based on a sample query
    components_query = "carbon emissions calculation components"
    extracted_components = extract_components(components_query)

    return extracted_components

def calculate_carbon_emissions(components_list):
    """
    Calculate carbon emissions based on extracted components for multiple devices.

    Parameters:
        components_list (list): A list of dictionaries containing extracted components necessary for carbon emissions calculation.

    Returns:
        list: A list of dictionaries containing calculated carbon emissions and context in a standardized format for each component.
    """
    results = []
    for components in components_list:
        try:
            component_name = components.get("Component Name", "Unknown")
            energy_consumption = components.get("Energy Consumption (kWh)", 0)
            power_usage = components.get("Power Usage (W)", 0)
            operational_hours = components.get("Operational Hours (h)", 0)
            emission_factor = components.get("Carbon Emission Factor", 0.233)

            # Skip components that do not have sufficient data
            if energy_consumption == 0 and power_usage == 0:
                results.append({
                    "Component Name": component_name,
                    "Carbon Emissions (kg CO2e)": "Insufficient data for calculation"
                })
                continue

            # Example calculation using energy consumption
            carbon_emissions = energy_consumption * emission_factor

            # Additional calculation using power usage and operational hours if available
            if power_usage > 0 and operational_hours > 0:
                carbon_emissions += (power_usage * operational_hours / 1000) * emission_factor

            results.append({
                "Component Name": component_name,
                "Carbon Emissions (kg CO2e)": carbon_emissions
            })
        except Exception as e:
            results.append({
                "Component Name": components.get("Component Name", "Unknown"),
                "Carbon Emissions (kg CO2e)": f"Error calculating emissions: {e}"
            })

    return results

# Example usage
components_list = extract_components_for_emissions(file_path)
print("Extracted Components:", components_list)
carbon_emissions_info_list = calculate_carbon_emissions(components_list)
print("Calculated Carbon Emissions Info:", carbon_emissions_info_list)