# from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
# from langchain.callbacks.manager import CallbackManager
# from file_processor import process_csv_with_chain_of_thought, process_pdf_with_chain_of_thought, process_xml_with,calculate_carbon_emissions
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# import os

# def handle_uploaded_file(file_path, user_query, context=None):
#     """
#     Handles file uploads and processes the file based on its type.
#     Returns qa_chain, context, summary, and a function to run the query.
#     """
#     # Extract file extension
#     _, file_extension = os.path.splitext(file_path)

#     # Check the file extension and call the appropriate processing function
#     try:
#         if file_extension.lower() == ".csv":
#             print(f"Processing CSV file: {file_path}")
#             qa_chain, vectorstore, summary, run_chain_with_query = process_csv_with_chain_of_thought(file_path)
#             context = context or "This is a CSV file with data on devices and power consumption for carbon emissions."
#         elif file_extension.lower() == ".pdf":
#             print(f"Processing PDF file: {file_path}")
#             qa_chain, vectorstore, summary, run_chain_with_query = process_pdf_with_chain_of_thought(file_path)
#             context = context or "This is a PDF document that the assistant will help analyze."
#             summary = "Processed 1 PDF document."
#         elif file_extension.lower() == ".xml":
#             print(f"Processing XML file: {file_path}")
#             # Provide default context if none is provided
#             context = context or "This is an XML document with structured data."
#             qa_chain, vectorstore, summary, run_chain_with_query = process_xml_with(file_path)
#             summary = "Processed 1 XML file."
#         else:
#             print(f"Unsupported file type: {file_extension}")
#             return None, None, None, None

#         return qa_chain, context, summary, run_chain_with_query

#     except Exception as e:
#         print(f"Error processing file: {e}")
#         return None, None, None, None


# def query_chatbot(file_path, user_input):
#     """
#     Function to query the chatbot with a given file and user input.
#     """

#     # Process the file and get the QA chain, context, and summary
#     qa_chain, context, summary, run_chain_with_query = handle_uploaded_file(file_path, user_input)

#     # If the QA chain could not be created, handle the error
#     if qa_chain is None:
#         print("File could not be processed. Please upload a supported file.")
#         return None

#     # Retrieve relevant content using the run_chain_with_query function
#     chain, vectorstore, summary, retrieved_content = run_chain_with_query(user_input)

#     if not retrieved_content:
#         print("No relevant content could be retrieved from the file.")
#         return None

#     # Query the chatbot using the run method
#     try:
#         # Generate the initial answer based on the context (optional, could be empty)
#         answer = f"I will start by analyzing the {context} data."
#         # Ensure input_documents is removed if not needed
#         final_result = qa_chain.run(retrieved_content=retrieved_content, query=user_input, summary=summary, context=context, answer=answer)

#         # extracted_data = f"Final Result: {final_result}"
#         with open("final_result.txt", "w") as file:
#             file.write(final_result)
          
        
#         return final_result
#     except Exception as e:
#         print(f"Error during query execution: {e}")
#         return None


from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from file_processor import process_csv_with_chain_of_thought, process_pdf_with_chain_of_thought, process_xml_with
from xml_test import calculate_carbon_emissions
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

def handle_uploaded_file(file_path, user_query, context=None):
    """
    Handles file uploads and processes the file based on its type.
    Returns qa_chain, context, summary, and a function to run the query.
    """
    # Extract file extension
    _, file_extension = os.path.splitext(file_path)

    # Check the file extension and call the appropriate processing function
    try:
        if file_extension.lower() == ".csv":
            print(f"Processing CSV file: {file_path}")
            qa_chain, vectorstore, summary, run_chain_with_query = process_csv_with_chain_of_thought(file_path)
            context = context or "This is a CSV file with data on devices and power consumption for carbon emissions."
        elif file_extension.lower() == ".pdf":
            print(f"Processing PDF file: {file_path}")
            qa_chain, vectorstore, summary, run_chain_with_query = process_pdf_with_chain_of_thought(file_path)
            context = context or "This is a PDF document that the assistant will help analyze."
            summary = "Processed 1 PDF document."
        elif file_extension.lower() == ".xml":
            print(f"Processing XML file: {file_path}")
            # Provide default context if none is provided
            context = context or "This is an XML document with structured data."
            qa_chain, vectorstore, summary, run_chain_with_query = process_xml_with(file_path)
            summary = "Processed 1 XML file."
        else:
            print(f"Unsupported file type: {file_extension}")
            return None, None, None, None

        return qa_chain, context, summary, run_chain_with_query

    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None, None, None

def query_chatbot(file_path, user_input):
    """
    Function to query the chatbot with a given file and user input.
    """

    # Process the file and get the QA chain, context, and summary
    qa_chain, context, summary, run_chain_with_query = handle_uploaded_file(file_path, user_input)

    # If the QA chain could not be created, handle the error
    if qa_chain is None:
        print("File could not be processed. Please upload a supported file.")
        return None

    # Retrieve relevant content using the run_chain_with_query function
    chain, vectorstore, summary, retrieved_content = run_chain_with_query(user_input)

    if not retrieved_content:
        print("No relevant content could be retrieved from the file.")
        return None

    # Query the chatbot using the run method
    try:
        # Generate the initial answer based on the context (optional, could be empty)
        answer = f"I will start by analyzing the {context} data."
        # Ensure input_documents is removed if not needed
        final_result = qa_chain.run(retrieved_content=retrieved_content, query=user_input, summary=summary, context=context, answer=answer)

        # Save the result to a text file
        with open("final_result.txt", "w") as file:
            file.write(final_result)
        print("Data extracted and saved to final_result.txt")
        
        # Now run the second chain (to calculate carbon emissions) after extracting data
        # v_path_vector_store = 'D:\\Carbon Emissions\\new_implementation\\vector_store\\textst'
        # emissions_result = calculate_carbon_emissions("final_result.txt", v_path_vector_store)
        # print(emissions_result)
        return final_result
    except Exception as e:
        print(f"Error during query execution: {e}")
        return None