import streamlit as st
from chatbot import query_chatbot

def main():
    st.title("File Upload for Chatbot")

    # Provide the user a way to upload a file
    uploaded_file = st.file_uploader("Upload your CSV, PDF, or XML file", type=['csv', 'pdf', 'xml'])

    if uploaded_file is not None:
        st.write(f"File selected: {uploaded_file.name}")
        
        # Allow the user to provide a custom query (or use a default one)
        user_input = st.text_input("Enter your question", "Can you give me a brief summary of this file?")
        
        if st.button('Submit'):
            # Save the file to a temporary location to be processed by the chatbot
            file_path = f"./temp_{uploaded_file.name}"
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Query the chatbot with the uploaded file
            result = query_chatbot(file_path, user_input)
            
            # Display the result or a message if no response is returned
            if result:
                st.write("### Chatbot Response:")
                st.write(result)
            else:
                st.write("No response from the chatbot.")
    else:
        st.write("Please upload a file to proceed.")

if __name__ == "__main__":
    main()
