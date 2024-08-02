import streamlit as st
from main import Chatbot  # Replace with your actual module name

# Initialize the chatbot
chatbot = Chatbot()

def main():
    st.title("Berry Chatbot")

    # Display a text input box for the user to ask a question
    question = st.text_input("Ask a question:")

    # Display a button to submit the question
    if st.button("Submit"):
        if question:
            # Get the response from the chatbot
            response = chatbot.get_response(question)
            # Display the response
            st.write("Answer:", response)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
