from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import streamlit as st

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Chatbot():
    def __init__(self):
        
        # Loading and Splitting Documents
        loader = TextLoader('./Hayao_Miyazaki.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Setting Up Embeddings and Pinecone Index
        embeddings = HuggingFaceEmbeddings()

        pinecone_api_key = os.getenv('PINECONE')
        # Initialize Pinecone client
        pinecone.init(api_key=pinecone_api_key, environment='gcp-starter')

        # Define Index Name
        index_name = "langchain-demo"

        # Checking Index
        if index_name not in pinecone.list_indexes():
            # Create new Index
            pinecone.create_index(name=index_name, metric="cosine", dimension=768)
            self.docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        else:
            # Link to the existing index
            self.docsearch = Pinecone.from_existing_index(index_name, embeddings)

        # Connecting to Hugging Face Model
        hf_key = os.getenv('HF_KEY')

        # Define the repo ID and connect to Mixtral model on Huggingface
        model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

        self.llm = HuggingFaceEndpoint(
            repo_id=model,
            temperature=0.8,
            top_k=50,
            huggingfacehub_api_token=hf_key
        )

        # Prompt Template Setup
        template = """
        The Human will ask questions about Hayao Miyazaki life. 
        If you don't know the answer, just say you don't know. 
        Keep the answer 3-4 sentences.
        
        Context: {context}
        Question: {question}
        Answer: 
        """

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Retriever: It takes your question, searches through all the stored information, 
        # and picks out the most relevant pieces. These pieces are then used to 
        # help generate a response to your question.
        self.retriever = self.docsearch.as_retriever()

    def get_response(self, question):
        # Retrieve context
        context_docs = self.retriever.get_relevant_documents(question)
        context = " ".join([doc.page_content for doc in context_docs])
        # Create the prompt with context and question
        prompt_text = self.prompt.format(context=context, question=question)
        # Get the answer from the LLM
        response = self.llm(prompt_text)
        return response

# Create an instance of the Chatbot
bot = Chatbot()

# Ask a question
user_input = input("Ask me anything: ")
result = bot.get_response(user_input)
print(result)


# STREAMLIT UI

# # Create an instance of the Chatbot
# bot = Chatbot()

# # Streamlit UI
# st.title("Hayao Miyazaki Chatbot")

# # Input field for user questions
# user_input = st.text_input("Ask me anything about Hayao Miyazaki:", "")

# if user_input:
#     result = bot.get_response(user_input)
#     st.text_area("Response", result)
