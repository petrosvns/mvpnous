import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
import pinecone

# Load environment variables
load_dotenv()

class ChatBot():
    def __init__(self):
        # Load API keys from .env
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.hf_api_key = os.getenv("HF_API_KEY")

        # Load and process the text file
        self.loader = TextLoader('brain_long.txt')
        self.documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = self.text_splitter.split_documents(self.documents)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings()

        # Initialize Pinecone
        pinecone.init(api_key=self.pinecone_api_key, environment="gcp-starter")
        self.index_name = "langchain-demo"

        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(name=self.index_name, metric="cosine", dimension=768)
            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
        else:
            self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)

        # Initialize Hugging Face model
        self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            repo_id=self.repo_id, 
            model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50}, 
            huggingfacehub_api_token=self.hf_api_key
        )
