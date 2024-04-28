import os
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.streaming_write import write
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_openai.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback


# Sidebar contents
with st.sidebar:
    st.title('ChatISO')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot designed to help query techincal ISO documents.
    ''')
    add_vertical_space(2)
    st.write('Made by [Adam Myers](https://github.com/AdamM1909)')

def main():

    # Load OpenAI API key.
    load_dotenv()

    # Add header.
    st.header("ChatISO")

    # Upload PDF.
    if (pdf := st.file_uploader("Upload your PDF", type='pdf')) is not None: 
        pdf_reader = PdfReader(pdf)
        
        # Chunk PDF.
        text = ""
        for page in pdf_reader.pages: text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text=text)
        
        # Get embeddings from saved else generate new.
        store_name = pdf.name[:-4]
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        if os.path.exists(f"{store_name}/index.pkl"):
            # TODO: Add s3 link
            vector_store = FAISS.load_local(f"{store_name}", embeddings, allow_dangerous_deserialization=True)
        else:
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            vector_store.save_local(f"{store_name}")

        
        # Get user query and look through vector database.
        query = st.text_input("Ask questions about your PDF file:")
      
        if query:
            snippets = vector_store.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=snippets, question=query)
                print(cb)
            st.write(response)

if __name__ == "__main__":
    main()