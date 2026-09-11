import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# FIX: Standard and stable imports for LangChain v0.2+
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page Configuration
st.set_page_config(page_title="CyberlawGPT", page_icon="⚖️", layout="wide")
st.title("⚖️ CyberlawGPT")
st.caption("AI-Powered Pakistan Cyber Law Assistant")

# Fetch API Key securely from secrets.toml or environment variables
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

RAW_PDF_TEXT = """
CYBER LAWS IN PAKISTAN
Justice (R) Khalil-ur-Rehman Khan
1. Need for Cyber Laws
1.1. E-commerce: tool and practices involving Internet technologies.
1.2. Reduces costs, higher margins, eliminates middlemen.
1.3. Legal issues: location of transaction, electronic contracts, electronic signatures.
2. E-Commerce in Shariah Perspective
2.1. Permissible unless prohibited by Shariah.
2.2. Contract requires offer, acceptance, valid consideration. Free from 'gharar' (uncertainty) and 'riba' (interest).
3. Laws for Electronic Transactions & UNCITRAL
3.5. UNCITRAL Model Law on Electronic Commerce (1996) and Signatures (2001).
4. International Consensus Principles
5. Electronic Transactions Ordinance 2002 (ETO)
5.2. Promulgated Sept 11, 2002. Gives legal recognition to electronic documents, records, and digital signatures.
5.3. Accreditation Council, admissibility in court, exemption from stamp duty for 2 years.
7. Electronic Crimes & Bills
7.2. Addresses hacking, denial of service (DDoS), fraud, identity theft, unauthorized access.
8. Electronic Crimes Bill
8.1.4. Chapters on offences from access to "waging cyber war". Link between imprisonment and fine (1 yr = 100,000 PKR).
10. Electronic Data Protection & Constitution
10.1. Article 14 guarantees dignity of man and privacy of home.
12. Electronic Banking & Payment Systems
13. Payment Systems and Electronic Fund Transfers Act 2005.
"""

@st.cache_resource(show_spinner="Initializing Vector Database...")
def initialize_vector_store():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(RAW_PDF_TEXT.encode("utf-8"))
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings)

vectorstore = initialize_vector_store()

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ App Configuration")
    
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password")
    else:
        st.success("Groq API Key loaded via secrets.toml!")

    technical_level = st.select_slider(
        "Technical / Legal Level",
        options=["Simple / Layman", "Intermediate", "Legal Expert / Detailed"],
        value="Intermediate"
    )
    
    response_size = st.select_slider(
        "Response Size",
        options=["Brief", "Standard", "Comprehensive"],
        value="Standard"
    )

    model_name = st.selectbox(
        "Select Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    )

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input & RAG Chain
if prompt := st.chat_input("Ask a question about Pakistan Cyber Laws..."):
    if not groq_api_key:
        st.error("Please provide a valid Groq API Key via secrets.toml or the sidebar input field.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=0.2
            )
            
            system_prompt = f"""
            You are CyberlawGPT, an expert AI assistant on Cyber Laws in Pakistan.
            Answer the user's question accurately using the provided context.
            
            Guidelines:
            1. Target Audience Level: {technical_level}.
            2. Detail Level: {response_size}.
            3. Legal Disclaimer: Always include a brief note stating this is for informational purposes only.

            Context:
            {{context}}
            """

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
