import os
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Stable LCEL Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page Setup
st.set_page_config(page_title="CyberlawGPT", page_icon="⚖️", layout="wide")
st.title("⚖️ CyberlawGPT")
st.caption("AI-Powered Pakistan Cyber Law Assistant")

# Retrieve API key from secrets or environment
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

RAW_TEXT = """
CYBER LAWS IN PAKISTAN
Justice (R) Khalil-ur-Rehman Khan

1. Need for Cyber Laws
1.1. E-commerce: conducting business online, buying/selling products using digital cash or cards over networks.
1.2. E-commerce promises reduced costs, higher margins, efficient operations, and higher profits without physical establishments.
1.3. Legal issues include determining physical location of transactions, electronic contracts, and electronic signatures.

2. E-Commerce in Shariah Perspective
2.1. Contract formation over the Internet is legal unless prohibited by Shariah. Requires offer, acceptance, and valid consideration.
2.2. Must be free from gharar (uncertainty) and riba (interest). Credit cards charging interest make transactions void under Islamic principles.

3. Laws for Electronic Transactions & UNCITRAL
3.1. UNCITRAL Model Law on Electronic Commerce (1996) and Electronic Signatures (2001) establish functional equivalence to paper-based concepts.

4. International Consensus Principles
4.1. ILPF principles: remove legal barriers, respect contract freedom, harmonize laws, avoid non-tariff barriers, promote market standards.

5. Electronic Transactions Ordinance 2002 (ETO)
5.1. Promulgated September 11, 2002 to transition Pakistan from paper-based to electronic transactions.
5.2. Creates Accreditation Council, grants legal recognition to electronic documents/signatures, and exempts stamp duty for 2 years.

6. Priority Areas & Electronic Crimes
6.1. Priority areas for Pakistan: Data Protection, Electronic Crimes, and Electronic Banking.
6.2. Computers play roles as targets (hacking/DDoS), medium (identity theft/fraud), or incidental evidence storage.

7. Electronic Crimes Bill
7.1. Addresses intangible objects like computer programs and data.
7.2. Penalty link: For every 1 year imprisonment, fine is 100,000 PKR (e.g., 3 years = 300,000 PKR fine).

8. Electronic Data Protection & Constitution
8.1. Article 14 of the Constitution guarantees dignity of man and privacy of home.
8.2. Draft Data Protection Law satisfies EU Directive 95/46 Article 25 to facilitate foreign outsourcing.

9. Electronic Banking & Payment Systems
9.1. Payment Systems and Electronic Fund Transfers Act 2005 covers payment systems, real-time gross settlement, and electronic fund transfers.
"""

@st.cache_resource(show_spinner="Initializing Vector Database...")
def initialize_vector_store():
    documents = [Document(page_content=RAW_TEXT, metadata={"source": "Cyber_laws_in_Pakistan.pdf"})]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings)

vectorstore = initialize_vector_store()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ App Configuration")
    
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password")
    else:
        st.success("Groq API Key loaded securely!")

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

    # Active and active supported Groq models list
    model_name = st.selectbox(
        "Select Model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-3.2-11b-vision-preview",
            "gemma2-9b-it"
        ]
    )

# Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# User Query Handling
if prompt := st.chat_input("Ask a question about Pakistan Cyber Laws..."):
    if not groq_api_key:
        st.error("Please configure GROQ_API_KEY in Streamlit Secrets or enter it in the sidebar.")
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
            Answer the question based strictly on the retrieved legal context.

            Formatting Rules:
            1. Target Audience Level: {technical_level}.
            2. Response Detail Level: {response_size}.
            3. Disclaimer: Include a brief note that this information does not constitute formal legal counsel.

            Context:
            {{context}}
            
            Question: {{question}}
            """

            prompt_template = ChatPromptTemplate.from_template(system_prompt)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(prompt)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error: {str(e)}")
