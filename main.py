try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings   
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()


#제목
st.title("ChatPDF with rag")
st.write("-------------------------")

# 파일업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type=['pdf'])
st.write("------------")


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath= os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath,"wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages=loader.load_and_split()
    return pages

#업로드된 파일처리

if uploaded_file is not None:
    st.success(f"✅ {uploaded_file.name} 업로드 완료")
    pages=pdf_to_document(uploaded_file)
    st.write(f"📑 총 {len(pages)} 페이지 로드됨")



    #loader = PyPDFLoader("unsu.pdf")
    #pages = loader.load_and_split()

    #Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=300,
        chunk_overlap = 20,
        length_function = len,
        is_separator_regex = False,
    
    )
    if pages:
        texts = text_splitter.split_documents(pages)



        #embeddings
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small",
            # With the 'text-embedding-3' class
            # if models, you can specify the size
            # if the embeddings you wnat returned
            # dimensions=1024
        )

        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        #chroma DB
        db= Chroma.from_documents(texts, embeddings_model)
    else:
        st.error("❌ PDF에서 페이지를 불러오지 못했습니다.")
else:
    st.warning("📂 먼저 PDF 파일을 업로드하세요")
#User Input
st.header("PDF에게 질문해보세요!!")
question=st.text_input("질문을 입력하세요.")

if st.button("질문하기"):
    with st.spinner("답변을 생성하는 중입니다..."):
        #Retriver
        # question = "아내가 먹고싶어하는 음식이 뭐야?"
        llm = ChatOpenAI(temperature=0)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(), llm=llm
        )
    

        # prompt Template
        prompt = hub.pull("rlm/rag-prompt")

        # docs = retriever_from_llm.invoke(question)
        # print(len(docs))
        # print(docs)

        # Generate
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        rag_chain = (
            {"context": retriever_from_llm | format_docs, "question" : RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        #Question
        result = rag_chain.invoke(question)

        st.write(result)


