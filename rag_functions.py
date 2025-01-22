#RAG기반 지능형 Q&A 웹서비스
#pip install wikipedia
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain.chains import RetrievalQA

OPENAI_API_KEY = "YOUR_API_KEY"

import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


#데이터 로드 & 분할 결과 반환  모듈화 
def load_docs(query) :
    loader = WikipediaLoader(query=query, load_max_docs=1)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    return splits

# 로드하고 분할된 데이터 임베딩 & Vector DB에 저장 모듈화
def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits,
                                        embedding=embeddings)
    return vectorstore

#RAG Chain 생성
def create_rag_chain(vectorstore):
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    prompt_template ="""아래의 문맥을 사용하여 질문에 답하십시오.
    만약 답을 모른다면, 모른다고 말하고 답을 지어내지 마십시오.
    최대한 세 문장으로 답하고 가능한 한 간결하게 유지하십시오.
    {context}
    질문: {question}
    유용한 답변 :"""
    PROMPT = ChatPromptTemplate.from_template(template=prompt_template, 
                                              input_variable=["context", "question"])

    chain_type_kwargs = {"prompt" : PROMPT }
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever =vectorstore.as_retriever(),
                                           chain_type_kwargs = chain_type_kwargs,
                                           return_source_documents=True
                                          )
    return qa_chain   
