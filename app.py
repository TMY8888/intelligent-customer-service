import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_community.chat_models import ChatZhipuAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")

st.set_page_config(page_title="智能客服知识库")
st.title("📚 智能客服知识库问答系统")


# 意图识别函数
def detect_intent(question: str) -> str:
    question_lower = question.lower()
    if "订单" in question_lower or "物流" in question_lower or "快递" in question_lower:
        return "order"
    else:
        return "policy"


def mock_order_status(order_id: str = "12345") -> str:
    return f"订单 {order_id} 状态：已发货，预计明天送达。"


uploaded_file = st.sidebar.file_uploader("上传知识库文件（TXT/PDF）", type=["txt", "pdf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    if uploaded_file.name.endswith(".txt"):
        loader = TextLoader(tmp_path, encoding="utf-8")
    else:
        loader = PyPDFLoader(tmp_path)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = ZhipuAIEmbeddings(zhipuai_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatZhipuAI(model="glm-4-flash", zhipuai_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    st.sidebar.success("知识库已加载！")

    question = st.text_input("请输入您的问题：")
    if question:
        with st.spinner("思考中..."):
            intent = detect_intent(question)
            if intent == "order":
                answer = mock_order_status()
                st.write("**回答：**", answer)
                st.info("注：此为模拟订单查询，实际应用需对接订单系统API。")
            else:
                answer = qa_chain.run(question)
                st.write("**回答：**", answer)
else:
    st.sidebar.info("请上传TXT或PDF文件")