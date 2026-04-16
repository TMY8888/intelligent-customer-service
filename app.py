import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
# 关键：从 langchain_classic.chains 导入
from langchain_classic.chains import RetrievalQA
from langchain_community.chat_models import ChatZhipuAI
from dotenv import load_dotenv
# ... (代码的其他部分保持不变)
# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量读取智谱API Key
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    st.error("请在项目根目录创建 .env 文件，并写入 ZHIPUAI_API_KEY=你的密钥")
    st.stop()

st.set_page_config(page_title="智能客服知识库")
st.title("📚 智能客服知识库问答系统")

# 侧边栏：上传知识库文件
uploaded_file = st.sidebar.file_uploader("上传知识库文件（TXT格式）", type=["txt"])
if uploaded_file is not None:
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # 加载文档
    loader = TextLoader(tmp_path, encoding="utf-8")
    documents = loader.load()

    # 文本分割（分块）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # 向量化并建立FAISS索引
    embeddings = ZhipuAIEmbeddings(zhipuai_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 创建问答链
    llm = ChatZhipuAI(model="glm-4-flash", zhipuai_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    st.sidebar.success("知识库已加载！")

    # 用户提问
    question = st.text_input("请输入您的问题：")
    if question:
        with st.spinner("思考中..."):
            answer = qa_chain.run(question)
        st.write("**回答：**", answer)
else:
    st.sidebar.info("请上传一个TXT格式的知识库文件")