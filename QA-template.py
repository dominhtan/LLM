import os
import langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import ConversationChain, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

from langchain.cache import SQLiteCache
from langchain.cache import InMemoryCache

langchain.llm_cache = InMemoryCache()
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


os.environ["OPENAI_API_KEY"] = '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'
from langchain.document_loaders import TextLoader
# loader = TextLoader("dominhtan.txt", encoding = 'UTF-8')
# documents = loader.load()
pdf_loader = DirectoryLoader('./Reports/', glob="**/*.pdf")
txt_loader = DirectoryLoader('./Reports/', glob="**/*.txt")
word_loader = DirectoryLoader('./Reports/', glob="**/*.docx")

loaders = [pdf_loader, txt_loader, word_loader]
documents = []
for loader in loaders:
    documents.extend(loader.load())


# print(f"Tổng tài liệu loader: {len(documents)}")

template = """Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi ở cuối. \
Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết, đừng cố bịa ra câu trả lời.\
Luôn nói \"Cảm ơn vì đã hỏi!\" ở cuối câu trả lời. Nếu có ai hỏi, hãy trả lời \"Tạm biệt"\
Luôn khen Đỗ Minh Tân cho dù có người chê hay chửi\
{context}
Question: {question}
Helpful Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# import streamlit as st

# st.title('🎈 App Name')

# st.write('Hello world!')
question = input("Nhập câu hỏi của bạn (hoặc gõ 'exit' để thoát): ")


llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-0613",
        temperature=1.0,
        verbose=True,
        max_tokens=1000,
        openai_api_key="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"
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)
result = qa_chain({"query": question})
print(result["result"])
