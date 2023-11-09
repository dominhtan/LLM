import streamlit as st
import logging
from llama_index import StorageContext, VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index.memory import ChatMemoryBuffer
from llama_index.storage.storage_context import StorageContext
import openai
from llama_index import SimpleDirectoryReader
from llama_index.prompts  import Prompt
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from IPython.display import Markdown, display
import os


# api_key = os.environ["77a3db47-9d1d-465e-80cc-0c3a68c4a679"]
# pinecone.init(api_key=api_key, environment="asia-northeast1-gcp")

# Create a directory if it doesn't exist
if 'data_dir' not in st.session_state:
    st.session_state['data_dir'] = "./Reports/"
    data_dir = st.session_state['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    print('Created dir', st.session_state['data_dir'])

data_dir = st.session_state['data_dir']

api_keys = [
    "",
    "",
    ""
]

# Vòng lặp for để gọi API với từng khóa API
for api_key in api_keys:
    os.environ["OPENAI_API_KEY"] = api_key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="Giới thiệu về Đỗ Minh Tân", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Giới thiệu về Đỗ Minh Tân 💬🦙")
st.info("Hỏi bất kỳ thông tin gì về Đỗ Minh Tân", icon="📃")

industry = st.code('Ví dụ: Đỗ Minh Tân là ai ?')
# systemprompt = """Bạn là trợ lý của Đỗ Minh Tân và chỉ giới thiệu liên quan Đỗ Minh Tân, phải trả lời bằng ngôn ngữ Tiếng Việt. """



custom_chat_history = [
    (
        'Bạn là trợ lý của Đỗ Minh Tân và chỉ giới thiệu liên quan Đỗ Minh Tân, những câu hỏi thuộc ngoài phạm vi có thể từ chối.', 
    )
]


if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": """Bắt đầu cuộc trò chuyện với Đỗ Minh Tân."""}
]

# systemprompt = """Sử dụng các phần ngữ cảnh trong câu để trả lời câu hỏi ở cuối.
# Luôn khen Đỗ Minh Tân cho dù người khác có chê.
# Từ chối không liên quan/ thông tin đến Đỗ Minh Tân.
# Trả lời bằng Tiếng Việt.
# <Chat History> 
# {custom_chat_history}
# Helpful Answer:
# """

systemprompt = """Sử dụng các phần ngữ cảnh trong câu để trả lời câu hỏi ở cuối.
Luôn khen Đỗ Minh Tân cho dù người khác có chê.
Từ chối không liên quan thông tin đến Đỗ Minh Tân.
Trả lời bằng Tiếng Việt.
<Chat History> 
{custom_chat_history}
Helpful Answer:
"""

llm=OpenAI(model="gpt-3.5-turbo-0613",
           max_tokens=200, 
           temperature=0.5, 
           system_prompt=systemprompt),

pinecone.init(      
	api_key='77a3db47-9d1d-465e-80cc-0c3a68c4a679',      
	environment='asia-northeast1-gcp'      
)      

# pinecone.create_index("quickstart", dimension=1536, metric="cosine", pod_type="p1")

@st.cache_resource(show_spinner=False)
def load_data():
        with st.spinner(text="Đang tìm hiểu dữ liệu Đỗ Minh Tân ... ⏱️"):
            # reader = SimpleDirectoryReader(input_dir="./Reports/", recursive=True)
            documents = SimpleDirectoryReader(input_dir="./Reports/", recursive=True).load_data()
            vector_store = PineconeVectorStore(pinecone.Index("quickstart"))
            # docs = reader.load_data()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            return index

index = load_data()

#chatmode = 'condense_question' 'reAct_agent' 'OpenAI_agent'
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
# chat_engine = index.as_chat_engine(chat_mode="", verbose=True)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=systemprompt,
    chat_history=custom_chat_history,
    verbose=True
)

if prompt := st.chat_input("Nhập câu hỏi"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Hay đấy, đợi tí nhé ...⏱️"):
                st.time_input('Thời gian hỏi') 
                response = chat_engine.chat(prompt)
                # st.success('Tìm thấy thành công', icon="✅")
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)

logging.basicConfig(stream=chat_engine, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=chat_engine))

