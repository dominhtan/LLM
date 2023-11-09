from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory

llm = ChatOpenAI(openai_api_key="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")
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
# print(memory.chat_memory.add_user_message("hi!"))
# print(memory.chat_memory.add_ai_message("whats up?"))
# memory = ConversationBufferWindowMemory(k=1)
# memory.save_context({"input": "Hello"}, {"output": "###"})
# memory.save_context({"input": "Sorry"}, {"output": "###"})
print(conversation({"question": "Translate this sentence from English to Việt Nam: I love programming"}))
print(memory.load_memory_variables({}))
