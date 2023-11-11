from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
chat = ChatOpenAI(openai_api_key="###")
template = "###"
system_prompt = SystemMessagePromptTemplate.from_template(template)
human = "### {var} ### {var2}"
human_prompt = HumanMessagePromptTemplate.from_template(human)

chat_prompt = ChatPromptTemplate.from_messages([system_prompt,human_prompt])
chat_prompt.format_messages(var="###", var2="###")
conversation = LLMChain(
    llm=chat,
    prompt=chat_prompt,
    verbose=True,
)
print(conversation(chat_prompt))
