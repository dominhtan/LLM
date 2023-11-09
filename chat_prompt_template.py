import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# chat = ChatOpenAI(openai_api_key="###",model_name="gpt-3.5-turbo-0613",temperature=1.0)
chat = ChatOpenAI(openai_api_key="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",model_name="gpt-3.5-turbo-0613",temperature=0.5)

template_string ="""####"""

prompt_human = HumanMessagePromptTemplate.from_template(template_string)
product = "###"
prompt_system = SystemMessagePromptTemplate.from_template(product)

prompt = ChatPromptTemplate.from_messages([prompt_human,prompt_system])

conversation = LLMChain(
    llm=chat,
    prompt=prompt,
    verbose=True,
)
start_time = time.time()
# result = conversation({"query": product})
with get_openai_callback() as cb:
    response = conversation.run({"query": template_string})
    end_time = time.time()
    # chat_history = response["choices"][0]["chat_history"]
    print(response)
    print(f"Thời gian phản hồi: {end_time - start_time:.2f} seconds")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

