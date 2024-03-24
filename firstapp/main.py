import os
from constant import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

st.title('Personal Teaching Assistant:')
input_text = st.text_input("Ask the question you want to ask...")

# memory
topic_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
history_memory = ConversationBufferMemory(input_key='history', memory_key='chat_history')

# prompt templates
first_imput_prompt = PromptTemplate(
    input_variables=['topic'],
    template="What is {topic}"
)



# OpenAI LLM
llm = OpenAI(temperature=0.8) # temperature for howmuch control you want to give to the agent
chain = LLMChain(llm=llm, prompt=first_imput_prompt, verbose=True, output_key='q_topic', memory=topic_memory)



second_imput_prompt = PromptTemplate(
    input_variables=['q_topic'],
    template="history of {q_topic}"
)

chain2 = LLMChain(llm=llm, prompt=second_imput_prompt, verbose=True, output_key='history', memory=history_memory)


parent_chain = SequentialChain(chains=[chain, chain2], input_variables=['topic'],
                               output_variable=['q_topic', 'history'], verbose=True)




if input_text:
    # st.write(llm(input_text))
    # st.write(chain.run(input_text))
    # st.write(parent_chain.run(input_text))
    st.write(parent_chain({"topic": input_text}))
    
    with st.expander('Topic Name:'):
        st.info(topic_memory.buffer)
        
    with st.expander('Major Event:'):
        st.info(history_memory.buffer)