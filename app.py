import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_classic.callbacks import StreamlitCallbackHandler


import os 
from dotenv import load_dotenv
load_dotenv()

api_key="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"

#ARXIV AND WIKIPEDIA TOOLS
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

search=DuckDuckGoSearchRun(name='Search')

st.title('CHAT WITH SEARCH')
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

#SIDEBAR SETTINGS
st.sidebar.title("Settings")

if "messages" not in st.session_state:
    st.session_state['messages']=[{"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is Machine Learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="openai/gpt-oss-20b",streaming=True)
    tools=[arxiv,wiki,search]

    search_agent=create_agent(model=llm,tools=tools,system_prompt="You are a helpful assistant. Use your tools to look up information when needed.")
    latest_user_message = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.invoke({"input": latest_user_message},callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response["content"])
