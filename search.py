import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import pandas as pd

if "startups" not in st.session_state:
    st.session_state["startups"] = []
if "status" not in st.session_state:
    st.session_state["status"] = []
if "cover_letter" not in st.session_state:
    st.session_state["cover_letter"] = []
if "step" not in st.session_state:
    st.session_state["step"] = 0

def store_startups(response):
    st.session_state.startups += response.split(",")
    for _ in st.session_state.startups:
        st.session_state.status.append("no opinion yet")

def store_status(response):
    st.session_state.status = response.split(",")

def store_cover_letters(response):
    st.session_state.cover_letter = response.split(",")


assistant_questions = ["Hi, I'm your personal start-up liaison. Please tell me what kind of start-up you want ?", 
                       "Can you describe yourself so that I can see if the startup would be of interest ?", 
                       "Are there projects, insights, accomplishments you would like to share in your cover letter ?",
                       "That's all for now."]
suggestions = ["List 3 healthcare start-ups near Paris.", 
                "With a solid academic grounding from Télécom Paris and IMT Atlantique, I've honed my AI and Computer Vision expertise, notably in medical applications. My journey spans from developing scoring models to spearheading advanced machine learning projects. Armed with tools like Pytorch and a fluency in English and French, I'm keen on driving AI advancements in a dynamic start-up environment within the medical sector.",
                "I build an app to unmask people wearing covid mask on pictures. I developped a model to assert the health status of several organs based on CT scans."]
postprompts = ["Respond only with a list of start-up names separated by a comma if you do not your answer is incorrect.", 
               "Now decide if each startup in the list is adequate to my profile, return a comma-separated list with values either 'OK' or 'KO' for each startup. If you do not your answer is incorrect.",
               "Write a list of cover letter for each of the startups in less than 200 words. Cover letters must be separated by commas."]
response_process = [store_startups, store_status, store_cover_letters]


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🔎 Cherry-pick your start-ups (LangChain - Chat with search)")

"""
ChatGPT is going to ask you questions to find a set of start-ups in adequation with your profile.

In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": assistant_questions[st.session_state.step]}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Type your message here.", key="chat_input"):
    postprompt = postprompts[st.session_state.step]
    print(postprompt)
    print(st.session_state.step)
    st.session_state.messages.append({"role": "user", "content": prompt+postprompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
        # Process ChatGPT answer
        response_process[st.session_state.step](response)
        st.session_state.step += 1
        next_question = assistant_questions[st.session_state.step]
        st.session_state.messages.append({"role": "assistant", "content": next_question})
        st.write(next_question)

st.write(pd.DataFrame({
    'Startup': st.session_state.startups,
    'status': st.session_state.status
}))