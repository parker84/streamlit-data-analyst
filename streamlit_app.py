from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_icon='💻')
st.title('Streamlit Data Analyst 💻')
st.markdown('**Purpose**: To enable users to understand their data')


with st.form('form'):
    df = None
    csv_file = st.file_uploader("Upload Your CSV", type={"csv"})
    st.caption("Or ... don't upload a csv and use the **Titanic Dataset**🚢 instead")

    if csv_file is not None:
        df = pd.read_csv(csv_file)

    if df is None:
        df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

    df.to_csv('data.csv', index=False)
    csv = 'data.csv'
    question = st.text_area(
        label='Ask Questions',
        value="""1. Show me a sample of this dataset (include all the columns)
2. What is the survival rate by Gender?
3. What is the survival rate by age bucket?
4. What is the survival rate by Class?
5. What is the survival rate by Embarked location?
        """,
        height=150,
    )

    st.sidebar.subheader("Enter Your API Key 🗝️")
    open_api_key = st.sidebar.text_input(
        "Open API Key 🗝️", 
        value=st.session_state.get('open_api_key', ''),
        help="Get your API key from https://openai.com/",
        type='password'
    )
    os.environ["OPENAI_API_KEY"] = open_api_key
    st.session_state['open_api_key'] = open_api_key
    load_dotenv(find_dotenv())

    submitted = st.form_submit_button("Analyze!")

if submitted:
    if open_api_key == '' or open_api_key is None:
        st.error("⚠️ Please enter your API key in the sidebar")
    else:
        pd_agent = create_csv_agent(
            # OpenAI(temperature=0),
            ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k'),
            # ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
            # ChatOpenAI(temperature=0, model='gpt-4'),
            csv,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        @st.cache_data
        def ask_the_csv_agent(prompt):
            result = pd_agent.run(prompt)
            return result

        prompt = f"""
        Write me a python script for a streamlit dashboard to analyze this data.
        Only return this python script, nothing else - no commentary / formatting.
        Assume your whole output is going to be inserted right into a python script and run.

        Answer these specific questions with your dashboard:
        {question}
        Display results with nice charts (using streamlit or plotly.express)
        
        The filepath to the csv you're analyzing is ./{csv}

        Only return this python script, nothing else - no commentary / formatting.
        Assume your whole output is going to be inserted right into a python script and run.
        """

        placeholder = st.empty() # we do this so 
        with placeholder:
            result = (
                ask_the_csv_agent(prompt)
                .replace('The Python script for the Streamlit dashboard is as follows:', '')
                .replace('```', '\n').replace('python', '')
            )
            result_lines = result.split('\n')
            if 'python' in result_lines[0].lower():
                result = '\n'.join(result_lines[1:])
        placeholder.empty()

        exec(result)

        st.markdown('### The Code')
        with st.expander('Code Written by Agent'):
            st.markdown(
        f"""
        ```py
        {result}
        ```
        """
        )