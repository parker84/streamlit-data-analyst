from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import pandas as pd
load_dotenv(find_dotenv())

st.set_page_config(page_icon='💻')
st.title('Streamlit Data Analyst 💻')
st.markdown('**Purpose**: To enable users to understand their data')

csv_file = st.file_uploader("Upload Your CSV", type={"csv"})
df = None

if csv_file is not None:
    df = pd.read_csv(csv_file)

if st.button("Or ... Use The Titanic's Data"):
    df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

if df is not None:
    df.to_csv('data.csv', index=False)
    csv = 'data.csv'
    pd_agent = create_csv_agent(
        # OpenAI(temperature=0),
        # ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k'),
        ChatOpenAI(temperature=0, model='gpt-4'),
        csv,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    @st.cache_data
    def ask_the_csv_agent(prompt):
        result = pd_agent.run(prompt)
        return result

    placeholder = st.empty()
    with placeholder:
        result = ask_the_csv_agent(
            f"""
            Write me a python script for a streamlit dashboard to analyze this data.
            Only return this python script, nothing else - no commentary / formatting.
            Assume your whole output is going to be inserted right into a python script and run. 

            I should know everything important about this dataset after looking at the dashboard.
            1. Pull important metrics out into st.metric (put these at the top of the dashboard) and round accordingly
            2. Plot any relevant trends over time (use px.line) - and label your axes / title your charts
            3. Use bar charts to show breakdowns by different categories (use px.bar) - and label your axes / title your charts
            4. use st.columns to nicely format your dashboard
            5. This should give me a complete overview of everything important I need to know about this dataset.

            The filepath to the csv you're analyzing is ./{csv}
            """
        ).replace('```', '\n').replace('python', '')
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

    st.markdown('### Ask a Specific Question')
    question = st.text_input(
        label='Ask a Specific Question About the Dataset',
        value='How many people are over the age of 65?'
    )
    result = ask_the_csv_agent(f"{question} (display response for markdown)")
    st.markdown(result)