from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import pandas as pd
load_dotenv(find_dotenv())

csv = "titanic.csv"
pd_agent = create_csv_agent(
    OpenAI(temperature=0),
    csv,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

with st.spinner('Thinking...'):
    placeholder = st.empty()
    with placeholder:
        result = pd_agent.run(
            f"""
            Create me an entire functioning streamlit dashboard to analyze this data.
            Plot trends over time, use st.metrics to display revelant big numbers.
            Use bar charts to show breakdowns by different categories.
            
            Return the python script I can use to create the entire dashboard.
            The filepath to the csv you're analyzing is is {csv}
            """
        )
    placeholder.empty()

exec(result)
with st.expander('Code Written by Agent'):
    st.markdown(
f"""
```py
{result}
```
"""
)
# result = pd_agent.run("how many rows are in this dataset?")