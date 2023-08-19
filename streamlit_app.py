from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from dotenv import find_dotenv, load_dotenv
import streamlit as st
load_dotenv(find_dotenv())

st.set_page_config(page_icon='💻')
st.title('Streamlit Data Analyst 💻')



csv = "titanic.csv"
pd_agent = create_csv_agent(
    OpenAI(temperature=0),
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
        Create me an entire functioning streamlit dashboard to analyze this data.
        Plot trends over time, use st.metrics to display revelant big numbers.
        Use bar charts to show breakdowns by different categories.
        
        Return the python script I can use to create the entire dashboard.
        The filepath to the csv you're analyzing is is {csv}
        """
    )
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