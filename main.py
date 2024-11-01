import operator
import os
from typing import Annotated, Sequence, TypedDict, List
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openapi_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import HumanMessage, SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools import tool
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import openai
import time

# Initialize session state
if 'api_key_verified' not in st.session_state:
    st.session_state.api_key_verified = False

load_dotenv()

st.title("Ravinthiran: Amazon Webscraper + Langgraph")
st.text("Github: https://github.com/ravinthiranpartheepan1407/langgraph_openai_webscraper")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-color: 'black';
        color: 'white';
    }
    [data-testid="stHeader"] {
        display: none;
    }
    [data-testid="stToolbar"] {
        display: none;
    }
    ._profileContainer_1yi6l_53 {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

openai_model = st.sidebar.selectbox(
    "Select GPT model",
    ["gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-4-0125-preview"]
)
openai_key = st.sidebar.text_input("Your OpenAPI Key", type="password")

# Verify API key when provided
if openai_key and not st.session_state.api_key_verified:
    try:
        # Test the API key with a minimal API call
        client = openai.OpenAI(api_key=openai_key)
        client.models.list()  # Simple API call to verify key
        os.environ["OPENAI_API_KEY"] = openai_key
        st.session_state.api_key_verified = True
        st.sidebar.success("✅ API key verified successfully!")
    except openai.RateLimitError:
        st.sidebar.error("❌ Rate limit exceeded. Please wait a few minutes and try again.")
    except openai.AuthenticationError:
        st.sidebar.error("❌ Invalid API key. Please check your key and try again.")
    except Exception as e:
        st.sidebar.error(f"❌ An error occurred: {str(e)}")

user_input = st.text_input("Enter your message here: ")

if st.button("Run Workflow"):
    if not st.session_state.api_key_verified:
        st.error("Please enter a valid OpenAI API key first.")
    elif not user_input:
        st.error("Please enter a message before running the workflow.")
    else:
        with st.spinner("Running Workflow..."):
            try:
                def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        MessagesPlaceholder(variable_name="messages"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    agent = create_openapi_agent(llm, tools, prompt)
                    return AgentExecutor(agent=agent, tools=tools)

                def create_supervisor(llm: ChatOpenAI, agents: List[str]):
                    system_prompt = "You are the supervisor over the following agents: {agents}."
                    options = ["FINISH"] + agents
                    func_def = {
                        "name": "supervisor",
                        "description": "Select next agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "next": {
                                    "anyOf": [
                                        {"enum": options},
                                    ],
                                }
                            },
                            "required": ["next"],
                        },
                    }

                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        MessagesPlaceholder(variable_name="messages"),
                        (
                            "system",
                            "Please select one of the following agents: {options}"
                        ),
                    ]).partial(options=str(options), agents=", ".join(agents))
                    return (
                        prompt
                        | llm.bind_functions(functions=[func_def], function_call="supervisor")
                        | JsonOutputFunctionsParser()
                    )

                @tool("Scrape web")
                def analyze(urls: list[str]) -> str:
                    """Web Scraper"""
                    try:
                        loader = WebBaseLoader(urls)
                        docs = loader.load()
                        return "\n\n".join(
                            [f'<Document Name="{doc.metadata.get("title", "")}">\n{doc.get_content}\n</Document>' for doc in docs])
                    except Exception as e:
                        return f"Error scraping web content: {str(e)}"

                @tool("Market Research")
                def research(content: str) -> str:
                    """Amazon Researcher"""
                    try:
                        chat = ChatOpenAI(model=openai_model)
                        messages = [
                            SystemMessage(content="Perform Amazon market analysis"),
                            HumanMessage(content=content),
                        ]
                        res = chat(messages)
                        return res.content
                    except openai.RateLimitError:
                        time.sleep(20)  # Wait and retry once
                        chat = ChatOpenAI(model=openai_model)
                        messages = [
                            SystemMessage(content="Perform Amazon market analysis"),
                            HumanMessage(content=content),
                        ]
                        return chat(messages).content

                @tool("DropShipping")
                def drop_ship(content: str) -> str:
                    """Drop shipper"""
                    try:
                        chat = ChatOpenAI(model=openai_model)
                        messages = [
                            SystemMessage(content="Perform DropShipping"),
                            HumanMessage(content=content),
                        ]
                        res = chat(messages)
                        return res.content
                    except openai.RateLimitError:
                        time.sleep(20)  # Wait and retry once
                        chat = ChatOpenAI(model=openai_model)
                        messages = [
                            SystemMessage(content="Perform DropShipping"),
                            HumanMessage(content=content),
                        ]
                        return chat(messages).content

                llm = ChatOpenAI(
                    model=openai_model,
                    temperature=0.7,
                    max_retries=2
                )

                def amazon_scraper() -> Runnable:
                    prompt = "Amazon Scraper"
                    return create_agent(llm, [analyze], prompt)

                def amazon_research() -> Runnable:
                    prompt = "Amazon Researcher"
                    return create_agent(llm, [research], prompt)

                def amazon_dropship() -> Runnable:
                    prompt = "Amazon Seller"
                    return create_agent(llm, [drop_ship], prompt)

                SCRAPER = "SCRAPER"
                RESEARCHER = "RESEARCHER"
                DROPSHIP = "DROPSHIP"
                SUPERVISOR = "SUPERVISOR"

                agents = [SCRAPER, RESEARCHER, DROPSHIP]

                class StateAgent(TypedDict):
                    messages: Annotated[Sequence[BaseMessage], operator.add]
                    next: str

                def scraper_node(state: StateAgent) -> dict:
                    res = amazon_scraper().invoke(state)
                    return {"messages": [HumanMessage(content=res["output"], name=SCRAPER)]}

                def researcher_node(state: StateAgent) -> dict:
                    res = amazon_research().invoke(state)
                    return {"messages": [HumanMessage(content=res["output"], name=RESEARCHER)]}

                def dropship_node(state: StateAgent) -> dict:
                    res = amazon_dropship().invoke(state)
                    return {"messages": [HumanMessage(content=res["output"], name=DROPSHIP)]}

                def supervisor_node(state: StateAgent) -> Runnable:
                    return create_supervisor(llm, agents)

                workflow = StateGraph(StateAgent)
                workflow.add_node(SCRAPER, scraper_node)
                workflow.add_node(RESEARCHER, researcher_node)
                workflow.add_node(DROPSHIP, dropship_node)
                workflow.add_node(SUPERVISOR, supervisor_node)

                workflow.add_edge(SCRAPER, SUPERVISOR)
                workflow.add_edge(RESEARCHER, SUPERVISOR)
                workflow.add_edge(DROPSHIP, SUPERVISOR)
                workflow.add_conditional_edges(
                    SUPERVISOR,
                    lambda x: x["next"],
                    {
                        SCRAPER: SCRAPER,
                        RESEARCHER: RESEARCHER,
                        DROPSHIP: DROPSHIP,
                        "FINISH": END
                    }
                )

                workflow.set_entry_point(SUPERVISOR)
                graph = workflow.compile()

                for sets in graph.stream({"messages": [HumanMessage(content=user_input)]}):
                    if "__end__" not in sets:
                        st.write(sets)
                        st.write("---")

            except openai.RateLimitError:
                st.error("Rate limit exceeded. Please wait a few minutes and try again.")
            except openai.AuthenticationError:
                st.error("Authentication error. Please check your API key.")
                st.session_state.api_key_verified = False  # Reset verification
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")