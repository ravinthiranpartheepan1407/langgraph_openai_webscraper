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

load_dotenv()

st.title("Ravinthiran: Amazon Webscraper + Langgraph")
st.text("Github: https://github.com/ravinthiranpartheepan1407/langgraph_openai_webscraper")
openai_model = st.sidebar.selectbox("Select GPT model", ["gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-4-0125-preview"])
openai_key = st.sidebar.text_input("Your OpenAPI Key", type="password")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

user_input = st.text_input("Enter your message here: ")
#
if st.button("Run Workflow"):
    with st.spinner("Running Workflow..."):
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
            loader = WebBaseLoader(urls)
            docs = loader.load()
            return "\n\n".join(
                [f'<Document Name="{doc.metadata.get("title", "")}">\n{doc.get_content}\n</Document>' for doc in docs])


        @tool("Market Research")
        def research(content: str) -> str:
            """Amazon Researcher"""
            chat = ChatOpenAI(model=openai_model)
            messages = [
                SystemMessage(content="Perform Amazon market analysis"),
                HumanMessage(content=content),
            ]
            res = chat(messages)
            return res.content


        @tool("DropShipping")
        def drop_ship(content: str) -> str:
            """Drop shipper"""
            chat = ChatOpenAI(model=openai_model)
            messages = [
                SystemMessage(content="Perform DropShipping"),
                HumanMessage(content=content),
            ]
            res = chat(messages)
            return res.content

        llm = ChatOpenAI(model=openai_model)

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

        agents = [SCRAPER, RESEARCHER, DROPSHIP, SUPERVISOR]

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
        workflow.add_conditional_edges(SUPERVISOR, lambda x: x["next"], {
            SCRAPER: SCRAPER,
            RESEARCHER: RESEARCHER,
            DROPSHIP: DROPSHIP,
            "FINISH": END
        })

        workflow.set_entry_point(SUPERVISOR)
        graph = workflow.compile()

        for sets in graph.stream({"messages": [HumanMessage(content=user_input)]}):
            if "__end__" not in sets:
                st.write(sets)
                st.write("---")
