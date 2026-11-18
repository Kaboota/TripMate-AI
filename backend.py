import os
import uuid
import operator
import json
import certifi

from typing import TypedDict, Annotated

from dotenv import load_dotenv

# =========================
# Environment Setup
# =========================

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# =========================
# Imports
# =========================

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


# =========================
# Environment Variables
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_database_url():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url += f"{separator}sslmode=require"

    return database_url


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Please add it to your .env file."
    )


# =========================
# LLM Configuration
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)


# =========================
# Helper Functions
# =========================

def convert_to_text(data):

    if data is None:
        return "No information available."

    if isinstance(data, str):
        return data

    try:
        return json.dumps(data, indent=2, default=str)

    except (TypeError, ValueError):
        return str(data)


def safe_tool_call(tool, query, tool_name):

    try:
        result = tool(query)

        if result is None or result == "":
            return f"No results found from {tool_name}."

        return convert_to_text(result)

    except Exception as e:

        print(f"\n{tool_name} ERROR: {e}")

        return (
            f"{tool_name} could not retrieve information. "
            f"Error: {str(e)}"
        )


# =========================
# LangGraph State
# =========================

class TravelState(TypedDict):

    messages: Annotated[list[AnyMessage], operator.add]

    user_query: str

    flight_results: str

    hotel_results: str

    itinerary: str

    llm_calls: int


# =========================
# Flight Agent
# =========================

def flight_agent(state: TravelState):

    print("\n[Flight Agent] Searching for flights...")

    query = state["user_query"]

    flight_data = safe_tool_call(
        search_flights,
        query,
        "Flight API"
    )

    return {
        "flight_results": flight_data,

        "messages": [
            AIMessage(
                content="Flight search completed."
            )
        ]
    }


# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):

    print("\n[Hotel Agent] Searching for hotels...")

    query = f"Best hotels for {state['user_query']}"

    hotel_data = safe_tool_call(
        tavily_search,
        query,
        "Tavily Search"
    )

    return {
        "hotel_results": hotel_data,

        "messages": [
            AIMessage(
                content="Hotel search completed."
            )
        ]
    }


# =========================
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):

    print("\n[Itinerary Agent] Creating travel itinerary...")

    prompt = f"""
You are an expert travel planner.

Create a complete and practical travel itinerary.

USER REQUEST:
{state['user_query']}

FLIGHT INFORMATION:
{state['flight_results']}

HOTEL INFORMATION:
{state['hotel_results']}

INSTRUCTIONS:

1. Create a day-by-day itinerary.
2. Consider the user's travel budget.
3. Include popular tourist attractions.
4. Suggest practical transportation options.
5. Include approximate meal and activity costs.
6. Recommend suitable hotels.
7. Avoid inventing flight availability or hotel bookings.
8. Clearly identify estimated costs.
9. Mention missing information where necessary.

Make the itinerary easy to understand.
"""

    response = llm.invoke([
        SystemMessage(
            content="You are an expert AI travel planner."
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,

        "messages": [response],

        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):

    print("\n[Final Agent] Generating final travel response...")

    final_prompt = f"""
You are a professional AI travel booking assistant.

Generate the final travel response using the information below.

USER REQUEST:
{state['user_query']}

FLIGHT RESULTS:
{state['flight_results']}

HOTEL RESULTS:
{state['hotel_results']}

GENERATED ITINERARY:
{state['itinerary']}

Format the final response using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

IMPORTANT INSTRUCTIONS:

- Be clear and practical.
- Respect the user's requested budget.
- Show estimated expenses in INR when appropriate.
- Mention when flight ticket prices are unavailable.
- Do not claim that flights or hotels are booked.
- Do not invent confirmed availability or prices.
- Clearly distinguish estimated prices from verified results.
- If information is unavailable, say so honestly.
- Avoid repeating unnecessary information.

Generate a polished and useful travel plan.
"""

    response = llm.invoke([
        SystemMessage(
            content="You are a professional AI travel booking assistant."
        ),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],

        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build LangGraph
# =========================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)

graph.add_node("hotel_agent", hotel_agent)

graph.add_node("itinerary_agent", itinerary_agent)

graph.add_node("final_agent", final_agent)


# =========================
# Graph Connections
# =========================

graph.add_edge(START, "flight_agent")

graph.add_edge("flight_agent", "hotel_agent")

graph.add_edge("hotel_agent", "itinerary_agent")

graph.add_edge("itinerary_agent", "final_agent")

graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
    connect_timeout=15
)

checkpointer = PostgresSaver(_conn)

# Create checkpoint tables if they do not exist
checkpointer.setup()

# Compile LangGraph
travel_graph = graph.compile(
    checkpointer=checkpointer
)


# =========================
# Main Travel Agent Function
# =========================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None
):

    if not user_input or not user_input.strip():
        raise ValueError("Travel request cannot be empty.")

    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    print("\n================================")
    print("       TRIPMATE AI STARTED")
    print("================================")

    try:

        result = travel_graph.invoke(

            {
                "messages": [
                    HumanMessage(content=user_input)
                ],

                "user_query": user_input,

                "flight_results": "",

                "hotel_results": "",

                "itinerary": "",

                "llm_calls": 0
            },

            config=config
        )

        final_answer = result["messages"][-1].content

        print("\n================================")
        print("       TRIPMATE AI COMPLETED")
        print("================================")

        return {

            "thread_id": thread_id,

            "answer": final_answer,

            "flight_results": result.get(
                "flight_results", ""
            ),

            "hotel_results": result.get(
                "hotel_results", ""
            ),

            "itinerary": result.get(
                "itinerary", ""
            ),

            "llm_calls": result.get(
                "llm_calls", 0
            )
        }

    except Exception as e:

        print("\nTRAVEL AGENT ERROR:")
        print(str(e))

        raise
