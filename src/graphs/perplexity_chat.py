# https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/#build-the-agent
import os
import sys
from uuid import uuid4
import asyncio

from dotenv import load_dotenv
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# for testing purposes:
p = "/".join(sys.path[0].split('/')[:-2])
if p not in sys.path:
    sys.path.append(p)

from src.registry import tool_graph
from utils import PerplexityChatModel

# loading variables from .env file
load_dotenv()


@tool_graph(name='get_openai_gpt', tag="deepsearch", 
            att_modals=['text'])
def get_perplexity_service(agent_graph: str, model_name: str, **kwargs):
    
    memory = MemorySaver()
    kwargs.pop("port", None)
    sampling_data = kwargs.pop("sampling", {})

    model = PerplexityChatModel(
        model_name=model_name,
        api_key=os.environ["PERPLEXITY_API_KEY"],
        **kwargs)
    
    # Define the function that calls the model
    async def call_model(state: MessagesState):
        response = await model.ainvoke(state["messages"], **sampling_data)
        return {"messages": response}

    # Define a new graph
    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", call_model)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    return workflow.compile(checkpointer=memory), model


if __name__ == "__main__":
    
    import asyncio
    import logging
    import tracemalloc
    from src.registry import GraphManager
    from src.graphs.utils import run_graph_in_terminal


    tracemalloc.start()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    GM = GraphManager()

    try:
        session_id = str(uuid4())
        asyncio.run(
            run_graph_in_terminal(
                graph_manager=GM,
                config="gpt-4o-mini-default",
                session_id=session_id))
    finally:
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logger.info("\nMemory traceback:")
        for stat in top_stats[:3]:
            logger.info(stat)