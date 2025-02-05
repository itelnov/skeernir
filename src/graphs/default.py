# https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/#build-the-agent
import sys
from uuid import uuid4
import asyncio
from typing import Any, Dict, Iterator, List, Optional

from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.outputs import ChatGenerationChunk
from langchain.chat_models.base import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field

# for testing purposes:
p = "/".join(sys.path[0].split('/')[:-2])
if p not in sys.path:
    sys.path.append(p)

from src.registry import tool_graph
from utils import PlaceholderModel



@tool_graph(name='Ghost', tag="welcom")
def get_ghost(**kwargs):
    
    memory = MemorySaver()
    kwargs.pop("port", None)
    sampling_data = kwargs.pop("sampling", {})

    response_text=("Oh, welcome to Skeernir! Pick your Agent, sit back, and "
                    "have a blast... or don't. Or maybe you'll just love "
                    "hating every second of it. Either way, we promise "
                    "something.")

    model = PlaceholderModel(model_name="Ghost")
    sampling_data["response_text"] = response_text
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
                config="ghost",
                session_id=session_id))
    finally:
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logger.info("\nMemory traceback:")
        for stat in top_stats[:3]:
            logger.info(stat)