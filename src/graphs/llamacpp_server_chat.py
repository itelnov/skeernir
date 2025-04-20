import os
import sys
import logging
from uuid import uuid4
from typing import Union, Tuple, Any, Dict

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    AIMessageChunk, HumanMessage)

# for testing purposes:
p = "/".join(sys.path[0].split('/')[:-2])
if p not in sys.path:
    sys.path.append(p)

from src.registry import tool_graph, terminate_processes
from src.graphs.utils import LlamaCppClient, run_server, check_server_healthy


@tool_graph(name='model_on_llamacpp_server', tag="chat")
def get_model_on_llamacpp_server(
    host_name: str,
    model_path: str,
    **kwargs) -> Union[CompiledStateGraph, Tuple[CompiledStateGraph, Any]]:
       
    memory = MemorySaver()
    model_name = os.path.basename(model_path)
    port = kwargs.pop("port", 8085)
    model_name = os.path.basename(model_path)
    host_params = kwargs.pop("host_params", {})
    
    graph_server_process = None
    try:
        graph_server_process = run_server(
            host_name,
            model_path,
            port=port, 
            **host_params)
    except Exception as e:
        logging.error(f"Graph failed to start:\n{e}")
        raise 
    
    model = LlamaCppClient(
        base_url=f'http://127.0.0.1:{port}')

    if not check_server_healthy(port, model_name, time_to_wait=3):
        try:
            terminate_processes([graph_server_process])
        except Exception as e:
            logging.error(f"Graph failed to start:\n\n{e}")
            raise 

    sampling_data = kwargs.get("sampling")
    # Define the function that calls the model
    async def call_model(state: MessagesState):
        response = await model.ainvoke(state["messages"], **sampling_data)
        # We return a list, because this will get added to the existing list
        return {"messages": response}
    
    # Define a new graph
    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", call_model)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    app = workflow.compile(checkpointer=memory)
    
    return app, graph_server_process


if __name__ == "__main__":

    import asyncio
    import os
    import tracemalloc
    import logging
    from src.registry import GraphManager
    from src.graphs.utils import run_graph_in_terminal

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    tracemalloc.start()

    GM = GraphManager()

    session_id = str(uuid4())
    try:
        session_id = str(uuid4())
        asyncio.run(
            run_graph_in_terminal(
                graph_manager=GM,
                config="phi3_5-mini-instruct",
                session_id=session_id))
    finally:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logger.info("\nMemory traceback:")
        for stat in top_stats[:3]:
            logger.info(stat)
