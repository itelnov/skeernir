# https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/#build-the-agent
import sys
import os
import logging
from uuid import uuid4
from langgraph.graph import MessagesState, StateGraph, START, END
from openai import OpenAI
from langgraph.checkpoint.memory import MemorySaver

# for testing purposes:
p = "/".join(sys.path[0].split('/')[:-2])
if p not in sys.path:
    sys.path.append(p)

from src.registry import tool_graph, terminate_processes
from src.graphs.utils import (
    OpenAICompatibleChatModel, run_server, check_server_healthy)


@tool_graph(name='llama3_2_3b_on_vllm_server', tag="chat", att_modals=['text'])
def get_llama3_2_b3_on_vllm_server(
    server: str,
    model_path: str,
    use_gpu: bool = False,
    **kwargs) -> StateGraph:
    
    model_name = os.path.basename(model_path)
    port = kwargs.pop("port", 8083)
    graph_model_process = None
    try:
        graph_server_process = run_server(
            server=server,
            model_path=model_path,
            port=port, 
            use_gpu=use_gpu, 
            **kwargs)
    except Exception as e:
        logging.error(f"Graph failed to start:\n{e}")
        raise 

    client = OpenAI(
        base_url=f"http://0.0.0.0:{port}/v1",
        api_key="token-abc123")
    
    if not check_server_healthy(port, model_name, timeout_seconds=150):
        try:
            terminate_processes([graph_server_process])
        except Exception as e:
            logging.error(f"Graph failed to start:\n\n{e}")
            raise

    memory = MemorySaver()

    model = OpenAICompatibleChatModel(
        client=client,
        model_name=model_path,
        model_type="Llama")
    
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
    return workflow.compile(checkpointer=memory), graph_server_process


if __name__ == "__main__":
    
    import asyncio
    import tracemalloc
    import logging

    from src.registry import GraphManager
    from src.graphs.utils import run_graph_in_terminal

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    tracemalloc.start()

    GM = GraphManager()

    try:
        session_id = str(uuid4())
        asyncio.run(
            run_graph_in_terminal(
                graph_manager=GM,
                config="llama3_2_3b_on_vllm_server",
                session_id=session_id))
    finally:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logger.info("\nMemory traceback:")
        for stat in top_stats[:3]:
            logger.info(stat)
