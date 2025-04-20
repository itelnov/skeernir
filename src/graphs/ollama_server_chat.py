import os
import sys
import logging
import subprocess
from uuid import uuid4
from typing import Union, Tuple, Any

from ollama import AsyncClient as AsyncOllamaClient
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    AIMessageChunk, HumanMessage)

# for testing purposes:
p = "/".join(sys.path[0].split('/')[:-2])
if p not in sys.path:
    sys.path.append(p)

from src.registry import tool_graph, terminate_processes, terminate_threads
from src.graphs.utils import (
    OllamaClientWrapper, run_server, check_server_healthy)


"""
https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
https://github.com/ollama/ollama/blob/main/docs/modelfile.md#build-from-a-gguf-file
"""


@tool_graph(name='model_on_ollama_server', tag="chat", att_modals=['text', 'image'])
def get_model_on_ollama_server(
    agent_graph: str,
    host_name: str,
    model_name: str,
    **kwargs) -> Union[CompiledStateGraph, Tuple[CompiledStateGraph, Any]]:
    try:   
        memory = MemorySaver()
        port = kwargs.pop("port", 11434)
        os.environ["OLLAMA_HOST"] = f"0.0.0.0:{port}"
        host_params = kwargs.pop('host_params', {})
        graph_server_process = run_server(
            host_name,
            model_name,
            port=port, 
            use_gpu=True, 
            **host_params)

        if not check_server_healthy(
            port, model_name, time_to_wait=2, timeout_seconds=10):
            try:
                terminate_processes([graph_server_process])
            except Exception as e:
                logging.error(f"Graph failed to start: \n{e}")
                raise 
            
        logging.info("Start pull if will be needed")
        pull_process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            )

        while True:
            stdout_line = pull_process.stdout.readline()
            stderr_line = pull_process.stderr.readline()
            
            if stdout_line == '' and stderr_line == '' and pull_process.poll() is not None:
                break
            
            if stdout_line:
                logging.info(f"STDOUT: {stdout_line.strip()}")
            if stderr_line:
                logging.info(f"STDERR: {stderr_line.strip()}")

        client = AsyncOllamaClient(
            host=f'http://0.0.0.0:{port}',
            headers={'x-some-header': 'some-value'}
            )
        model = OllamaClientWrapper(
            model_name=model_name,
            client=client,
            )
        # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        # Sampling data should be aligned with these from link above
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
    
    except Exception as e:
        logging.error(e)
        try:
            terminate_processes([graph_server_process, pull_process])
        except:
            pass
        raise

    return app, graph_server_process, pull_process


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
        
    session_id = str(uuid4())
    try:
        session_id = str(uuid4())
        asyncio.run(
            run_graph_in_terminal(
                graph_manager=GM,
                config="ollama_server_model",
                session_id=session_id))
    finally:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logger.info("\nMemory traceback:")
        for stat in top_stats[:3]:
            logger.info(stat)
