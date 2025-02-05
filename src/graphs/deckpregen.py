import os
import sys
import logging
import subprocess
import json
from uuid import uuid4
from typing import Union, Tuple
from typing import List, TypedDict, Dict, Any, Optional, Type, TypeVar

from pydantic import BaseModel as DataBaseModel
from pydantic import Field, field_validator, Field, PrivateAttr
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.runnables import Runnable, chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from jinja2.environment import Template
from fastapi.templating import Jinja2Templates
from langchain_core.messages import RemoveMessage
from ollama import AsyncClient as AsyncOllamaClient
from ollama import Client as OllamaClient
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain.callbacks.manager import CallbackManager


# for testing purposes:
p = "/".join(sys.path[0].split('/')[:-2])
if p not in sys.path:
    sys.path.append(p)


from src.registry import tool_graph
from src.entry import LoggedAttribute
from src.graphs.utils import OllamaClientWrapper, PlaceholderModel, flatten_list
from src.registry import tool_graph, terminate_processes
from src.graphs.utils import (
    OllamaClientWrapper, run_server, check_server_healthy)


CLEANUP = "cleanup"
DRAFTER = "drafter"
VERIFY_DRAFTER = "verify_drafter"


T = TypeVar('T', bound='FlowState')


class NoMessageCallbackManager(CallbackManager):

    def __init__(self):
        super().__init__(handlers=[])  # Initialize with empty handlers list

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        pass


class UserIntention(BaseModel):
  action: int


class FlowState(TypedDict):

    source: str
    messages: str
    deck_draft: str
    feedback: str
    last_decision: str
    web_search: bool
    last_node: str
    documents: List[Document]
    from_graph: Optional[LoggedAttribute] = None
    human_input_required: bool = False
    human_input_intro: str = ""

    @classmethod
    def reset(cls: Type[T], state: T, preserve_messages: bool = False) -> T:

        fresh_state: T = {
            "messages": state["messages"] if preserve_messages else [
                RemoveMessage(id=m.id) for m in state["messages"]],
            "source": "",
            "deck_draft": "",
            'last_node': "",
            "web_search": False,
            "last_decision": "",
            "human_input_required": False,
            "human_input_intro": "",
            "documents": [],
            "from_graph": None
        }
        
        return fresh_state


DRAFTER_SYSTEM_MESSAGE = (
        "You are an Assistant which helps to write "
        "presentation deck based on the information provided by User. "
        "Precisely follow the next instructions:\n\n"
        "- Use only information provided by User. Do not "
        "consider formatting options for the deck, you have to focus "
        "on content only.\n"
        "- Start writing deck with a tag `!START!`.\n"
        "- Take into account all information and write content "
        "for each slide in deck according to it.\n"
        "- Finish draft with a tag `!FINISH!`.\n"
        "- Write suggestions how to improve the draft with additrional "
        "information.\n\n"
        "Example of a draft:\n"
        "!START!\n\n"
        "Slide 1: `some content you generate`\n\n"
        "Slide 2: `some content you generate`\n\n"
        "..."
        "!FINISH!\n\n"
        "My Suggestions how to improve: `suggestions for improvement`\n\n"
        "\n\n"
        )



@tool_graph(name='deckgen', tag="Agent", entries_map=True)
def get_deckgen(
        server: str,
        model_name: str,
        **kwargs) -> Union[CompiledStateGraph, Tuple[CompiledStateGraph, Any]]:
    try:
        port = kwargs.pop("port", 11434)
        tool_model_name = kwargs["tool_model_name"]
        sampling_data = kwargs.get("sampling")
        tools_sampling_data = kwargs.get("tools_sampling_data")
        
        os.environ["OLLAMA_HOST"] = f"0.0.0.0:{port}"
        graph_server_process = None

        graph_server_process = run_server(
            server,
            model_name,
            port=port, 
            use_gpu=True, 
            **kwargs)

        if not check_server_healthy(
            port, model_name, time_to_wait=2, timeout_seconds=10):
            try:
                terminate_processes([graph_server_process])
            except Exception as e:
                logging.error(f"Graph failed to start: \n{e}")
                raise 
            
        logging.info("Start pull if will be needed")
        pull_process_a = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            )

        while True:
            stdout_line = pull_process_a.stdout.readline()
            stderr_line = pull_process_a.stderr.readline()
            
            if stdout_line == '' and stderr_line == '' and pull_process_a.poll() is not None:
                break
            
            if stdout_line:
                logging.info(f"STDOUT: {stdout_line.strip()}")
            if stderr_line:
                logging.info(f"STDERR: {stderr_line.strip()}")

        pull_process_b = subprocess.Popen(
            ["ollama", "pull", tool_model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            )

        while True:
            stdout_line = pull_process_b.stdout.readline()
            stderr_line = pull_process_b.stderr.readline()
            
            if stdout_line == '' and stderr_line == '' and pull_process_b.poll() is not None:
                break
            
            if stdout_line:
                logging.info(f"STDOUT: {stdout_line.strip()}")
            if stderr_line:
                logging.info(f"STDERR: {stderr_line.strip()}")

        
        ds_drafter = OllamaClientWrapper(
            model_name=model_name,
            client=AsyncOllamaClient(
            host=f'http://0.0.0.0:{port}',
            headers={'x-some-header': 'some-value'}
            )
        )
        
        ds_drafter.system_message = DRAFTER_SYSTEM_MESSAGE
        
        tool_caller = OllamaClientWrapper(
            model_name=tool_model_name,
            client=OllamaClient(
                host=f'http://0.0.0.0:{port}',
                headers={'x-some-header': 'some-value'}
            )
        )

        system_messager = PlaceholderModel(model_name="ghost")

        async def run_drafter(state: FlowState):
            
            user_content = state["messages"]
            if state.get("last_node", "") == VERIFY_DRAFTER:
                user_content = flatten_list([ #user_content, 
                    state["deck_draft"], state["feedback"]])

            generated = await ds_drafter.ainvoke(user_content, **sampling_data)
            
            sampling_data["response_text"] = (
                "\n\n*************************************************\n\n"
                "\n\n"
                "I need your feedback now. Type what to improve or let me "
                "continue")
            _ = await system_messager.ainvoke([], **sampling_data)
            
            return {
                "source": user_content,
                "deck_draft": generated,
                "last_node": DRAFTER
                }

        async def verify_drafter(state: FlowState):
            feedback = interrupt("Feedback")
            state["feedback"] = feedback
            state["last_node"] = VERIFY_DRAFTER
            return state


        async def clean_up(state: FlowState) -> Dict[str, Any]:

            """End node that resets state after processing"""
            sampling_data["response_text"] = (
                "\n\nCleaning up...\n\nBye Bye!")
            _ = await system_messager.ainvoke([], **sampling_data)
            
            return FlowState.reset(state, preserve_messages=False)


        async def should_continue(state: FlowState):
            if state["last_node"] == VERIFY_DRAFTER:
                tool_caller.system_message = (
                    "You are an assitent. Try to understand the User intention "
                    "from the message and conclude:\n\n"
                    "0. The User wants to step back and improve the content.\n"
                    "1. The User wants to continue with the content.\n"
                    "2. You don't understand the message and the User intention.\n"
                    "You might not be provided with content, make a conclusion "
                    "from the User message only. Respond with a JSON object "
                    "containing parameter `action` and the value of conclusion: [0, 1, 2]."
                    "Examples:\n\n"
                    "User: `I dont know what to say...` -> action: 2\n\n"
                    "User: `Great, let's go on!` -> action: 1\n\n"
                    "User: `Lets add some info into slide 3` -> action: 0\n\n"
                    "User: `wtf` -> action: 2\n\n"
                    )
                try:
                    result = tool_caller.invoke(
                        [state["feedback"]], 
                        format=UserIntention.model_json_schema(),
                        **tools_sampling_data)
                    action = json.loads(result.content)['action']
                except:
                    action = 2
                
                if int(action) == 2:
                    sampling_data["response_text"] = (
                        "\n\nNot really understood your intention. Write what "
                        "to improve or move on!")
                    _ = await system_messager.ainvoke(
                        [], **sampling_data)
                    return VERIFY_DRAFTER
                
                if int(action) == 1:
                    sampling_data["response_text"] = (
                        "\n\nGreat, Let's finish with this task!")
                    _ = await system_messager.ainvoke(
                        [], **sampling_data)
                    return CLEANUP
                
                if int(action) == 0:
                    return DRAFTER

        # Define a new graph
        workflow = StateGraph(MessagesState)
        workflow.add_node(DRAFTER, run_drafter)
        workflow.add_node(VERIFY_DRAFTER, verify_drafter)
        workflow.add_node(CLEANUP, clean_up)
        workflow.add_edge(START, DRAFTER)
        workflow.add_edge(DRAFTER, VERIFY_DRAFTER)
        # workflow.add_edge(START, VERIFY_DRAFTER)
        workflow.add_conditional_edges(
            source=VERIFY_DRAFTER,
            path=should_continue
        )

        workflow.add_edge(CLEANUP, END)
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
    
    except Exception as e:
        logging.error(e)
        try:
            terminate_processes(
                [graph_server_process, pull_process_a, pull_process_b])
        except:
            pass
        raise

    app.get_graph().draw_mermaid_png(
        output_file_path="deckgen.png")

    return app, graph_server_process, pull_process_a, pull_process_b


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
                config="deckgen",
                session_id=session_id))
    finally:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logger.info("\nMemory traceback:")
        for stat in top_stats[:3]:
            logger.info(stat)
