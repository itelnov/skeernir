import os
import sys
import re
import logging
import subprocess
import json
from uuid import uuid4
from typing import Union, Tuple
from typing import Annotated, TypedDict, Dict, Any, Type, TypeVar


from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import RemoveMessage, AnyMessage

from ollama import AsyncClient as AsyncOllamaClient
from ollama import Client as OllamaClient
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel
from langchain.callbacks.manager import CallbackManager
from langchain_core.messages import HumanMessage
from pathlib import Path
from deckgen.utils import run_code, save_and_run_python


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
DECKGEN = "deckgen"
EXECUTOR = "executor"


T = TypeVar('T', bound='FlowState')


class NoMessageCallbackManager(CallbackManager):

    def __init__(self):
        super().__init__(handlers=[])  # Initialize with empty handlers list

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        pass


class UserIntention(BaseModel):
    action: int


class SourceCode(BaseModel):
    python_script: str


class FlowState(TypedDict):

    source: str
    messages: Annotated[list[AnyMessage], add_messages]
    deck_draft: str
    feedback: str
    last_decision: str
    web_search: bool
    last_node: str
    deck_code: Annotated[list[AnyMessage], add_messages]
    exec_required: bool = False

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
            "deck_code": state["deck_code"] if preserve_messages else [
                RemoveMessage(id=m.id) for m in state["deck_code"]],
            "exec_required": False
        }
        
        return fresh_state


DECKGEN_FORMAT = (
    "To sucessfully execute this task, always precisely follow the next "
    "instructions:\n\n"
    "- Always iclude all parts of the deck into draft, starting from `Slide 1`.\n"       
    "- Do not consider formatting options for the draft, you have to focus "
    "on content only.\n"
    "- Start the draft with tag `!DRAFT_STARTS!`\n"
    "- End the draft with tag `!DRAFT_ENDS!`\n"
    "- Enumerate all slides. Keep slides enumeration in order!\n"
    "- Only after !DRAFT_ENDS! always write suggestions how to improve the deck draft with additrional "
    "information. \n\n"
    "Example of a deck draft:\n"
    "Slide `slide number`: `some content you generate`\n\n"
    "Slide `slide number`: `some content you generate`\n\n"
    "..."
    "My Suggestions how to improve: `suggestions for improvement`\n"
    "\n\n")


DRAFTER_SYSTEM_MESSAGE = (
    "You are an Assistant which helps to write a draft for presentation "
    "deck based on the context provided.\n\n"
    f"{DECKGEN_FORMAT}"
    )


def get_drafter_prompt_to_improve(draft: str, feedback: str) -> str:
    prompt = (
        f"Improve the deck draft according to feedback:\n"
        f"Previus version:\n {draft}\n\n"
        f"Feedback:\n {feedback}\n\n"
        f"{DECKGEN_FORMAT}"
        )
    return prompt


def get_deckgen_sys_message()-> str:
    
    prompt = (
        "You are an Assistant which helps to write code using python-pptx "
        "user guide"
        )
    return prompt


@tool_graph(name='deckgen', tag="Agent", entries_map=True)
def get_deckgen(
        host: Dict,
        output_pth: str,
        **kwargs) -> Union[CompiledStateGraph, Tuple[CompiledStateGraph, Any]]:
    try:
        port = kwargs.pop("port", 11434)
        tool_model_name = host["tool_model_name"]
        model_name = host["model_name"]
        sampling_data = kwargs.get("sampling", {})
        tools_sampling_data = kwargs.get("tools_sampling_data", {})
        Path(output_pth).mkdir(parents=True, exist_ok=True)

        graph_server_process = None
        graph_server_process = run_server(
            host["host_name"],
            model_name,
            port=port,
            use_gpu=host["use_gpu"], 
            **kwargs)

        if not check_server_healthy(
            port, model_name, time_to_wait=2, timeout_seconds=10):
            try:
                terminate_processes([graph_server_process])
            except Exception as e:
                logging.error(f"Graph failed to start: \n{e}")
                raise 
        
        # TODO do smth with this sheet
        my_env = os.environ.copy()
        my_env["OLLAMA_HOST"] = f"0.0.0.0:{port}"
        logging.info("Start pull if will be needed")
        pull_process_a = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=my_env
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

        # port2 = port + 77
        # graph_server_process = run_server(
        #     server,
        #     model_name,
        #     port=port2,
        #     use_gpu=False, 
        #     **kwargs)
        # my_env = os.environ.copy()
        # my_env["OLLAMA_HOST"] = f"0.0.0.0:{port2}"
        # pull_process_b = subprocess.Popen(
        #     ["ollama", "pull", tool_model_name],
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     text=True,
        #     bufsize=1,
        #     env=my_env
        #     )

        # while True:
        #     stdout_line = pull_process_b.stdout.readline()
        #     stderr_line = pull_process_b.stderr.readline()
            
        #     if stdout_line == '' and stderr_line == '' and pull_process_b.poll() is not None:
        #         break
            
        #     if stdout_line:
        #         logging.info(f"STDOUT: {stdout_line.strip()}")
        #     if stderr_line:
        #         logging.info(f"STDERR: {stderr_line.strip()}")

        
        llm = OllamaClientWrapper(
            model_name=model_name,
            client=AsyncOllamaClient(
            host=f'http://0.0.0.0:{port}',
            headers={'x-some-header': 'some-value'}
            )
        )

        # from langchain_openai import ChatOpenAI
        # generator = ChatOpenAI(
        #     streaming=True,
        #     api_key=api_token,
        #     model="gpt-4o-mini-2024-07-18",
        #     **kwargs)
        
        # tool_caller = OllamaClientWrapper(
        #     model_name=tool_model_name,
        #     client=OllamaClient(
        #         host=f'http://0.0.0.0:{port2}',
        #         headers={'x-some-header': 'some-value'}
        #     )
        # )

        system_messager = PlaceholderModel(model_name="ghost")

        async def run_drafter(state: FlowState):
            
            user_content = state["messages"]
            llm.system_message = DRAFTER_SYSTEM_MESSAGE

            if state.get("last_node", "") == VERIFY_DRAFTER:                

                llm.system_message = "You are helpfull assistant."
                user_message = get_drafter_prompt_to_improve(
                    state["deck_draft"].content,
                    state["feedback"].content
                )
                user_content = [HumanMessage(content=[
                    {"type": "text", "text": user_message}])]
                
            generated = await llm.ainvoke(user_content, **sampling_data)
            sampling_data["response_text"] = (
                "I need your feedback now. Type what to improve or type /yes "
                "to continue")
            _ = await system_messager.ainvoke([], **sampling_data)
            return {
                "deck_draft": generated,
                "messages": [generated, _],
                "last_node": DRAFTER
            }
        

        async def verify_drafter(state: FlowState):
            feedback = interrupt("Feedback")["Feedback"]
            return {
                "feedback": feedback,
                "last_node": VERIFY_DRAFTER
            }

        async def should_continue(state: FlowState):
            if state["last_node"] == VERIFY_DRAFTER:
                llm.system_message = (
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
                    "User: `Lets add some info into slide 3 ...` -> action: 0\n\n"
                    "User: `wtf` -> action: 2\n\n"
                    "User: `it's ok, lets continue` -> action: 1\n\n"
                    )
                if state["feedback"].content[0]["text"] == "/yes":
                    action = 1
                else:
                    try:
                        result = llm.invoke(
                            [state["feedback"]], 
                            format=UserIntention.model_json_schema(),
                            **tools_sampling_data)
                        action = json.loads(result.content)['action']
                    except Exception as e:
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
                        "\n\nGreat, Let's continue!\n\n")
                    _ = await system_messager.ainvoke(
                        [], **sampling_data)
                    return CLEANUP
                
                if int(action) == 0:
                    return DRAFTER
                
            # if state["last_node"] == EXECUTOR:
            #     if state["exec_required"]:
            #         return DECKGEN
            #     return CLEANUP

        async def run_deckgen(state: FlowState):
            
            with open("src/graphs/deckgen/test.md", 'r', encoding='utf-8') as file:
                mk_user_gide = file.read()
            generator.system_message = get_deckgen_sys_message()
            
            if not state["deck_code"]:
                try:
                    m = state["deck_draft"].content
                    # start_index = m.index('<think>')
                    # end_index = m.index('</think>') + len('</think>')
                    start_index = m.index('!DRAFT_STARTS!')
                    end_index = m.index('!DRAFT_ENDS!')
                    deck_draft = m[start_index:end_index]
                except ValueError:
                    pass

                context = (
                    f"PYTHON-PPTX USER GUIDE:\n\n{mk_user_gide}\n\n"
                    f"TEXT DRAFT FOR THE DECK:{deck_draft}\n\n"
                    "Given python-pptx user guide and detailed text draft for the deck your task is to convert all this"
                    " text into a PPTX file using python-pptx. To sucessfully execute this task, "
                    "precisely follow the next INSTRUCTIONS:\n\n"
                    "1. Plan the structure of the code you will write using python-pptx lib and the user guide, "
                    "considering the next features:\n"
                    "- Consider the case when `python-pptx` library is installed in your working environment.\n"
                    f"- The PPTX file should be saved in {output_pth} folder.\n"
                    "- The solution should be defined as a funciton which does not take arguments\n"
                    "- The code needs to create a new presentation, add slides one by one, and populate each with the"
                    " corresponding content from the text draft\n"
                    "- Use titles, lists and texts. Do not apply formatting options and do not add images.\n"
                    "2. Analyze your plan regarding python-pptx user guide and code examples it contains. Focus on:\n"
                    "- Usage objects and methods which are available for python-pptx lib and used in user guide\n"
                    "- Usage python-pptx lib conventionally.\n"
                    "- Avoiding AttributeError using python-pptx objects during script execution\n"
                    "3. Write full end-to-end python script which generates and save PPTX file with all slides inside. "
                    "Be sure that the script is executable and safe.\n"
                    "4. Check if content for all slides in the text draft was hardcoded. If not, improve the script.\n")
                
                state["deck_code"] = [
                    HumanMessage(
                        content=[{"type": "text", "text": context}])
                    ]
            
                generated = await generator.ainvoke(
                    state["deck_code"], **sampling_data)
                state["deck_code"].append(generated)
            
            else:
                generated = await generator.ainvoke(
                    state["deck_code"], **sampling_data)
                state["deck_code"] = [
                    RemoveMessage(id=m.id) for m in state["deck_code"][-2:]]
                state["deck_code"].append(generated)

            return {
                "messages": generated,
                "deck_code": state["deck_code"]
            }

        async def execute_code(state: FlowState):
            
            try:
                code_pattern = r'```(?:python|Python|py|Py)\s([\s\S]*?)\s*```'
                source_code = re.findall(
                    code_pattern, 
                    state["deck_code"][-1].content, re.IGNORECASE)[-1]
                result, stdout, stderr = save_and_run_python(
                    source_code,
                    allowed_path=output_pth,
                    filename="generated_script.py",
                )
            
            except Exception as e:
                return {
                    "last_node": EXECUTOR,
                    "deck_code": e,
                    "exec_required": True
                    }

            if stderr:
                return {
                    "last_node": EXECUTOR,
                    "deck_code": (
                        f"The script written by you was executed and the next error was received:\n\n{stderr}\n\n."
                        "Your task is to solve the issue using the python-pptx user guide and re-write the corrected "
                        "script. To sucessfully execute this task, precisely follow the next INSTRUCTIONS:\n\n"
                        "1. If the error related to python-pptx library, find the solution in python-pptx user guide."
                        " Otherwise use your PYTHON knowledge to solve the issue\n"
                        "2. Define what changes to the code you need to apply\n"
                        "3. Write corrected end-to-end python script with changes which generates and save PPTX file in"
                        f" {output_pth} folder. Be sure that the script is executable, safe and the code "
                        "creates a new presentation, add all slides one by one, and populate each with the "
                        "corresponding content from the text draft  for the deck ."),
                    "exec_required": True
                    }

            return {
                "last_node": EXECUTOR,
                "exec_required": False
                }

        async def clean_up(state: FlowState) -> Dict[str, Any]:

            """End node that resets state after processing"""
            sampling_data["response_text"] = (
                "\n\nCleaning up...\n\nBye Bye!")
            _ = await system_messager.ainvoke([], **sampling_data)
            return FlowState.reset(state, preserve_messages=False)

        # Define a new graph
        workflow = StateGraph(MessagesState)
        workflow.add_node(DRAFTER, run_drafter)
        workflow.add_node(VERIFY_DRAFTER, verify_drafter)
        # workflow.add_node(DECKGEN, run_deckgen)
        # workflow.add_node(EXECUTOR, execute_code)
        workflow.add_node(CLEANUP, clean_up)
        
        workflow.add_edge(START, DRAFTER)
        workflow.add_edge(DRAFTER, VERIFY_DRAFTER)
        
        workflow.add_conditional_edges(
            source=VERIFY_DRAFTER,
            path=should_continue,
            path_map={
                VERIFY_DRAFTER: VERIFY_DRAFTER,
                CLEANUP: CLEANUP,
                DRAFTER: DRAFTER
            }
        )
        # workflow.add_edge(DECKGEN, EXECUTOR)
        # workflow.add_conditional_edges(
        #     source=EXECUTOR,
        #     path=should_continue,
        #     path_map={
        #         DECKGEN: DECKGEN,
        #         CLEANUP: CLEANUP
        #     }
        # )
        workflow.add_edge(CLEANUP, END)
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
    
    except Exception as e:
        logging.error(e)
        try:
            terminate_processes(
                [graph_server_process, pull_process_a])
        except:
            pass
        raise

    # app.get_graph().draw_mermaid_png(
    #     output_file_path="deckgen.png")

    return app, graph_server_process, pull_process_a


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
