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


class PlaceholderModel(BaseChatModel):
    
    model_name: str = Field(default="ghost")
    response_text: str

    """A placeholder model that always returns a predefined AI message."""
    
    def __init__(self, response_text: str = "This is a placeholder response.", 
                 **kwargs):
        """
        Initialize the placeholder model.
        
        Args:
            response_text (str): The text to return in the AI message.
                               Defaults to "This is a placeholder response."
        """
        super().__init__(response_text=response_text, **kwargs)

    def _generate(self, messages, stop = None, run_manager = None, **kwargs):
        return super()._generate(messages, stop, run_manager, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        
        for word in self.response_text.split(" "):

            if word:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=word + " "))
                
                yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "ghost_placeholder"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {"model_name": self.model_name}
    

@tool_graph(name='Ghost', tag="welcom")
def get_ghost(**kwargs):
    
    memory = MemorySaver()
    kwargs.pop("port", None)
    sampling_data = kwargs.pop("sampling", {})

    model = PlaceholderModel(
        model_name="Ghost",
        response_text=("Oh, welcome to Skeernir! Pick your Agent, sit back, and "
                       "have a blast... or don't. Or maybe you'll just love "
                       "hating every second of it. Either way, we promise "
                       "something.")
    )
    
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