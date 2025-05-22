import os
import sys
from uuid import uuid4
import json
import asyncio
import requests
from typing import Annotated, TypedDict, TypeVar, List, Dict, Any, Optional
from pydantic import BaseModel as DataBaseModel
from pydantic import Field, field_validator, Field, PrivateAttr
from dotenv import load_dotenv
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.messages import RemoveMessage, AnyMessage
from langgraph.graph import add_messages
from langchain_openai import ChatOpenAI
from fastapi.templating import Jinja2Templates
from jinja2.environment import Template


# for testing purposes:
p = "/".join(sys.path[0].split('/')[:-2])
if p not in sys.path:
    sys.path.append(p)

from src.registry import tool_graph
from src.entry import LoggedAttribute, BaseEntry
from mediacheck.utils import RedditClient
from utils import PlaceholderModel
from src import models

# loading variables from .env file
load_dotenv()

T = TypeVar('T', bound='FlowState')
templates = Jinja2Templates(directory="templates")


class PostEntry(DataBaseModel):
    
    title: str
    author: str
    url: str
    score: int
    num_comments: int
    created_utc: float
    media: Dict | None
    is_video: bool
    permalink: str
    thumbnail: str | None
    selftext_html: str | None


class PostEntrySet(BaseEntry):

    type: str = Field(default="posts_gallery", frozen=True)
    posts: List[PostEntry]
    graph_session: str

    # # Private template field using PrivateAttr
    # _template: Template = PrivateAttr()

    @classmethod
    def restore_from_record(cls, record: models.GraphLog):
        restore_data = json.loads(record.item_content)
        return cls(**restore_data)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize template
        self._template = templates.get_template("partials/gallery_items.html")

    def _format_content(self) -> dict:
        return {"posts": [
                    {
                        "title": post.title,
                        "author": post.author,
                        "url": post.url,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "created_utc": post.created_utc,
                        "media": post.media,
                        "is_video": post.is_video,
                        "permalink": post.permalink } for post in self.posts]
                }

    def content_to_send(self) -> str:
        """ Renders the document content using the template.
        
        Returns:
            str: HTML-rendered content of the document which will be shown in UI
        """
        html = self._template.render({
            "posts": self._format_content()["posts"],
            "graph_session": self.graph_session
        })
        return html

    def content_to_store(self) -> str:
        """ Prepare data of the item to be stored in DataBase """

        return json.dumps(self._format_content())




class FlowState(TypedDict):

    source: str
    messages: Annotated[list[AnyMessage], add_messages]
    posts: List[PostEntry]
    content: Optional[LoggedAttribute] = None


@tool_graph(name='mediacheck', tag="Agent", entries_map=True)
def get_mediacheck(agent_graph: str, model_name: str, **kwargs):
    
    kwargs.pop("port", None)
    sampling_data = kwargs.pop("sampling", {})

    llm = ChatOpenAI(
        streaming=True,
        api_key=os.environ["OPENAI_API_KEY"],
        model=model_name,
        **kwargs)
    
    reddit_client = RedditClient()

    async def run_reddit_client(state: FlowState, config: dict):
        graph_session = config["metadata"]["thread_id"]
        subreddit = 'singularity'
        response_text = f"""I will take new posts from Subreddit {subreddit}"""
        model = PlaceholderModel()
        posts = reddit_client.get_subreddit_posts(subreddit, limit=20)
        response = await model.ainvoke(response_text)
        posts = reddit_client.get_subreddit_posts(subreddit, limit=20)
            
        logged_attrs = LoggedAttribute(
            content=PostEntrySet.model_validate(
                {
                    "posts": [PostEntry.model_validate(post) for post in posts],
                    "graph_session": graph_session
                }
            )
        )

        return {"content": logged_attrs}


    workflow = StateGraph(FlowState)
    workflow.add_node("run_reddit_client", run_reddit_client)

    workflow.set_entry_point("run_reddit_client")
    workflow.add_edge("run_reddit_client", END)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app, llm


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
                config="mediacheck",
                session_id=session_id))
    finally:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logger.info("\nMemory traceback:")
        for stat in top_stats[:3]:
            logger.info(stat)