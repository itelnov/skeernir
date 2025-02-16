import sys
import json
from typing import List, TypedDict, Dict, Any, Optional, Type, TypeVar, Annotated
from uuid import uuid4

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
from langgraph.graph import add_messages
from jinja2.environment import Template
from fastapi.templating import Jinja2Templates
from langchain_core.messages import RemoveMessage, AnyMessage



# for testing purposes:
p = "/".join(sys.path[0].split('/')[:-2])
if p not in sys.path:
    sys.path.append(p)
from src.registry import tool_graph
from src.entry import LoggedAttribute, BaseEntry
from src import models


templates = Jinja2Templates(directory="templates")

RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"
CLEANUP = "cleanup"


T = TypeVar('T', bound='FlowState')


class FlowState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    web_search: bool
    documents: List[Document]
    from_graph: Optional[LoggedAttribute] = None

    @classmethod
    def reset(cls: Type[T], state: T, preserve_messages: bool = False) -> T:
        """
        Reset the FlowState to its initial values.
        
        Args:
            state: Current FlowState instance
            preserve_messages: Whether to preserve the messages field
            
        Returns:
            A new FlowState instance with reset values
        """

        fresh_state: T = {
            "messages": state["messages"] if preserve_messages else [
                RemoveMessage(id=m.id) for m in state["messages"]],
            "generation": "",
            "web_search": False,
            "documents": [],
            "from_graph": None
        }
        
        return fresh_state


class GradeAnswer(DataBaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GradeDocuments(DataBaseModel):
    
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

    # You can add custom validation logic easily with Pydantic.
    @field_validator("binary_score")
    def binary_score_is_yes_or_no(cls, field):
        if field.lower() not in ['yes', 'no']:
            raise ValueError("Badly formed score!")
        return field


class DocumentEntry(BaseEntry):
    """A Pydantic model representing a document entry that renders HTML output
       into Skeernir UI.
    
    The class has immutable type and template attributes, while requiring
    title and page_content for initialization.
    """
    
    type: str = Field(default="document", frozen=True)
    title: str
    page_content: str
    source: str = Field(default="")

    # Private template field using PrivateAttr
    _template: Template = PrivateAttr()

    @classmethod
    def restore_from_record(cls, record: models.GraphLog):
        restore_data = json.loads(record.item_content)
        return cls(**restore_data)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize template
        self._template = templates.get_template("partials/document_log.html")

    def _format_content(self) -> dict:
        return {
            "title": self.title,
            "source": self.source,
            "page_content": self.page_content
        }

    def content_to_send(self) -> str:
        """ Renders the document content using the template.
        
        Returns:
            str: HTML-rendered content of the document which will be shown in UI
        """
        return self._template.render(self._format_content())

    def content_to_store(self) -> str:
        """ Prepare data of the item to be stored in DataBase """

        return json.dumps(self._format_content())
    

def get_retrival_grader_chain(api_token: str) -> Runnable:
    """
    Creates a runnable chain for grading the relevance of retrieved documents.

    Args:
        api_token (str): OpenAI API token for authentication.

    Returns:
        Runnable: A chain of operations for grading documents.
    """
    tool_name = convert_to_openai_tool(GradeDocuments)["function"]["name"]
    llm = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        api_key=api_token,
        temperature=0).bind_tools(
        [GradeDocuments], tool_choice=tool_name, parallel_tool_calls=False)
    
    parser = PydanticToolsParser(tools=[GradeDocuments], first_tool_only=True)

    system = ("You are a grader assessing relevance of a retrieved document to"
              " a user question.\n If the document contains keyword(s) or "
              "semantic meaning related to the question, grade it as relevant. "
              "\nGive a binary score 'yes' or 'no' score to indicate whether "
              "the document is relevant to the question.")

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", ("Retrieved document: \n\n {document} \n\n "
                       "User question: {question}")),
        ])

    retrieval_grader = grade_prompt | llm | parser

    return retrieval_grader


def get_generation_chain(api_token: str) -> Runnable:
    """
    Creates a runnable chain for generating text based on the context provided 
    by documents.

    Args:
        api_token (str): OpenAI API token for authentication.

    Returns:
        Runnable: A chain of operations for text generation.
    """
    llm = ChatOpenAI(
        temperature=0.75,
        model="gpt-4o-mini-2024-07-18",
        api_key=api_token)
    prompt = hub.pull("rlm/rag-prompt")
    generation_chain = prompt | llm
    return generation_chain


def get_web_search_tool(tavily_api_key: str) -> Runnable:
    """
    Creates a runnable tool for performing web search based on a query.

    Args:
        tavily_api_key (str): Tavily API key for authentication.

    Returns:
        Runnable: A chain of operations for web search.
    """
    search = TavilySearchAPIWrapper(
        tavily_api_key=tavily_api_key)
    web_search_tool = TavilySearchResults(api_wrapper=search, max_results=5)
    return web_search_tool


@tool_graph(name='openai-rag-corrective', tag="RAG", entries_map=True)
def get_corrective_rag_graph(
    api_token: str,
    tavily_api_key: str,
    **kwargs) -> StateGraph:
    """
    Constructs and returns a StateGraph for the RAG (Retrieval-Generation) pipeline.

    Args:
        api_token (str): OpenAI API token for authentication.
        tavily_api_key (str): Tavily API key for web search.
        **kwargs: Additional configuration parameters.

    Returns:
        StateGraph: The constructed graph with nodes and edges for the RAG pipeline.
    """
    # Define our chain and tools
    retriever = Chroma(
        collection_name=kwargs["collection_name"],
        persist_directory=kwargs["vectordb_path"],
        # client_settings=Settings(anonymized_telemetry=False),
        embedding_function=OpenAIEmbeddings(api_key=api_token)).as_retriever(
            search_type="similarity", search_kwargs={"k": 5})
    
    retrieval_grader_chain = get_retrival_grader_chain(api_token)
    generation_chain = get_generation_chain(api_token)
    web_search_tool = get_web_search_tool(tavily_api_key)
    
    # Define functions on our nodes
    async def run_retrieve(state: FlowState) -> Dict[str, Any]:
        question = state["messages"][0].content
        if isinstance(question, List):
            question = question[0]["text"]
        documents = await retriever.ainvoke(question)

        """ We add `from_graph` into state which is defined as LoggedAttribute 
        class. During streaming the output from graph, all classes which inherit 
        from LoggedAttribute class will be considered to log into RIGHT-SIDE 
        panel. See LoggedAttribute and DocumentEntry class definitions for more 
        details."""
        
        return {
            "documents": documents,
            "from_graph": LoggedAttribute(content="Documents retrieved")
            }

    async def run_grade_documents(state: FlowState) -> Dict[str, Any]:
        """
        Grades the retrieved documents for relevance to the question.

        Args:
            state (FlowState): The current graph state.

        Returns:
            dict: Contains filtered documents and web search flag.
        """
        question = state["messages"][0].content
        documents = state["documents"]

        filtered_docs = []
        web_search = False if documents else True
        for d in documents:
            score = await retrieval_grader_chain.ainvoke(
                {"question": question, "document": str(d.page_content)}
            )
            grade = score.binary_score
            if grade.lower() == "yes":
                filtered_docs.append(d)
            else:
                web_search = True
                continue
        
        return {
            "web_search": web_search,
            "documents": filtered_docs,
            "from_graph": LoggedAttribute(content=(
                f"Documents graded. Filtered documents: {len(filtered_docs)}"
                f" Web search required: {str(web_search)}"))
            }

    async def run_web_search(state: FlowState) -> Dict[str, Any]:
        """
        Performs web search if required.

        Args:
            state (FlowState): The current graph state.

        Returns:
            dict: Contains web search results and question.
        """
        question = state["messages"][0].content
        if isinstance(question, List):
            question = question[0]["text"]
        documents = state["documents"]
        web_docs = await web_search_tool.ainvoke({"query": question})
        if "Exception" in web_docs:
            return {
                "from_graph": LoggedAttribute(
                    content="Web search failed, continue")
                    }
        else:
            web_docs = [
                Document(
                    page_content=d["content"], 
                    metadata={
                        "source": d["url"], 
                        "title": ' '.join(
                            str(d["content"]).split(maxsplit=5)[:5]) + "..."
                        }) for d in web_docs]

        if documents is not None:
            documents.extend(web_docs)
        else:
            documents = web_docs
            
        return {
            "documents": documents,
            "from_graph": LoggedAttribute(content="Web search results added")
                }

    async def clean_up(state: FlowState) -> Dict[str, Any]:

        """End node that resets state after processing"""
        
        return FlowState.reset(state, preserve_messages=False)

    # Define our edges 
    def decide_to_generate(state: FlowState) -> str:
        if state["web_search"]:
            return WEBSEARCH
        else:
            return GENERATE

    async def run_generation(state: FlowState) -> Dict[str, Any]:
        question = state["messages"][0].content
        documents = state["documents"]

        generation = await generation_chain.ainvoke(
            {"context": documents, "question": question})
        
        """ Here we add documents as instances of DocumentEntry class
        This class defines how our data would be saved in database and 
        how it will be rendered in UI """

        log_items = ["Answer generated based on documents:"]
        for doc in documents:
            log_items.append(DocumentEntry(
                title=doc.metadata.get('title', ''),
                page_content=doc.page_content,
                source=doc.metadata.get('source', '')))
        
        return {
            "messages": generation,
            "from_graph": LoggedAttribute(content=log_items)
            }

    # Define graph
    workflow = StateGraph(FlowState)
    workflow.add_node(RETRIEVE, run_retrieve)
    workflow.add_node(GRADE_DOCUMENTS, run_grade_documents)
    workflow.add_node(GENERATE, run_generation)
    workflow.add_node(WEBSEARCH, run_web_search)
    workflow.add_node(CLEANUP, clean_up)

    workflow.set_entry_point(RETRIEVE)
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    workflow.add_conditional_edges(
        source=GRADE_DOCUMENTS,
        path=decide_to_generate,
        # for optional extended communication between nodes
        path_map={
            WEBSEARCH: WEBSEARCH,
            GENERATE: GENERATE,
        },
    )

    workflow.add_edge(WEBSEARCH, GENERATE)
    workflow.add_edge(GENERATE, CLEANUP)
    workflow.add_edge(CLEANUP, END)
    memory = MemorySaver()

    app = workflow.compile(checkpointer=memory)
    
    # app.get_graph().draw_mermaid_png(
    #     output_file_path="corrective_rag_graph.png")
    
    return app


if __name__ == "__main__":

    import logging
    import tracemalloc
    import asyncio
    
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
                config="openai-rag-corrective",
                session_id=session_id))
    finally:
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logger.info("\nMemory traceback:")
        for stat in top_stats[:3]:
            logger.info(stat)