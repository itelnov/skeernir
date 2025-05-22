import os
import asyncio
import logging
from uuid import uuid4
import json
from asyncio.queues import QueueEmpty
from itertools import chain
from collections import defaultdict

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List, Dict, AsyncGenerator, Tuple
from fastapi.staticfiles import StaticFiles
from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    Form,
    Depends,
    status,
    )
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import (HTMLResponse, 
                               RedirectResponse, 
                               StreamingResponse, 
                               JSONResponse)
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage
from sqlalchemy import create_engine, desc, select, delete, update
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker
)
from dotenv import load_dotenv 
from langchain_core.messages import RemoveMessage
from langgraph.types import Command

import src.models as models
from src.graphs.utils import flatten_list
from src.text_utils import MarkdownConverter, clean_user_text
from src.entry import LoggedAttribute, get_entry_type_registry
from src.registry import GraphManager
from src.messages import (AsyncMessageStreamHandler,
                          AttachmentData,
                          MessageInput, 
                          AttachmentProcessingError,
                          StreamHandlerError)


# loading variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GM = GraphManager()
SMH = AsyncMessageStreamHandler()
ENTRY_TYPE_REGISTRY = get_entry_type_registry()

# Database setup
SQLALCHEMY_DATABASE_URL = os.environ.get(
    "SQLALCHEMY_DATABASE_URL", "sqlite+aiosqlite:///./chat.db")
async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=False,
    pool_pre_ping=True
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Example of executing VACUUM command properly
async def vacuum_database():
    async with AsyncSessionLocal() as session:
        try:
            await session.execute(text('VACUUM'))
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()

# Get the sync engine to create tables
async def init_models():
    async with async_engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)


asyncio.run(vacuum_database())
asyncio.run(init_models())


templates = Jinja2Templates(directory="templates")
converter = MarkdownConverter()
templates.env.filters["markdown_to_html"] = converter.convert
templates.env.filters["clean_user_text"] = clean_user_text


USER_TEMPLATE = templates.get_template("partials/user_message.html")
CHUNK_TEMPLATE = templates.get_template('partials/chunk_message.html')
BOT_TEMPLATE = templates.get_template('partials/bot_message.html')
CONV_TEMPLATE = templates.get_template('partials/conversation.html')
RC_TEMPLATE = templates.get_template('partials/right_container.html')
SYS_TEMPLATE = templates.get_template('partials/sys_message.html')
PP_WARNING_TEMPLATE = templates.get_template('partials/popup_warning.html')
CONVS_LIST = templates.get_template('conversations_lists.html')


def sys_message(message: str ="", message_type: str ="warning") -> str:
    sys_message_html = SYS_TEMPLATE.render(
        {"sys_message": message,
         "message_type": message_type})
    out =  json.dumps({"content": sys_message_html})
    return f"event: sys_warning\ndata: {out}\n\n"


def send_user_message(user_message: MessageInput) -> str:
    template = USER_TEMPLATE.render(user_message)
    data = json.dumps({"content": template})
    return f"event: user_message\ndata: {data}\n\n"


def send_chunk_template(uuid="") -> str:
    template = CHUNK_TEMPLATE.render({"uuid": uuid})
    data = json.dumps({"content": template})
    return  f"event: chunk_template\ndata: {data}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting up application...")
        yield
    finally:
        logger.info("Shutting down application...")
        await GM.terminate_all()  # Make this async if possible


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
security = HTTPBasic()


# Add session middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SECRET_KEY", "your-secret-key-here"),
    session_cookie="chat_session")


PORT = os.environ.get("SKEERNIR_PORT", "8899")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{PORT}"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"])


# Dependency to get database session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Optional[models.User]:
    user_id = request.session.get("user_id")
    if user_id:
        stmt = select(models.User).where(models.User.id == user_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    return None


# Message operations
async def get_conversation_messages(
    db: AsyncSession,
    session_id: str
) -> List[models.Message]:
    """
    Retrieve all messages for a specific conversation, sorted by timestamp.
    """
    stmt = (
        select(models.Message)
        .where(models.Message.session_uuid == session_id)
        .order_by(desc(models.Message.id))
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_conversation(
    db: AsyncSession,
    user: models.User,
    session_id: str,
) -> models.Conversation:
    stmt = (
        select(models.Conversation)
        .where(models.Conversation.session_uuid == session_id)
    )
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()

    # if not conv:
    #     conv = models.Conversation(
    #         session_uuid=session_id,
    #         user_id=user.id,
    #         graph_name=graph_name,
    #         title="new conversation ..."
    #     )
    #     db.add(conv)
    #     await db.commit()
    #     await db.refresh(conv)

    return conv


async def add_conversation(
    db: AsyncSession,
    user: models.User,
    session_id: str,
    graph_name: str
) -> models.Conversation:
    
    conv = models.Conversation(
        session_uuid=session_id,
        user_id=user.id,
        graph_name=graph_name,
        title="new conversation ..."
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)

    return conv



async def get_message_attachments(
    db: AsyncSession,
    message: models.Message
) -> List[models.Attachment]:
    stmt = (
        select(models.Attachment)
        .where(models.Attachment.message_uuid == message.message_uuid)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_graphlogs(
    db: AsyncSession,
    session_uuid: str,
) -> List[models.GraphLog]:
    stmt = (
        select(models.GraphLog)
        .where(models.GraphLog.session_uuid == session_uuid)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_conversations_with_earliest_messages(
    db: AsyncSession,
    user: models.User
) -> List[Dict[str, str]]:
    # Find all conversations for the user
    stmt = (
        select(models.Conversation)
        .where(models.Conversation.user == user)
    )
    result = await db.execute(stmt)
    user_conversations = result.scalars().all()

    result = []
    for conversation in user_conversations:
        # Find earliest message for each conversation
        stmt = (
            select(models.Message)
            .where(models.Message.session_uuid == conversation.session_uuid)
            .order_by(models.Message.timestamp)
            .limit(1)
        )
        msg_result = await db.execute(stmt)
        earliest_message = msg_result.scalar_one_or_none()

        if earliest_message:
            result.append({
                "session_id": conversation.session_uuid,
                "earliest_message": earliest_message.content,
                "graph_name": conversation.graph_name,
                "created_at": conversation.created_at,
            })

    return result


async def delete_conversation_and_related(
    db: AsyncSession,
    conv: models.Conversation
):
    # Delete related messages
    messages_stmt = delete(models.Message).where(
        models.Message.session_uuid == conv.session_uuid
    )
    await db.execute(messages_stmt)

    # Delete related attachments
    attachments_stmt = delete(models.Attachment).where(
        models.Attachment.session_uuid == conv.session_uuid
    )
    await db.execute(attachments_stmt)

    # Delete the conversation itself
    await db.delete(conv)
    await db.commit()


async def delete_message_with_attachments(
    db: AsyncSession,
    message_uuid: str
) -> bool:
    """
    Delete a message and all its attachments from the database.
    """
    try:
        stmt = select(models.Message).where(
            models.Message.message_uuid == message_uuid
        )
        result = await db.execute(stmt)
        message = result.scalar_one_or_none()

        if not message:
            return False

        await db.delete(message)
        await db.commit()
        return True

    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Error deleting message: {str(e)}")
        return False


async def update_con_interrupted_status(
    db: AsyncSession, 
    session_uuid: str, 
    interrupted: bool, 
    interrupted_value: str
) -> bool:
    """
    Static method to update the interrupted status and value for a conversation 
    by session_uuid.
    
    Args:
        db (AsyncSession): SQLAlchemy async database session
        session_uuid (str): The unique session UUID of the conversation
        interrupted (bool): New interrupted status
        interrupted_value (str): New interrupted value
        
    Returns:
        bool: True if update was successful, False if conversation not found
    """
    try:
        result = await db.execute(
            update(models.Conversation)
            .where(models.Conversation.session_uuid == session_uuid)
            .values(
                interrupted=interrupted,
                interrupted_value=interrupted_value
            )
        )
        
        await db.commit()
        
        return result.rowcount > 0
        
    except Exception as e:
        await db.rollback()
        raise e


async def get_interrupted_status(
    db: AsyncSession,
    session_uuid: str
) -> Optional[Tuple[bool, str]]:
    """
    Function to get the interrupted status and value for a 
    conversation by session_uuid.
    
    Args:
        db (AsyncSession): SQLAlchemy async database session
        session_uuid (str): The unique session UUID of the conversation
        
    Returns:
        Optional[Tuple[bool, str]]: Tuple of 
        (interrupted, interrupted_value) if found, None if not found
    """
    try:
        query = select(
            models.Conversation.interrupted, 
            models.Conversation.interrupted_value).where(
            models.Conversation.session_uuid == session_uuid
        )
        result = await db.execute(query)
        row = result.first()
        
        interrupted = bool(int(row[0]))
        interrupted_value = str(row[1]) if row[1] is not None else ""  
        return (interrupted, interrupted_value)
        
    except Exception as e:
        raise e


def create_message_record(
    message_uuid: str,
    message: str, 
    conv: models.Conversation,
    is_bot: bool,
):
    message_item = models.Message(
        message_uuid = message_uuid,
        session_uuid = conv.session_uuid,
        content = message,
        is_bot = is_bot)
    
    return message_item


def create_attachment_record(
    message_uuid: str,
    conv: models.Conversation,
    attch: AttachmentData,
):
    attachment_item = models.Attachment(
        message_uuid = message_uuid,
        session_uuid = conv.session_uuid,
        filename = attch.filename,
        mime_type = attch.type,
        file_size = attch.size,
        # Store file content
        file_content = attch.processed_content,
        attachemnt_metadata={
            'upload_timestamp': datetime.now(
                timezone.utc).isoformat()
        }
    )
    return attachment_item


def create_graphlog_record(
    conv: models.Conversation,
    item_node: str,
    item_type: str,
    item_content: str,
):
    graphlog_item = models.GraphLog(
        session_uuid = conv.session_uuid,
        item_node = item_node,
        item_type = item_type,
        item_content = item_content,
    )
    return graphlog_item


@app.get("/", response_class=RedirectResponse)
async def root(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    user = await get_current_user(request, db)
    if user:
        session_id = str(uuid4())
        try:
            GM.connect_session(session_id, GM.default_graph)
            redirect_url = request.url_for("chat", session_id=session_id)
            response = RedirectResponse(
                url=redirect_url, status_code=status.HTTP_302_FOUND)
            return response        
        
        except Exception as e:
            return RedirectResponse(
                url=(f"/login?error_message=Default grpah {GM.default_graph} "
                     f"was not compiled with Exception:\n\n{e}. \n"
                     f"Check {GM.default_graph}.json in config and graph "
                     f"definition in src/graphs"), 
                status_code=status.HTTP_302_FOUND)
        
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@app.get("/login", response_class=HTMLResponse)
async def get_login_form(
    request: Request,
    error_message: str | None = None,
    success_message: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    user = await get_current_user(request, db)
    if not user or error_message:
        return templates.TemplateResponse(request, "login.html", 
            {
                "error_message": error_message,
                "success_message": success_message
            }
        )
    responce = RedirectResponse(
        url="/", status_code=status.HTTP_302_FOUND)
    return responce


@app.post("/login", tags=["authentication"], response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    user_stmt = select(models.User).where(models.User.username == username)
    result = await db.execute(user_stmt)
    existing_user = result.scalar_one_or_none()
    if not models.User.verify_password(
        password, existing_user.hashed_password):
        
        return  templates.TemplateResponse(request, "login.html", 
        {   
            "error_message": "Invalid username or password",
            "success_message": None,
            "username": username
        })
    
    request.session["user_id"] = existing_user.id
    return RedirectResponse(url="/", status_code=303)


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)


@app.get("/register", response_class=HTMLResponse)
async def register_form(request: Request):
    return templates.TemplateResponse(request, "register.html",
        {
            "error_message": None,
        })


@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    # Validate passwords match
    if password != confirm_password:
        return templates.TemplateResponse(request, "register.html",
        {
            "error_message": "Passwords do not match",
            "username": username
        })
    
    # Check if username or email already exists
    username_stmt = select(models.User).where(models.User.username == username)
    existing_username = await db.execute(username_stmt)
    if existing_username.scalar_one_or_none():
        return templates.TemplateResponse(request, "register.html",
        {
            "error_message": "Username already taken", 
            "username": username
        })
    
    email_stmt = select(models.User).where(models.User.email == email)
    existing_email = await db.execute(email_stmt)
    if existing_email.scalar_one_or_none():
        return templates.TemplateResponse(request, "register.html",
        {
            "error_message": "Email already registered",
            "username": ""
        })
    
    # Create new user
    hashed_password = models.User.hash_password(password)
    user = models.User(
        username=username,
        email=email,
        hashed_password=hashed_password)
    db.add(user)
    await db.commit()
    
    return templates.TemplateResponse(request, "login.html", 
        {   
            "error_message": None,
            "success_message": "Registration was successful, please login",
            "username": username
        })


@app.delete('/remove-popup-warning', response_class=HTMLResponse)
async def remove_warning():
    return ""


@app.get('/remove_message/{uuid}', response_class=HTMLResponse)
async def delmess(
    request: Request,
    uuid: str,
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    redirect_url = request.url_for("chat", session_id=session_id)
    if not await delete_message_with_attachments(db, uuid):
        wm = f"Smth goes wrong, check logs. Conversation restored from database"
        redirect_url = redirect_url.include_query_params(
            warning_message=f"{wm}")

    """ Reset Graph memory """
    session = GM.get_session(session_id)
    config = {"configurable": {"thread_id": session_id}}
    graphai = session.graph.graph_call
    graphai.update_state(config, {"messages": []})    
    response = RedirectResponse(
            url=redirect_url, status_code=status.HTTP_302_FOUND)
    return response


@app.get("/newconv/{session_id}", response_class=RedirectResponse)
async def start_newconv(
    request: Request,
    session_id: str,
):  
    error = None
    try:
        session = GM.get_session(session_id)
        if session is None:
            session = GM.connect_session(session_id, GM.default_graph)
        else:
            graphai_name = session.graph.name
            GM.remove_session(session_id, with_graph=False)
            await SMH.delete_queue(session_id)
            session_id = str(uuid4())
            GM.connect_session(session_id, graphai_name)
    except Exception as error:
        GM.remove_session(session_id, with_graph=True)
        await SMH.delete_queue(session_id)
        logger.error(error)

    finally:
        redirect_url = request.url_for("chat", session_id=session_id)
        if error:
            redirect_url = redirect_url.include_query_params(
                warning_message=f"{e}")
        response = RedirectResponse(
            url=redirect_url, status_code=status.HTTP_302_FOUND)
        return response
    

@app.get("/loadmodel/{session_id}", response_class=RedirectResponse)
async def loadmodel(
    request: Request,
    session_id: str,
    selected_graph: Optional[str] = None
):  
    redirect_url = request.url_for("chat", session_id=session_id)
    try:
        GM.remove_session(session_id)
        await SMH.delete_queue(session_id)
        GM.connect_session(session_id, selected_graph)
    except Exception as e:
        GM.remove_session(session_id, with_graph=True)
        await SMH.delete_queue(session_id)
        GM.connect_session(session_id, GM.default_graph)
        wm = f"{e}" + "\nSee details in app logs, switch to the default graph"
        redirect_url = redirect_url.include_query_params(
            warning_message=f"{wm}")
        logger.error(e)
    finally:
        response = RedirectResponse(
            url=redirect_url, status_code=status.HTTP_302_FOUND)
        return response


@app.get("/main/{session_id}", response_class=HTMLResponse | RedirectResponse)
async def chat(
    request: Request,
    session_id: str,
    parent_session_id: str = "",
    warning_message: str = "",
    db: AsyncSession = Depends(get_db),
):
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse(
            url="/login", status_code=status.HTTP_302_FOUND)

    # restore user conversations
    user_convs = await get_conversations_with_earliest_messages(db, user)

    # [conv for conv in reversed(user_convs)]
    
    # Step 1: Group and sort entries
    grouped_data = defaultdict(list)
    # Grouping the items
    for entry in user_convs:
        grouped_data[entry['graph_name']].append(entry)
    
    prev_convs_html = ""
    for graph_name, entries in grouped_data.items():
        sorted_entries = sorted(entries, key=lambda x: x['created_at'])
        prev_convs_html += CONV_TEMPLATE.render({
            "sorted_entries": reversed(sorted_entries),
            "graph_name": graph_name,
            "parent_session_id": session_id
        })
    
    session = GM.get_session(session_id)
    
    if session is None:
        try:
            graphai_name = GM.default_graph
            if parent_session_id:
                parent_session = GM.get_session(parent_session_id)
                graphai_name = parent_session.graph.name
                GM.remove_session(parent_session_id, with_graph=False)
                await SMH.delete_queue(parent_session_id)
            session = GM.connect_session(session_id, graphai_name)
            
        except Exception as e:
            request.session.clear()
            return RedirectResponse(
                url=(f"/login?error_message=Smth goes wrong!\n"
                     f"Check detailes:\n\n{e}."),
                status_code=status.HTTP_302_FOUND)

    # restore current conversation history
    graph_logs_html = []
    right_container_html = ""

    if session.graph.entries_map:
        # The decision the right panel appear
        right_container_html = RC_TEMPLATE.render()

    prev_messages_html = ""
    messages = await get_conversation_messages(db, session_id=session_id)    
    
    graph_logs_records = await get_graphlogs(db, session_id)
    if graph_logs_records:
        # There are Graph Logs in conversation, lets check if 
        # current graph has templates to render logs.
        for item in graph_logs_records:
            entry = ENTRY_TYPE_REGISTRY[
                item.item_type].restore_from_record(item)
            graph_content = (
                f"<div>{item.item_node} - "
                f"{item.item_type}: {entry.content_to_send()}</div>")
            graph_logs_html.append(graph_content)    
    
    if messages:
        message_history = []
        for m in reversed(messages):
            if int(m.is_bot):
                message_history.append(AIMessage(content=m.content))
                prev_message = BOT_TEMPLATE.render(
                    {
                        "bot_message": m.content,
                        "uuid": m.message_uuid
                     })
            else:
                attachments = await get_message_attachments(db, m)
                user_message = MessageInput(
                    message=m.content, 
                    attachments=[
                        AttachmentData.restore_from_record(att) 
                        for att in attachments],
                    uuid=m.message_uuid)
                            
                prev_message = USER_TEMPLATE.render(user_message)
                message_history.append(
                    HumanMessage(content=flatten_list(user_message.content)))
            prev_messages_html += prev_message

        config = {"configurable": {"thread_id": session_id}}
        graphai = session.graph.graph_call
        graph_memory_messages = graphai.get_state(config).values.get(
            "messages", None)
        if graph_memory_messages:
            graphai.update_state(config, {"messages":[
                RemoveMessage(id=m.id) for m in graph_memory_messages]})
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # TODO For more advance graph flow it might be nessary to save and      #
        # be able to deserialize not only messages but at least the last state  # 
        # within the history of outputs (LogAttributes). Should be implemented  #
        # in future relises.                                                    #  
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        graphai.update_state(config, {"messages": message_history})

    if graph_logs_html and session.graph.entries_map:
        right_container_html = RC_TEMPLATE.render(graphlogs=graph_logs_html)

    if warning_message:
        warning_message = PP_WARNING_TEMPLATE.render(
            warning_message=warning_message)

    return templates.TemplateResponse(request, "main.html",
        {   
            "warning_message": warning_message,
            "graph": f'{session.graph.name}: {session.graph.tag}',
            "user": user.username,
            "previous_messages": prev_messages_html,
            "conversation_list": prev_convs_html,
            "session_id": session_id,
            "right_container": right_container_html
        })


@app.get("/delconv/{session_id}", response_class=RedirectResponse)
async def delconv(
    request: Request,
    session_id: str,
    parent_session_id: str,
    db: AsyncSession = Depends(get_db),
):
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse(
            url="/login", status_code=status.HTTP_302_FOUND)
    try:
        conv = await get_conversation(db, user, session_id)
        if conv:
            await delete_conversation_and_related(db, conv)
        e = f"Conversation {conv.session_uuid} was deleted from database"
        if session_id == parent_session_id:
            GM.remove_session(parent_session_id, with_graph=True)
            await SMH.delete_queue(parent_session_id)
            session_id = str(uuid4())
        else:
            session_id = parent_session_id
    except Exception as e:
        logger.error(e)
    finally:
        redirect_url = request.url_for("get_convs_list", session_id=session_id)
        response = RedirectResponse(
            url=redirect_url, status_code=status.HTTP_302_FOUND)
        return response


@app.get("/get_convs_lists/{session_id}", response_class=HTMLResponse)
async def get_convs_list(
    request: Request,
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse(
            url="/login", status_code=status.HTTP_302_FOUND)
    prev_convs_html = ''
    try:
        # restore user conversations
        user_convs = await get_conversations_with_earliest_messages(db, user)    
        grouped_data = defaultdict(list)
        # Grouping the items
        for entry in user_convs:
            grouped_data[entry['graph_name']].append(entry)
        
        prev_convs_html = ""
        for graph_name, entries in grouped_data.items():
            sorted_entries = sorted(entries, key=lambda x: x['created_at'])
            prev_convs_html += CONV_TEMPLATE.render({
                "sorted_entries": reversed(sorted_entries),
                "graph_name": graph_name,
                "parent_session_id": session_id
            })

    except Exception as e:
        logger.error(e)
    finally:
        return templates.TemplateResponse(request, "conversations_lists.html",
            {
                "conversation_list": prev_convs_html,
            },
        )


@app.get("/select_graph/{session_id}", response_class=HTMLResponse)
async def get_graphs_list(
    request: Request,
    session_id: str,
):  
    items = [
        {"name": n, "tag": t} for n, t in GM._graphs_registry.list_graphs()]

    return templates.TemplateResponse(request, "dropdown_items.html",
        {
            "session_id": session_id,
            "items": items
        },
    )


@app.get("/user_settings/{session_id}", response_class=HTMLResponse)
async def get_user_details(
    request: Request,
    session_id: str,
    db: AsyncSession = Depends(get_db),
):  
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    return templates.TemplateResponse(request, "user_details.html",
        {
            "user": {"name": user.username, "email": user.email},
            "session_id": session_id,
         }
    )


@app.delete("/user_settings/{session_id}", response_class=HTMLResponse)
async def get_user_details(
    request: Request,
    session_id: str,
    db: AsyncSession = Depends(get_db),
):  
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    return templates.TemplateResponse(request, "user_details.html",
        {
            "user": {"name": user.username, "email": user.email},
            "session_id": session_id,
         }
    )


@app.post("/sendmessage/{session_id}", response_class=JSONResponse)
async def send_input_message(
    request: Request,
    session_id: str,
    message: str = Form(...),
    file: UploadFile | List[UploadFile] | None = None,
    db: AsyncSession = Depends(get_db),
):
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    session = GM.get_session(session_id)
    attachments = []
    if file:
        files = [file] if isinstance(file, UploadFile) else file
        for upload_file in files:
            content = await upload_file.read()
            attachments.append(AttachmentData(
                type=upload_file.content_type,
                filename=upload_file.filename,
                content=content,
                valid_modalities=session.graph.att_modals))

    await SMH.put_message(session_id, MessageInput(
        message=message, attachments=attachments))
    return JSONResponse({"status": "processing_started"})


@app.post("/hardstopstream/{session_id}", response_class=JSONResponse)
async def send_stop_message(
    session_id: str,
):
    await SMH.put_interrupt_flag(session_id)
    return JSONResponse({"status": "processing_stop"})


async def check_stop_signal(queue, session_id):
    try:
        red_flag = queue[session_id].get_nowait()
        return red_flag.get("stop", False)
    except QueueEmpty:
        return False


@app.get("/stream/{session_id}", response_class=StreamingResponse)
async def stream_endpoint(
    request: Request,
    session_id: str,
    db: AsyncSession = Depends(get_db),
    ):
        
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    
    session = GM.get_session(session_id)
    graphai = session.graph.graph_call
    conv = await get_conversation(db, user, session_id)
    if not conv:
        conv = await add_conversation(db, user, session_id, session.graph.name)
    async def stream_generator(session_id):
        """ Direct stream data for a specific session """
        try:
            # Start a transactions
            user_message = await SMH.get_queue(session_id)
            user_message.uuid = str(uuid4())
        
            db.add(create_message_record(
                user_message.uuid,
                user_message.message,
                conv, 
                is_bot=False))
        
            input_message = HumanMessage(content=[
                {"type": "text", "text": user_message.message}])
            
            if user_message.attachments:

                # Add all attachments in proper format
                input_message.content.extend(
                    chain.from_iterable(
                        attch.formatted_content 
                        for attch in user_message.attachments
                    )
                )
                
                # Create and add attachment records to db
                db.add_all([
                    create_attachment_record(
                        message_uuid=user_message.uuid,
                        conv=conv,
                        attch=attch
                    ) for attch in user_message.attachments
                ])

            # send user message as html keeping original formatting
            yield send_user_message(user_message)
            await db.commit()

            user_convs = await get_conversations_with_earliest_messages(
                db, user)

            grouped_data = defaultdict(list)
            # Grouping the items
            for entry in user_convs:
                grouped_data[entry['graph_name']].append(entry)
            prev_convs_html = ""
            for graph_name, entries in grouped_data.items():
                sorted_entries = sorted(entries, key=lambda x: x['created_at'])
                prev_convs_html += CONV_TEMPLATE.render({
                    "sorted_entries": reversed(sorted_entries),
                    "graph_name": graph_name,
                    "parent_session_id": session_id
                })

            conv_list_html = CONVS_LIST.render(
                {"conversation_list": prev_convs_html})
            out =  json.dumps({"content": conv_list_html})
            yield f"event: conv_list_update\ndata: {out}\n\n"

            config = {"configurable": {"thread_id": session_id}}
        
            # nex_node will not be empty if we conitunue graph execution after 
            # it was interrupt
            was, val = await get_interrupted_status(
                db=db,
                session_uuid=session_id
            )
            if was:
                logger.info((
                    f"Graph {session.graph.name} on value {val}: "
                    "user was required"))
                stream_in = Command(resume={val: input_message})
                await update_con_interrupted_status(
                    db=db,
                    session_uuid=session_id,
                    interrupted=False,
                    interrupted_value=""
                    )
            else:
                stream_in = {"messages": [input_message]}
            
            new_stream_message = True
            async for chunk_type, graph_state in graphai.astream(
                stream_in,
                config=config,
                stream_mode=["messages", "updates"]):
                
                # Check for stop signal again after each chunk processing
                if await check_stop_signal(SMH.interrupt_queues, session_id):
                    raise StreamHandlerError("User stopped streaming!")
                
                match chunk_type:
                    case "messages":
                        chunk, graph_metas = graph_state
                        if isinstance(chunk, AIMessageChunk):
                            if new_stream_message:
                                yield send_chunk_template(chunk.id)
                                new_stream_message = False
                            
                            if chunk.content:
                                data = json.dumps({"content": chunk.content})
                                yield f"data: {data}\n\n"
                        else:
                            pass
                
                    case "updates":
                        for node, node_updates in graph_state.items():
                            if node == "__interrupt__":
                                await update_con_interrupted_status(
                                    db=db,
                                    session_uuid=session_id,
                                    interrupted=True,
                                    interrupted_value=node_updates[0].value
                                    )
                                continue
                            if node_updates:
                                for state_attr, val in node_updates.items():
                                    if isinstance(val, LoggedAttribute):
                                        for item in val:
                                            db.add(create_graphlog_record(
                                                conv=conv,
                                                item_node=node,
                                                item_type=item.type,
                                                item_content=item.content_to_store()
                                            ))
                                            data = json.dumps(
                                                {
                                                    "State": f'{node}-{state_attr}-{item.type}',
                                                    "content": item.content_to_send()
                                                })
                                            yield f"event: agent_log\ndata: {data}\n\n"
                                    if state_attr == "messages":
                                        if isinstance(val, AIMessage):
                                            val = [val]
                                        for el in val:
                                            if isinstance(el, AIMessage):
                                                bot_message_item = create_message_record(
                                                    message_uuid=el.id,
                                                    message=el.content, 
                                                    conv=conv, 
                                                    is_bot=True)
                                                db.add(bot_message_item)
                                                new_stream_message = True
                                await db.commit()
                    
        except StreamHandlerError as e:
            await db.rollback()
            yield sys_message(
                message=(f"{e}"),
                message_type="warning")

        except AttachmentProcessingError as e:
            await db.rollback()
            yield sys_message(
                message=(
                    "File extension in the attachment is not supported. "
                    f"Details:\n\n {e}"),
                message_type="warning")

        except Exception as e:
            # Clean up stream when cancelled
            await db.rollback()
            yield sys_message(
                message=(
                    "System: Shit happens. Nobody ever promised you a "
                    f"smooth ride. Deal with it and move on. This "
                    f"conversation is finished. Additional info:\n\n {e}"),
                message_type="error")
            
        finally:          
                yield f"event: streamend\ndata: \n\n"

    return StreamingResponse(
        stream_generator(session_id),
        media_type="text/event-stream"
        )


if __name__ == '__main__':

    import uvicorn
    import tracemalloc
    tracemalloc.start()
    try:
        uvicorn.run(
            app, 
            host="localhost", 
            port=int(PORT),
            loop="asyncio",
            log_level="info",
            access_log=True,
            )

    except Exception as e:
        logger.info(f"An error occurred: {e}")

    finally:
        # Always perform cleanup
        logger.info("Cleanup completed. Exiting program.")
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logger.info("\nMemory traceback:")
        for stat in top_stats[:3]:
            logger.info(stat)