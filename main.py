import os
import logging
from uuid import uuid4
import json

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List, Dict, Union, Tuple, Literal, Generator
from fastapi.staticfiles import StaticFiles
from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    Form,
    Depends,
    status
    )
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import text
from dotenv import load_dotenv 

import src.models as models
from src.text_utils import MarkdownConverter
from src.entry import LoggedAttribute, get_entry_type_registry
from src.registry import GraphManager
from src.messages import (AsyncMessageStreamHandler, 
                          AttachmentData,
                          MessageInput, 
                          AttachmentProcessingError)


# loading variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GM = GraphManager()
SMH = AsyncMessageStreamHandler()
ENTRY_TYPE_REGISTRY = get_entry_type_registry()

# Database setup
SQLALCHEMY_DATABASE_URL = os.environ["SQLALCHEMY_DATABASE_URL"]
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Execute the VACUUM command
db = SessionLocal()
db.execute(text('VACUUM'))
db.commit()
models.Base.metadata.create_all(bind=engine)


templates = Jinja2Templates(directory="templates")
converter = MarkdownConverter()
templates.env.filters["markdown_to_html"] = converter.convert

USER_TEMPLATE = templates.get_template("partials/user_message.html")
CHUNK_TEMPLATE = templates.get_template('partials/chunk_message.html')
BOT_TEMPLATE = templates.get_template('partials/bot_message.html')
CONV_TEMPLATE = templates.get_template('partials/conversation.html')
RC_TEMPLATE = templates.get_template('partials/right_container.html')
SYS_TEMPLATE = templates.get_template('partials/sys_message.html')


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


def send_chunk_template() -> str:
    template = CHUNK_TEMPLATE.render()
    data = json.dumps({"content": template})
    return  f"event: chunk_template\ndata: {data}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up application...")
    yield
    # Shutdown
    logger.info("Shutting down application...")
    GM.terminate_all()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
security = HTTPBasic()


# Add session middleware
app.add_middleware(
    SessionMiddleware,
    secret_key="your-secret-key-here",
    session_cookie="chat_session"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{os.environ['SKEERNIR_PORT']}"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
        request: Request, 
        db: Session = Depends(get_db)
    ) -> Optional[models.User]:
    user_id = request.session.get("user_id")
    if user_id:
        return db.query(models.User).filter(models.User.id == user_id).first()
    return None


def get_conversation_messages(
        db: Session, 
        session_id: str) -> List[models.Message]:
    """
    Retrieve all messages for a specific conversation, sorted by timestamp in 
    descending order.
    
    Args:
        db (Session): SQLAlchemy database session
        session_id (str): ID of the conversation to fetch messages for
        
    Returns:
        List[models.Message]: List of messages sorted by timestamp 
        (newest first)
    """
    
    messages = (
        db.query(models.Message)
            .filter(models.Message.session_uuid == session_id)
            .order_by(desc(models.Message.id))
            .all()
    )
    
    return messages


def get_conversation(
    db: Session,
    user: models.User,
    session_id: str,
):
    conv = db.query(models.Conversation)\
    .filter(models.Conversation.session_uuid == session_id)\
    .first()
    if not conv:
        _conv = models.Conversation(
            session_uuid = session_id,
            user_id = user.id,
            title = "new conversation ..."
        )
        db.add(_conv)
        db.commit()
        conv = db.query(models.Conversation)\
            .filter(models.Conversation.session_uuid == session_id)\
            .first()
    return conv


def get_message_attachments(    
    db: Session, 
    message: models.Message
    )-> List[models.Attachment]:

    message_attachments = (
        db.query(models.Attachment)
        .filter(models.Attachment.message_uuid == message.message_uuid)
        .all()
    )
    return message_attachments


def get_message_graphlogs(    
    db: Session, 
    message: models.Message
    )-> List[models.GraphLog]:

    message_graphlogs = (
        db.query(models.GraphLog)
        .filter(models.GraphLog.message_uuid == message.message_uuid)
        .all()
    )
    return message_graphlogs


def get_conversations_with_earliest_messages(
    db: Session, 
    user: models.User
    )-> Dict[str, str]:
    
    # Step 1: Find all conversations bounded with the given user
    user_conversations = (
        db.query(models.Conversation)
          .filter(models.Conversation.user == user)
          .all()
    )
        
    result = []
    # Step 2: For each user conversation, find the earliest message
    for conversation in user_conversations:
        earliest_message = (
            db.query(models.Message)
              .filter(models.Message.session_uuid == conversation.session_uuid)
              .order_by(models.Message.timestamp)
              .first()
        )
        # Step 3: Add the session ID and earliest message to the result
        if earliest_message:
            result.append({
                "session_id": conversation.session_uuid,
                "earliest_message": earliest_message.content,
            })
    
    return result


def delete_conversation_and_related(db: Session, conv: models.Conversation):

    # Delete related messages
    db.query(models.Message)\
        .filter(models.Message.session_uuid == conv.session_uuid)\
        .delete(synchronize_session=False)

    # Delete related attachments
    db.query(models.Attachment)\
        .filter(models.Attachment.session_uuid == conv.session_uuid)\
        .delete(synchronize_session=False)

    # Finally, delete the conversation itself
    db.delete(conv)
    db.commit()


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
    message_uuid: str,
    conv: models.Conversation,
    item_node: str,
    item_type: str,
    item_content: str,
):
    graphlog_item = models.GraphLog(
        message_uuid = message_uuid,
        session_uuid = conv.session_uuid,
        item_node = item_node,
        item_type = item_type,
        item_content = item_content,
    )

    return graphlog_item


@app.get("/", response_class=RedirectResponse)
async def root(
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if user:
        session_id = str(uuid4())

        GM.connect_session(session_id, GM.default_graph)
        
        redirect_url = request.url_for("chat", session_id=session_id)
        response = RedirectResponse(
            url=redirect_url, status_code=status.HTTP_302_FOUND)
        return response

    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@app.get("/login", response_class=HTMLResponse)
async def get_login_form(
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return templates.TemplateResponse(request, "login.html", 
            {
                "error_message": None,
                "success_message": None
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
    db: Session = Depends(get_db)
):
    user = db.query(models.User)\
        .filter(models.User.username == username)\
        .first()
    
    if not user or not models.User.verify_password(
        password, user.hashed_password):
        
        return  templates.TemplateResponse(request, "login.html", 
        {   
            "error_message": "Invalid username or password",
            "success_message": None,
            "username": username
        })
    
    request.session["user_id"] = user.id
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
    db: Session = Depends(get_db)
):
    # Validate passwords match
    if password != confirm_password:
        return templates.TemplateResponse(request, "register.html",
        {
            "error_message": "Passwords do not match",
            "username": username

        })
    # Check if username or email already exists
    if db.query(models.User).filter(models.User.username == username).first():
        return templates.TemplateResponse(request, "register.html",
        {
            "error_message": "Username already taken", 
            "username": username

        })
    
    if db.query(models.User).filter(models.User.email == email).first():
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
    db.commit()
    
    return templates.TemplateResponse(request, "login.html", 
        {   
            "error_message": None,
            "success_message": "Registration was successful, please login",
            "username": username
        })


@app.get("/newconv/{session_id}", response_class=RedirectResponse)
async def start_newconv(
    request: Request,
    session_id: str,
    db: Session = Depends(get_db),
):  
    user = get_current_user(request, db)
    if user:
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
        except Exception as e:
            logger.error(e)
    
        finally:
            redirect_url = request.url_for("chat", session_id=session_id)
            response = RedirectResponse(
                url=redirect_url, status_code=status.HTTP_302_FOUND)
            return response
    
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@app.get("/loadmodel/{session_id}", response_class=RedirectResponse)
async def loadmodel(
    request: Request,
    session_id: str,
    selected_graph: Optional[str] = None
):  
    try:
        GM.remove_session(session_id)
        await SMH.delete_queue(session_id)
        GM.connect_session(session_id, selected_graph)
    except Exception as e:
        logger.error(e)

    redirect_url = request.url_for("chat", session_id=session_id)
    response = RedirectResponse(
        url=redirect_url, status_code=status.HTTP_302_FOUND)
    return response


@app.get("/main/{session_id}", response_class=HTMLResponse)
async def chat(
    request: Request,
    session_id: str,
    parent_session_id: str = "",
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(
            url="/login", status_code=status.HTTP_302_FOUND)
    
    # restore user conversations
    user_convs = get_conversations_with_earliest_messages(db, user)

    prev_convs_html = ''.join(
        CONV_TEMPLATE.render(
            session_id=conv['session_id'],
            conv_tag=str(conv['earliest_message']),
            parent_session_id=session_id
        )
        for conv in reversed(user_convs)
    ) if user_convs else ''

    session = GM.get_session(session_id)
    if session is None:
        # The case when session is restored from conversation history and is not 
        # available in GraphRegistry memory. In this case we can't garantee the 
        # graph used for that conversation is now available (as for now the name 
        # of the graph for particular conversation is not stored either). To 
        # handle this event, we restore session with a default graph and let 
        # user to choose the graph from available list. The parent session will
        # be removed.
        GM.remove_session(parent_session_id)
        await SMH.delete_queue(parent_session_id)
        session = GM.connect_session(session_id, GM.default_graph)
    
    # restore conversation
    graph_logs_html = []
    right_container_html = ""

    if session.graph.entries_map:
        # The decision the right panel appear
        right_container_html = RC_TEMPLATE.render()

    prev_messages_html = ""
    messages = get_conversation_messages(db, session_id=session_id)    
    if messages:
        message_history = []
        for m in reversed(messages):
            if int(m.is_bot):
                message_history.append(AIMessage(content=m.content))
                prev_message = BOT_TEMPLATE.render({"bot_message": m.content})
                graph_logs_records = get_message_graphlogs(db, m)
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
            else:
                attachments = get_message_attachments(db, m)
                user_message = MessageInput(
                    message=m.content, 
                    attachments=[
                        AttachmentData.restore_from_record(att) 
                        for att in attachments])
                            
                prev_message = USER_TEMPLATE.render(user_message)
                message_history.append(
                    HumanMessage(content=user_message.content))
            
            prev_messages_html += prev_message

        config = {"configurable": {"thread_id": session_id}}
        graphai = session.graph.graph_call
        graphai.update_state(config, {"messages": message_history})


    if graph_logs_html and session.graph.entries_map:
        right_container_html = RC_TEMPLATE.render(graphlogs=graph_logs_html)


    return templates.TemplateResponse(request, "main.html",
        {   
            "graph": f'{session.graph.name}: {session.graph.tag}',
            "user": user.username,
            "previous_messages": prev_messages_html,
            "conversation_list": prev_convs_html,
            "session_id": session_id,
            "right_container": right_container_html
        })


@app.get("/delconv/{session_id}", response_class=JSONResponse)
async def delconv(
    request: Request,
    session_id: str,
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(
            url="/login", status_code=status.HTTP_302_FOUND)
    conv = get_conversation(db, user, session_id)
    delete_conversation_and_related(db, conv)

    return {"status": "deleted"}


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
    db: Session = Depends(get_db)
):  
    user = get_current_user(request, db)
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
    db: Session = Depends(get_db)
):  
    user = get_current_user(request, db)
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
    db: Session = Depends(get_db)
):
    user = get_current_user(request, db)
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


def process_chunk(
    db: Session,
    message_uuid: str,
    conv: models.Conversation,
    chunk_type: Literal["messages", "values"],
    stream_data: Union[Tuple, Dict],
)-> Generator:
    
    match chunk_type:
        case "messages":
            chunk, _ = stream_data
            if isinstance(chunk, AIMessageChunk):
                if chunk.content:
                    data = json.dumps({"content": chunk.content})
                    yield (f"data: {data}\n\n", chunk.content)
    
        case "values":
            for node_name, logged_attr in stream_data.items():
                if isinstance(logged_attr, LoggedAttribute):
                    for item in logged_attr:
                                                                        
                        db.add(create_graphlog_record(
                            message_uuid=message_uuid,
                            conv=conv,
                            item_node=node_name,
                            item_type=item.type,
                            item_content=item.content_to_store()
                        ))
                        data = json.dumps(
                            {
                                "Node": f'{node_name} - {item.type}',
                                "content": item.content_to_send()
                            })
                        yield (f"event: agent_log\ndata: {data}\n\n", "")


@app.get("/stream/{session_id}", response_class=StreamingResponse)
async def stream_endpoint(
    request: Request,
    session_id: str,
    db: Session = Depends(get_db)
    ):
        
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    conv = get_conversation(db, user, session_id)

    session = GM.get_session(session_id)    
    graphai = session.graph.graph_call
    
    async def stream_generator(session_id):
        
        """ Direct stream data for a specific session """
        try:
            user_message = await SMH.get_queue(session_id)
            user_message_uuid = str(uuid4())
            
            db.add(create_message_record(
                user_message_uuid,
                user_message.message,
                conv, 
                is_bot=False))
    
            input_message = HumanMessage(content=[
                {"type": "text", "text": user_message.message}])
            
            if user_message.attachments:
                # history of messages
                input_message.content.extend(
                    [attch.formatted_content for attch in 
                     user_message.attachments])
                
                # Create and add attachment records to db
                db.add_all([
                    create_attachment_record(
                        message_uuid=user_message_uuid,
                        conv=conv,
                        attch=attch
                    ) for attch in user_message.attachments
                ])
            db.commit()

            # send user message as html keeping original formatting
            yield send_user_message(user_message)
            
            # Here we send only html template where the content will be 
            # streamed
            yield send_chunk_template()

            config = {"configurable": {"thread_id": session_id}}
            bot_message = ''
            bot_message_uuid = str(uuid4())

            async for chunk_type, stream_data in graphai.astream(
                {"messages": [input_message]}, 
                config=config,
                stream_mode=["messages", "values"]):
                
                for out in process_chunk(
                    db=db,
                    message_uuid=bot_message_uuid,
                    conv=conv,
                    chunk_type=chunk_type,
                    stream_data=stream_data):

                    to_post, bot_message_chunk = out
                    bot_message += bot_message_chunk

                    yield to_post
            
            bot_message_item = create_message_record(
                message_uuid=bot_message_uuid,
                message=bot_message, 
                conv=conv, 
                is_bot=True)
            db.add(bot_message_item)
            db.commit()
    
        except AttachmentProcessingError as e:
            yield sys_message(
                message=(
                    "File extension in the attachment is not supported. "
                    f"Details:\n\n {e}"),
                message_type="warning")

        except Exception as e:
            # Clean up stream when cancelled
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
            port=int(os.environ["SKEERNIR_PORT"]),
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
    