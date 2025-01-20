import io
import requests
import base64
import asyncio
from typing import Dict, List, Optional, Any, ClassVar, ClassVar, Union
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel, computed_field
from markitdown import MarkItDown
from fastapi.templating import Jinja2Templates
import json

from src import models
from src.registry import ModalsType

# Custom exceptions
class AttachmentProcessingError(Exception):
    """Raised when there's an error processing attachments"""
    pass


class BaseProcessor(BaseModel, ABC):
    
    @abstractmethod
    def process_content(self, attach: Any) -> Any:
        """Process the input data according to specific rules."""
        pass
    
    @abstractmethod
    def format_content(self, attach: Any) -> Any:
        """ returns attachment data in vaild format for modality """
        pass


class AttachmentData(BaseModel):

    filename: str
    type: str
    content: str | bytes
    content_processed: Optional[str] = None
    content_size: Optional[int] = None
    processor_map: ClassVar[Dict[str, BaseProcessor]] = {}
    valid_modalities: List[ModalsType] = ['text']

    @computed_field
    @property
    def modality(self) -> str:
        return self.type.split('/')[0]

    @computed_field
    @property
    def size(self) -> int:
        if not self.content_size:
            self.content_size = len(self.content)
        return self.content_size

    @computed_field
    @property
    def processed_content(self) -> str:
        if self.content_processed is None:
            if self.modality in ModalsType._value2member_map_ and self.modality not in self.valid_modalities:
                raise AttachmentProcessingError((
                    f"Error processing for modality type: {self.modality}.\n"
                    "It seems modality is known but not valid for this Graph"))

            processor = self.processor_map.get(self.modality, TextProcessor())
            self.content_processed = processor.process_content(self)
        
        return self.content_processed
    
    @property
    def formatted_content(self) -> str:
        processor = self.processor_map.get(self.modality,  TextProcessor())
        return processor.format_content(self)

    @classmethod
    def register_processor(cls, modality: str):
        def decorator(processor):
            cls.processor_map[modality] = processor()
            return processor
        return decorator
    
    @classmethod
    def restore_from_record(cls, attach: models.Attachment):
        return cls(
            filename=attach.filename,
            type=attach.mime_type,
            content="not defined",
            content_processed=attach.file_content,
            content_size=attach.file_size)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@AttachmentData.register_processor('text')
class TextProcessor(BaseProcessor):

    @classmethod
    def process_content(cls, attach: AttachmentData)-> str:
        
        """Process text attachments"""
        try:
            markitdown = MarkItDown()
            response = requests.Response()
            response._content = attach.content
            response.raw = io.BytesIO(attach.content)
            response.status_code = 200
            response.headers = {
                'Content-Type': attach.type,
                'Content-Length': str(len(attach.content)),
                'Content-Disposition': f'attachment; filename="{attach.filename}"'
            }
            converted = markitdown.convert(response)
            if converted is None:
                raise AttachmentProcessingError(
                    f"Text conversion failed for {attach.filename}")
            return converted.text_content
        except Exception as e:
            raise AttachmentProcessingError(
                f"Error processing text for {attach.filename}: \n\n{str(e)}")

    @classmethod
    def format_content(cls, attach: AttachmentData)-> Dict[str, str]:
        return {"type": "text", "text": attach.processed_content}


@AttachmentData.register_processor('image')
class ImageProcessor(BaseProcessor):

    @classmethod
    def process_content(cls, attach: AttachmentData) -> str:
        """Process image attachments"""
        try:
            processed_content = base64.b64encode(attach.content).decode('utf-8')
            return processed_content
        except Exception as e:
            raise AttachmentProcessingError(
                f"Error processing image for {str(attach.filename)} :\n\n {e}")

    @classmethod
    def format_content(cls, attach: AttachmentData)-> Dict[str, str | Dict[str, str]]:
        return  {
            "type": "image_url",
            "image_url": {
                "url": f"data:{attach.type};base64,{attach.processed_content}"
            }
        }   


class MessageInput(BaseModel):

    """ Handler for inputs from Front-end """
    
    message: str
    attachments: Optional[List[AttachmentData]] = None

    @property
    def content(self):
        content = [{"type": "text", "text": self.message}]
        for attch in self.attachments:
            content.append(attch.formatted_content) 
        return content


class AsyncMessageStreamHandler:
    
    def __init__(self):
        # Dictionary of queues keyed by ID
        self.message_queues = defaultdict(asyncio.Queue)

    async def put_message(self, id: str, message: MessageInput):
        """
        Adds a message to the queue associated with the given ID.
        
        :param id: The identifier for the queue
        :param message: The message to put into the queue
        """
        await self.message_queues[id].put(message)

    async def get_queue(self, id)-> MessageInput:
        """
        Retrieves the next message from the queue associated with the given ID.
        Waits if the queue is empty.
        
        :param id: The identifier for the queue
        :return: The next message from the queue
        """
        return await self.message_queues[id].get()

    async def delete_queue(self, id: str):
        """
        Safely deletes the message queue associated with the given ID.
        
        :param id: The identifier for the queue to be deleted
        """
        if id in self.message_queues:
            # Get the queue object
            queue = self.message_queues[id]
            
            # Clear any remaining messages
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
            # Remove the queue from the dictionary
            del self.message_queues[id]



class MessageOutputType(Enum):
    USER = "user"
    BOT = "bot"
    CHUNK = "chunk"
    SYSTEM = "sys"
    CONVERSATION = "conv"
    RIGHT_CONTAINER = "right_container"


class TemplateConfig(BaseModel):
    template_name: str
    event_name: str
    
    # Add model configuration
    model_config = {
        "frozen": True,  # Makes instances immutable
        "extra": "forbid"  # Prevents additional attributes
    }


class TemplateRegistry:
    def __init__(self, template_dir: str = "templates"):
        self.templates = Jinja2Templates(directory=template_dir)
        
        # Configure template mappings
        self._template_configs = {
            MessageOutputType.USER: TemplateConfig(
                template_name="partials/user_message.html", 
                event_name="user_template"),
            MessageOutputType.BOT: TemplateConfig(
                template_name="partials/bot_message.html", 
                event_name="bot_template"),
            MessageOutputType.CHUNK: TemplateConfig(
                template_name="partials/chunk_message.html", 
                event_name="chunk_template"),
            MessageOutputType.SYSTEM: TemplateConfig(
                template_name="partials/sys_message.html", 
                event_name="system_template"),
            MessageOutputType.CONVERSATION: TemplateConfig(
                template_name="partials/conversation.html", 
                event_name="conversation_template"),
            MessageOutputType.RIGHT_CONTAINER: TemplateConfig(
                template_name="partials/right_container.html", 
                event_name="right_container_template")
        }
        
        # Load all templates at initialization
        self._message_templates = {
            msg_type: self.templates.get_template(config.template_name)
            for msg_type, config in self._template_configs.items()
        }
    
    def render_and_format(
        self, message_type: MessageOutputType, 
        context: Dict[str, Any]
    ) -> str:
        """
        Renders the template with given context and formats it as an SSE event.
        
        Args:
            message_type: Type of message to render
            context: Dictionary of variables to pass to the template
            
        Returns:
            Formatted SSE event string
        """
        template = self._message_templates[message_type]
        config = self._template_configs[message_type]
        
        rendered_content = template.render(**context)
        data = json.dumps({"content": rendered_content})
        
        return f"event: {config.event_name}\ndata: {data}\n\n"
    
    def render_only(
        self, message_type: MessageOutputType, 
        context: Dict[str, Any]
    ) -> str:
        """
        Just renders the template without SSE formatting.
        
        Args:
            message_type: Type of message to render
            context: Dictionary of variables to pass to the template
            
        Returns:
            Rendered template string
        """
        return self._message_templates[message_type].render(**context)