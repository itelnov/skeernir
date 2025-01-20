from abc import ABC, abstractmethod
from typing import Dict, List, Union, Iterator, Type
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from src import models


class BaseEntry(BaseModel, ABC):

    type: str

    @classmethod
    @abstractmethod
    def restore_from_record(cls):
        pass

    @abstractmethod
    def content_to_send(self) -> str:
        """ Send prepareed content to frontend"""
        pass
    
    @abstractmethod
    def content_to_store(self) -> str:
        """ Send prepareed content to store in database """
        pass


class LogEntry(BaseEntry):

    type: str = Field(default="log")
    content: str

    def content_to_send(self) -> str:
        return self.content

    def content_to_store(self) -> str:
        return self.content

    @classmethod
    def restore_from_record(cls, datamodel: models.GraphLog)-> BaseEntry:
        return cls(content=datamodel.item_content)


class LoggedAttribute(BaseModel):
    """Pydantic model for attributes that should be logged."""
    content: Annotated[
        Union[str, List[Union[str, BaseEntry]], BaseEntry],
        Field(description=("Content to be logged - can be string, BaseEntry,"
                           " or list of these types"))
    ]
    log_entries: List[LogEntry] = Field(default_factory=list, exclude=True)

    def model_post_init(self, __context) -> None:
        """Initialize log entries after model validation."""
        self.log_entries = self._process_content()

    def _process_content(self) -> List[LogEntry]:
        """Process content into LogEntry objects."""
        if isinstance(self.content, str):
            return [LogEntry(content=self.content)]
        
        if isinstance(self.content, BaseEntry):
            return [self.content]
        
        if isinstance(self.content, list):
            return [
                item if isinstance(item, BaseEntry)
                else LogEntry(content=item) if isinstance(item, str)
                else LogEntry(content=str(item))  # Handle dict case
                for item in self.content
            ]

        raise ValueError("Content must be str, BaseEntry, or list of these types")

    def __iter__(self) -> Iterator[LogEntry]:
        return iter(self.log_entries)

    def __getitem__(self, index: int) -> LogEntry:
        return self.log_entries[index]

    def __len__(self) -> int:
        return len(self.log_entries)


def get_entry_type_registry() -> Dict[str, Type[BaseEntry]]:
    """
    Automatically creates the ENTRY_TYPE_REGISTRY based on the subclasses of BaseEntry.
    """
    registry: Dict[str, Type[BaseEntry]] = {}
    for subclass in BaseEntry.__subclasses__():
        if issubclass(subclass, BaseModel):
            type_field = subclass.model_fields.get('type')
            if type_field is not None:
                registry[type_field.default] = subclass
    return registry