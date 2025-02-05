import os
import logging
import subprocess
from typing import List, Optional, Any, Dict, Iterator, ClassVar, Literal, AsyncIterator
import requests
import json
from inspect import signature
import time

from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.environment import Template
from pydantic import Field
from langchain_core.messages import (
    AIMessageChunk, HumanMessage, AIMessage, BaseMessage)
from langchain.chat_models.base import BaseChatModel
from langchain_core.outputs import ChatGenerationChunk, ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langgraph.graph import MessagesState
from openai import OpenAI
from ollama import AsyncClient as AsyncOllamaClient
from ollama import Client as OllamaClient

from src.registry import GraphManager
from src.entry import LoggedAttribute


class LlamacppException(Exception):
  pass


class vLLMException(Exception):
  pass


def flatten_list(nested_list):
    return sum([
        flatten_list(item) if isinstance(item, list) else [item]
        for item in nested_list], [])


def run_server(
    server: Literal["llamacpp", "vllm", "ollama"],
    model_path: str,
    port: int,
    clip_model_path=None,
    use_gpu=False, 
    **kwargs
    ):
    
    """
    """
    if server == "llamacpp":
        llamacpp_path = os.environ.get("LLAMA_CPP_PATH", None)
        if not llamacpp_path:
            raise LlamacppException(
                "LLAMA_CPP_PATH was not found. Check .env file and README.md for details")
        
        executable = os.path.normpath(
            os.environ["LLAMA_CPP_PATH"] + "llama-server")
        cmd = [executable]
        
        arguments = [
            "--port", str(port),
            "-m", os.path.normpath(model_path),
            "--n-gpu-layers", "0" if not use_gpu else kwargs.pop("n_gpu_layers", "0"),
        ]
        # TODO IS NOT SUPPORTED NOW!
        if clip_model_path:
            arguments.extend(
                ["--clip_model_path", os.path.normpath(clip_model_path)])
        arguments.append("--no-webui")
    
    elif server == "vllm":
        vllm_path = os.environ.get("VLLM_PATH", None)
        if not vllm_path:
            raise vLLMException(
                "VLLM_PATH was not found. Check .env file and README.md for details")
        cmd = [vllm_path,  "serve"]
        arguments = [
            model_path,
            "--port", str(port)
        ]
        if not use_gpu:
            #TODO Play with vLLM parameters
            pass
    
    elif server == "ollama":
        cmd = ["ollama",  "serve"]
        # Start the subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            )
        logging.info(f"Started process with PID: {process.pid}")
        return process
    
    kwargs.pop("sampling", None)

    for k, v in kwargs.items():
        if isinstance(v, (dict, list)):
            continue
        if isinstance(v, bool):
            arguments.append("--" + str(k))
            continue
        arguments.extend(["--" + str(k), str(v)])
    
    cmd.extend([str(arg) for arg in arguments])

    # Start the subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        )
    logging.info(f"Started process with PID: {process.pid}")
    
    
    return process


def check_server_healthy(
    port: int, 
    model_name: str, 
    timeout_seconds: int = 100, 
    time_to_wait:int = 10) -> bool:
    
    start_time = time.time()    
    while True:
        try:
            # Try to retrieve models as a health check
            response = requests.get(
                f"http://0.0.0.0:{port}/v1/models", timeout=5)
            if response.status_code == 200:
                logging.info(
                    (f"✓ Graph server is healthy! Model {model_name}" 
                     f" is available. Check http://0.0.0.0:{port}/v1/models")
                    )
                break

        except Exception as e:
            elapsed_time = time.time() - start_time
            
            if elapsed_time >= timeout_seconds:
                logging.error(
                    (f"✗ Health check failed after {timeout_seconds} "
                     "seconds timeout.")
                )
                logging.error(f"Last error: {str(e)}")
                return False
            
            logging.info(
                (f"✗ Server not healthy ({str(e)}). "
                 f"Retrying in {time_to_wait} second(s)...")
            )
            time.sleep(time_to_wait)

    return True


async def run_graph_in_terminal(
    graph_manager: GraphManager, 
    config: str, 
    session_id: str ="001"
    ):

    graph_manager.connect_session(session_id, config)
    session = graph_manager.get_session(session_id)

    try:
        graphai = session.graph.graph_call
        config = {
            "configurable": 
            {"thread_id": f"thread-test-{session_id}"}
                }
        
        while True:
                
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            input_message = HumanMessage(content=user_input)

            print("\nAssistant:", end="", flush=True)
            async for chunk_type, stream_data in graphai.astream(
                {"messages": [input_message]},
                config=config,
                stream_mode=["messages", "values"]):
                
                if chunk_type == "messages":
                    chunk, metadata = stream_data
                    if isinstance(chunk, AIMessageChunk):
                        if chunk.content:
                            print(chunk.content, end="", flush=True)

                if chunk_type == "values":
                    for k, v in stream_data.items():
                        if isinstance(v, LoggedAttribute):
                            for item in v:
                                print(f'{k} - {item.content_to_send()}')

            print()
    except Exception as e:
        print(e)
    finally:
        graph_manager.remove_session(session_id, with_graph=True)


# Based on
# https://python.langchain.com/docs/how_to/custom_chat_model/

class LlamaCppClient(BaseChatModel):

    model_name: str = Field(default="model", alias="llamacpp_server:")
    base_url: str
    chat_template: Template

    SYSTEM_MESSAGE: ClassVar[str] = (
        "A chat between a curious human and an artificial intelligence "
        "assistant.  The assistant gives helpful, detailed, and polite answers "
        "to the human's questions."
    )

    TEMPLATE: ClassVar[str] = """
        {# Loop through all messages #}
        {%- for message in messages -%}
            {# System message #}
            {%- if message.role == 'system' -%}
                <|system|>\n{{ message.content }}<|end|>
            {%- endif -%}
            
            {# User message #}
            {%- if message.role == 'user' -%}
                <|user|>
                {%- if message.content is defined -%}
                    {%- if message.content is string -%}
                        \n{{ message.content }}<|end|>
                    {%- endif -%}
                    {%- if message.content is iterable and not message.content is string -%}
                        {%- for content in message.content -%}
                            {%- if content.type == 'text' -%}
                                \n{{ content.text }}<|end|>
                            {%- endif -%}
                        {%- endfor -%}
                    {%- endif -%}
                {%- endif -%}
            {%- endif -%}
            
            {# Assistant message #}
            {%- if message.role == 'assistant' -%}
                <|assistant|>\n{{ message.content }}<|end|>
            {%- endif -%}
        {%- endfor -%}
        {# Generation prompt #}
        {%- if add_generation_prompt -%}
            <|assistant|>\n
        {%- endif -%}
    """
    
    def __init__(self, base_url: str, **kwargs):
        # Use Pydantic's `construct` method for proper initialization
        chat_template = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True).from_string(self.TEMPLATE)
        super().__init__(
            base_url=base_url, chat_template=chat_template, **kwargs)

    def _format_message_state(self, state: MessagesState):

        formatted_messages = []
        if self.SYSTEM_MESSAGE:
            formatted_messages.append({
                "role": "system",
                "content": self.SYSTEM_MESSAGE
            })
            
        for message in state:
            
            if isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, HumanMessage):
                role = "user"
            else:
                raise ValueError(f"Unknown message type: {type(message)}")
            
            message_dict = {
                "role": role,
                "content": message.content
            }

            formatted_messages.append(message_dict)

        prompt = self.chat_template.render(
            messages=formatted_messages,
            add_generation_prompt=True)

        return prompt
    
    def _generate(self, messages, stop = None, run_manager = None, **kwargs):
        return super()._generate(messages, stop, run_manager, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        https://github.com/ggerganov/llama.cpp/tree/master/examples/server
        """
        prompt = self._format_message_state(messages)
        endpoint = f"{self.base_url}/completion"
        payload = {
            "prompt": prompt,
            "stream": True,
        }
        payload.update(kwargs)
        
        try:
            response_stream = requests.post(endpoint, json=payload, stream=True)

        except requests.exceptions.RequestException as e:
            raise Exception(
                f"Error making request to llama.cpp server: {str(e)}")

        for line in response_stream.iter_lines():

            line = line.decode('utf-8')
            if line:
                line_data = json.loads(line.replace("data: ", ""))
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=line_data['content']))

                if run_manager:
                    # This is optional in newer versions of LangChain
                    # The on_llm_new_token will be called automatically
                    run_manager.on_llm_new_token(line_data, chunk=chunk)
                
                yield chunk

        # Let's add some other information (e.g., response metadata)
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={
                "tokens_predicted": line_data["tokens_predicted"],
                "tokens_evaluated": line_data["tokens_evaluated"]}))

        yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {"model_name": self.model_name}
    

class OllamaClientWrapper(BaseChatModel):

    model_name: str
    client: AsyncOllamaClient | OllamaClient
    DEFAULT_SYSTEM_MESSAGE: ClassVar[str] = (
        "A chat between a curious human and an artificial intelligence "
        "assistant.  The assistant gives helpful, detailed, and polite answers "
        "to the human's questions."
    )
    
    def __init__(
            self, 
            model_name: str, 
            client: AsyncOllamaClient | OllamaClient,
            system_message: Optional[str] = None,
            **kwargs):
        
        super().__init__(model_name=model_name, client=client, **kwargs)
        self._system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE

    @property
    def system_message(self) -> str:
        """Get the current system message."""
        return self._system_message

    @system_message.setter
    def system_message(self, value: str) -> None:
        """Set a new system message."""
        if not isinstance(value, str):
            raise TypeError("System message must be a string")
        if not value.strip():
            raise ValueError("System message cannot be empty")
        self._system_message = value

    def _format_message_state(self, state: MessagesState):

        formatted_messages = []
        formatted_messages.append({
            "role": "system",
            "content": self.system_message
        })
        
        for message in state:
            
            if isinstance(message, AIMessage):
                message_dict = {
                    "role": "assistant",
                    "content": message.content
                }
            
            elif isinstance(message, HumanMessage):
                role = "user"
                message_dict = {"role": role}
                if isinstance(message.content, list):
                    images = []
                    texts = []
                    for part in message.content:
                        if part["type"] == "text":
                            texts.append(part["text"])
                            continue

                        if part["type"] == "image_url":
                            content = part["image_url"]["url"].split("base64,")[-1]
                            images.append(content)
                            continue
                    if images:
                        message_dict["images"] = images
                    if texts:
                        message_dict["content"] = "\n\n".join(texts)
                else:
                    message_dict["content"] = message.content

            else:
                raise ValueError(f"Unknown message type: {type(message)}")
            
            formatted_messages.append(message_dict)

        return formatted_messages
    
    def _generate(self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        tools: Optional[Any] = None,
        **kwargs: Any,
    )-> ChatResult:
        
        formatted_state = self._format_message_state(messages)
        result = self.client.chat(
            model=self.model_name, 
            messages=formatted_state, 
            stream=False,
            tools=tools,
            options=kwargs)

        # additional statistics could be set        
        generation = ChatGeneration(
            message=AIMessage(
                content=result['message']['content'], 
                response_metadata={
                    "eval_count": result["eval_count"],
                    "prompt_eval_count": result["prompt_eval_count"]
                }
            )
        )
        
        return ChatResult(generations=[generation])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        tools: Optional[Any] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        
        formatted_state = self._format_message_state(messages)

        async for part in await self.client.chat(
            model=self.model_name, 
            messages=formatted_state, 
            stream=True,
            tools=tools,
            options=kwargs):

            if part['message']['content']:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=part['message']['content']))

                if run_manager:
                    # This is optional in newer versions of LangChain
                    # The on_llm_new_token will be called automatically
                    run_manager.on_llm_new_token(part, chunk=chunk)
                
                yield chunk

        # Let's add some other information (e.g., response metadata)
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={
                "eval_count": part["eval_count"],
                "prompt_eval_count": part["prompt_eval_count"]
                }))

        yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "from_ollama"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {"model_name": self.model_name}


class OpenAICompatibleChatModel(BaseChatModel):

    "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#cli-reference"

    model_name: str
    client: OpenAI
    model_type: str = "Llama"
    SYSTEM_MESSAGE: ClassVar[str] = (
        "A chat between a curious human and an artificial intelligence "
        "assistant.  The assistant gives helpful, detailed, and polite answers "
        "to the human's questions.")
    TEMPLATE: ClassVar[str] = ""
    
    def __init__(self, model_name: str, client: Any, **kwargs):
        super().__init__(model_name=model_name, client=client, **kwargs)

    def _format_message_state(self, state: MessagesState):

        formatted_messages = []
        if self.SYSTEM_MESSAGE:
            formatted_messages.append({
                "role": "system",
                "content": self.SYSTEM_MESSAGE
            })
        for message in state:
            
            if isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, HumanMessage):
                role = "user"
            else:
                raise ValueError(f"Unknown message type: {type(message)}")
            
            message_dict = {
                "role": role,
                "content": message.content
            }

            formatted_messages.append(message_dict)

        return formatted_messages
    
    def _generate(self, messages, stop = None, run_manager = None, **kwargs):
        return super()._generate(messages, stop, run_manager, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        
        messages = self._format_message_state(messages)
        
        if kwargs.get("chat_template", False):
            kwargs["chat_template"] = self.TEMPLATE
        
        sig = signature(self.client.chat.completions.create)
        extra_body = kwargs.pop("extra_body", {})
        reset_kwargs = {}
        for key, value in kwargs.items():
            if key in sig.parameters:
                reset_kwargs[key] = value
            else:
                extra_body[key] = value
        
        reset_kwargs["extra_body"] = extra_body
        
        response_stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            # stream_options={'include_usage': True}
            **reset_kwargs,
            )

        for _token in response_stream:

            token = _token.choices[0].delta.content
            if token:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=token)
                )

                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                
                yield chunk

        # Let's add some other information (e.g., response metadata)
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={}))
        yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_type

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {"model_name": self.model_name}
    

class PlaceholderModel(BaseChatModel):
    
    model_name: str = Field(default="ghost")

    """A placeholder model that always returns a predefined AI message."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _generate(self, messages, stop = None, run_manager = None, **kwargs):
        return super()._generate(messages, stop, run_manager, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        response_text: str = "",
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        
        for word in response_text.split(" "):

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
    