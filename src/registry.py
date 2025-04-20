import importlib
import logging
import json
import os
import sys
import gc
import socket
import threading
import subprocess
from typing import Dict, Any, Callable, Optional, Literal, Set, List, Union
from threading import Lock
from collections import defaultdict
from functools import wraps
from dataclasses import dataclass
from enum import Enum

from langchain.chat_models.base import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
import psutil


def terminate_processes(processes: Set[psutil.Process], timeout=5):
    """
    Safely terminate a processes and their children
    """
    
    for process in processes:
        try:
            # Check if process still exists
            if not psutil.pid_exists(process.pid):
                continue
            pid = process.pid
            # try:
            #     process.stdout.close()
            #     process.stderr.close()
            # except AttributeError:
            #     pass
            parent = psutil.Process(process.pid)
            # Get all child processes
            children = parent.children(recursive=True)
            # Terminate children first
            for child in children:
                try:
                    child.terminate()
                    child.wait(timeout=timeout)  # Wait for child to terminate
                except (psutil.NoSuchProcess, subprocess.TimeoutExpired):
                    logging.info((f"Child process {child.pid} already "
                                  "terminated or could not be terminated."))
            # Terminate parent process
            process.terminate()
            # Wait for process to terminate
            process.wait(timeout=timeout)
        except psutil.NoSuchProcess:
            logging.info(f"Process {pid} already terminated.")
        except subprocess.TimeoutExpired:
            logging.info(f"Force killing process {pid}")
            process.kill()  # Force kill if it doesn't terminate
        except Exception as e:
            logging.error(f"An error occurred with process {pid}: {e}")
            continue


def terminate_threads(threads: Set[threading.Thread], timeout=5):
    # Cleanup threads
    for thread in threads:
        try:
            if thread.is_alive():
                # Set thread termination flag if supported
                if hasattr(thread, '_stop'):
                    thread._stop()
                
                # Wait for thread to finish
                thread.join(timeout=timeout)
                
                if thread.is_alive():
                    logging.info(f"thread {thread.name} is alive")
                else:
                    logging.info(f"thread {thread.name} is dead")
        except Exception as e:
            logging.error(f"{str(e)} for {thread.name}")
            continue


@dataclass
class CompiledGraphResult:
    graph_call: CompiledStateGraph
    processes: Set[psutil.Process]
    threads: Set[threading.Thread]
    clients: List[BaseChatModel]
    metadata: Dict


class ModalsType(str, Enum):
    TEXT = 'text'
    IMAGE = 'image'
    VIDEO = 'video'
    AUDIO = 'audio'


def tool_graph(
    name: str, 
    tag: str,
    att_modals: List[ModalsType] = ['text'],
    entries_map: bool = False
):
    """
    Enhanced decorator that tracks processes, threads, and metadata.
    
    :param name: User-friendly name for the graph
    :param tag: Tag for categorizing the graph
    :param att_modals: Type of modal
    :param entries_map: Available for agent outputs.
    :return: Decorator function that tracks processes, threads and metadata
    """
    def decorator(func: Callable) -> Callable:

        # Store metadata
        metadata = {
            'name': name or func.__name__,
            'tag': tag,
            'att_modals': att_modals,
            'entries_map': entries_map,
            'outputs': []
        }
        
        # Store metadata on the wrapper function for backwards compatibility
        func._graph_metadata = metadata

        @wraps(func)
        def wrapper(*args, **kwargs) -> CompiledGraphResult:
            
            # threads_before = set(threading.enumerate())
            # Execute the function

            results = func(*args, **kwargs)
            if not results:
                return
            if not isinstance(results, (tuple, list)):
                results = (results,)
            new_processes = []
            new_clients = []
            for out in results:
                if isinstance(out, CompiledStateGraph):
                    compiled_state_graph = out
                if isinstance(out, subprocess.Popen):
                    new_processes.append(out)
                if isinstance(out, BaseChatModel):
                    new_clients.append(out)
                else:
                    func._graph_metadata["outputs"].append(out)

            # threads_after = set(threading.enumerate())
            # new_threads = threads_after - threads_before
            
            return CompiledGraphResult(
                graph_call=compiled_state_graph,
                processes=set(new_processes),
                threads=set(),
                clients=new_clients,
                metadata=metadata
            )

        return wrapper
    
    return decorator


class GraphContainer:
    """
    
    """
    def __init__(
            self, 
            graph_func: Callable[..., CompiledGraphResult],
            name: str,
            tag: str,
            att_modals: Literal['text', 'image', 'video', 'audio'],
            entries_map: Any = None,
            **kwargs):
        
        self.graph_compiled = graph_func()
        self.name = name
        self.tag = tag
        self.att_modals = att_modals
        self.entries_map = entries_map


    @property
    def graph_call(self):
        return self.graph_compiled.graph_call


class Session:
    """
    Represents a user session.
    """
    def __init__(self, session_id: str, graph: GraphContainer):
        self.session_id = session_id
        self.graph = graph
        self.graph_tempates = {}

    def __repr__(self):
        return f"Session(session_id={self.session_id}, graph={self.graph.name})"


class GraphRegistry:
    
    """
    A registry class for managing graph modules and their configurations.

    This class provides the functionality to initialize, load, and manage graph 
    plugins and their associated configuration files. It allows for automatic 
    discovery and loading of graph modules from a specified directory, 
    and provides methods to retrieve and list available graph functions along 
    with their descriptions.

    Parameters:
    ----------
    graph_dir (str): Directory where graph modules are stored.
    config_dir (str): Directory where configuration JSON files are stored.

    Examples:
    ----------
    >>> registry = GraphRegistry(graph_dir='src/graphs', config_dir='configs')
    >>> registry.load_graphs()
    Loaded graph modules and configurations from 'src/graphs' and 'configs' 
    directories.

    To retrieve a specific graph function by its user-friendly name and 
    optionally pass configuration parameters:
    >>> graph_func = registry.get_graph('example_graph', with_config=True)
    >>> result = graph_func(data, config_params)

    To list all available graph names with their descriptions:
    >>> graph_descriptions = registry.list_graphs()
    >>> for name, description in graph_descriptions.items():
    ...     print(f"Graph: {name}, Description: {description}")

    The class also contains internal methods to handle loading configurations 
    from JSON files, registering module plugins, and generating wrapper 
    functions for injecting configurations into graph functions.
    """
    
    def __init__(self, graph_dir: str = 'src/graphs',
                       config_dir: str = 'configs'):
        """
        Initialize the graphs registry.
        :param plugin_dir: Directory where graphs modules will be stored
        :param config_dir: Directory where configuration JSON files will be 
        stored
        """

        self._graphs: Dict[str, Any] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self.graph_dir = graph_dir
        self.config_dir = config_dir
        # Ensure plugin directory exists
        os.makedirs(graph_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        # Add graph directory to Python path
        if graph_dir not in sys.path:
            sys.path.insert(0, graph_dir)
        self.port_range_start = 10000
        self.port_range_end = 11000
        self.used_ports: Set[int] = set()

    def load_graphs(self):
        """
        Automatically discover and load graphs from the plugin directory.
        Also load corresponding configuration files.
        """
        # Clear existing plugins and configs
        self._graphs.clear()
        self._configs.clear()

        # Load configuration files first
        self._load_configurations()

        for agent_name, params in self._configs.items():
            filename = params["agent_graph"]
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]  # Remove .py extension
                try:
                    module = importlib.import_module(module_name)
                    self._register_module(module, agent_name)
                except ImportError as e:
                    logging.error(f"Error loading graph {module_name}: {e}")
        
        return self._graphs
    
    def _load_configurations(self):
        """
        Load JSON configuration files from the config directory.
        Configurations are named after their plugin with a .json extension.
        """
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json') and not filename.startswith('__'):
                config = filename[:-5]  # Remove .json extension
                try:
                    with open(os.path.join(self.config_dir, filename), 'r') as f:
                        self._configs[config] = json.load(f)
                except (IOError, json.JSONDecodeError) as e:
                    logging.error(f"Error loading config {filename}: {e}")

    def is_port_available(self, port: int) -> bool:
        """Check if a port is available on localhost."""
        try:
            # Create a socket object
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Try to bind to the port
                sock.bind(('localhost', port))
                return True
        except (socket.error, OSError):
            return False

    def find_available_port(self) -> Optional[int]:
        """Find the first available port in the specified range."""
        for port in range(self.port_range_start, self.port_range_end + 1):
            if port not in self.used_ports and self.is_port_available(port):
                return port
        return None

    def _register_module(self, module, agent_name):
        """
        Register a module's plugin functions.
        
        :param module: The imported module to register
        """
        module_graphs = {}
        # Look for functions with our graph decorator
        for name, func in module.__dict__.items():
            if hasattr(func, '_graph_metadata'):
                # Extract metadata
                if isinstance(func._graph_metadata, dict):
                    metadata = func._graph_metadata
                    available_port = self.find_available_port()
                    if available_port is None:
                        logging.error("No available ports in the specified range!!")
                        break
                    metadata.update({"port": available_port})
                    # Store graphs with its metadata
                    module_graphs[agent_name] = {
                        'function': func,
                        'graph_params': metadata
                    }
        # Update the main plugins dictionary
        self._graphs.update(module_graphs)
    
    def get_graph(self, name: str, with_config: bool = True) -> Callable:
        """
        Retrieve a specific graph function.
        
        :param name: User-friendly name of the graph function
        :param with_config: Whether to pass configuration to the function
        :return: The graph function
        :raises KeyError: If the graph is not found
        """
        if name not in self._graphs:
            raise KeyError(f"Graph '{name}' not found")
        
        graph_info = self._graphs[name]
        func = graph_info['function']
        graph_params = graph_info['graph_params']
        
        # If configuration exists and with_config is True, inject config
        config = {}
        if with_config:
            config = self._configs.get(name, {})
            
        # Create a wrapper function that injects configuration
        def config_wrapper(*args, **kwargs):
            # Combine explicit kwargs with config, giving precedence to 
            # explicit kwargs
            merged_kwargs = {**config, **kwargs}
            merged_kwargs['port'] = graph_params.get("port")
            return func(*args, **merged_kwargs)
        
        return config_wrapper, graph_params
        
    def list_graphs(self) -> Dict[str, str]:
        """
        List all available graph names with their tags.
        
        :return: Dictionary of graph names to tags
        """
        return (
            (name, graph_info['graph_params']['tag'])
            for name, graph_info in self._graphs.items()
            )


class GraphManager:
    """ 
    Manages the graphs in memory and handles session connections.
    """
    def __init__(
        self, 
        graph_dir: str = 'src/graphs', 
        config_dir: str = 'configs',
        default_graph_name: str = 'Ghost'
        ):
        self._graphs_registry = GraphRegistry(
            graph_dir=graph_dir,
            config_dir=config_dir)
        registered_graphs = self._graphs_registry.load_graphs()
        self._default_graph = default_graph_name
        # self._default_graph = list(
        #     k for k in registered_graphs.keys() if "default" in k)[0]
        
        self._graphs: Dict[str, GraphContainer] = {}  # Holds graphs by name
        self._sessions: Dict[str, Session] = {}  # Maps session IDs to sessions
        # Maps graph names to sessions which use it
        self._graph_sessions = defaultdict(list) 
        self._lock = Lock()  # Ensures thread-safe operations

    @property
    def default_graph(self):
        return self._default_graph
    
    def reload_graphs(self):
        self._graphs_registry.load_graphs()
        # sessions clean up ??

    def _get_or_create_graph(self, graph_name: str, session_id: str) -> GraphContainer:
        """
        Retrieves a graph by name. Creates it if not already in memory.
        """
        if graph_name not in self._graphs:
            get_graph_func, graph_params  = self._graphs_registry.get_graph(
                graph_name)
            graph_params["name"] = graph_name
            self._graphs[graph_name] = GraphContainer(
                get_graph_func, **graph_params)
        
        self._graph_sessions[graph_name].append(session_id)

        return self._graphs[graph_name]

    def connect_session(
            self, 
            session_id: str, 
            graph_name: str,
        ) -> Session:
        """
        Connects a session to the specified graph. If no graph is specified, 
        the default graph is used.
        """
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(
                    f"Session with ID {session_id} already exists.")

            graph = self._get_or_create_graph(graph_name, session_id)
            session = Session(session_id, graph)
            self._sessions[session_id] = session            
            return session


    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieves an existing session by ID.
        """
        with self._lock:
            return self._sessions.get(session_id, None)

    def remove_session(self, session_id: str, with_graph: bool =True) -> None:
        """
        Removes a session by ID.
        """
        try:
            with self._lock:
                if session_id in self._sessions:
                    session = self._sessions.pop(session_id)
                    graph_name = session.graph.name
                    self._graph_sessions[graph_name].remove(session_id)
                    del session
                    
                    if not self._graph_sessions[graph_name] and with_graph:
                        unused_graph = self._graphs.pop(graph_name)

                        if unused_graph:
                            if unused_graph.graph_compiled.processes:
                                terminate_processes(
                                    unused_graph.graph_compiled.processes)
                            if unused_graph.graph_compiled.threads:
                                terminate_threads(
                                    unused_graph.graph_compiled.threads)
                            del unused_graph
        
        except Exception as e:
            logging.error(e)
            
        finally:
            gc.collect()
        

    async def terminate_all(self) -> None:
        """
        Unconditionally terminates all sessions, graphs, and their associated
        processes and threads. This is a destructive operation that should be
        used with caution, typically during shutdown or cleanup.
        
        This method:
        1. Acquires the thread lock to prevent new sessions/graphs from being created
        2. Terminates all processes and threads associated with each graph
        3. Removes all sessions and graphs from memory
        4. Forces garbage collection
        """
        with self._lock:
            # First terminate all processes and threads for each graph
            for graph in self._graphs.values():
                if graph.graph_compiled.processes:
                    terminate_processes(graph.graph_compiled.processes)
                if graph.graph_compiled.threads:
                    terminate_threads(graph.graph_compiled.threads)
            
            # Clear all session mappings
            self._sessions.clear()
            self._graph_sessions.clear()
            
            # Clear graphs dictionary after terminating all processes/threads
            self._graphs.clear()
            
            # Force garbage collection to clean up deleted objects
            gc.collect()


    def list_sessions(self) -> Dict[str, Session]:
        """
        Returns all active sessions.
        """
        with self._lock:
            return dict(self._sessions)