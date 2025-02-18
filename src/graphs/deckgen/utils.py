from RestrictedPython import compile_restricted
from RestrictedPython import safe_builtins, safe_globals
from RestrictedPython.PrintCollector import PrintCollector
import io
import contextlib
import traceback
import resource
import signal
import importlib
import multiprocessing
from multiprocessing import Process, Queue
import builtins
import os
import sys
from pathlib import Path


class SafeFileOps:
    def __init__(self, allowed_path):
        self.allowed_path = Path(allowed_path).resolve()
        # Create the directory if it doesn't exist
        os.makedirs(self.allowed_path, exist_ok=True)

    def _is_path_allowed(self, filepath):
        # Resolve the absolute path
        try:
            absolute_path = Path(filepath).resolve()
            # Check if the path is within allowed directory
            return self.allowed_path in absolute_path.parents or self.allowed_path == absolute_path
        except Exception:
            return False

    def safe_open(self, filepath, mode='r', *args, **kwargs):
        if not self._is_path_allowed(filepath):
            raise PermissionError(
                f"Access denied. Can only access files in {self.allowed_path}")
        
        # Restrict modes to only reading and writing
        if not any(m in mode for m in 'rwa'):
            raise ValueError("Invalid file mode")
        
        return open(filepath, mode, *args, **kwargs)

    def safe_write(self, filepath, content):
        if not self._is_path_allowed(filepath):
            raise PermissionError(
                f"Access denied. Can only write files to {self.allowed_path}")
        
        with open(filepath, 'w') as f:
            f.write(content)


def get_safe_globals(allowed_path="/tmp/restricted"):
    safe_env = {}

    # Create SafeFileOps instance
    safe_file_ops = SafeFileOps(allowed_path)

    # Add safe file operations
    safe_env['open'] = safe_file_ops.safe_open
    safe_env['write_file'] = safe_file_ops.safe_write

    # Add all built-ins
    safe_env.update({
        name: getattr(builtins, name)
        for name in dir(builtins)
        if not name.startswith('_')  # Exclude private attributes
    })

    # Add specific modules you want to make available
    modules_to_import = ['os', 'pptx', 'pptx.util']  # Add more as needed
    for module_name in modules_to_import:
        try:
            safe_env[module_name] = importlib.import_module(module_name)
        except ImportError:
            pass

    # Add specific classes/functions from modules
    if 'pptx' in safe_env:
        safe_env['Presentation'] = safe_env['pptx'].Presentation
    if 'pptx.util' in safe_env:
        safe_env['Inches'] = safe_env['pptx.util'].Inches

    return safe_env


def set_resource_limits(mbs=1000, cpu_time=100):
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mbs * 1024 * 1024, -1))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_time, -1))
    except (ValueError, resource.error):
        pass


def execute_code(source_code, allowed_path, result_queue):
    try:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):

            globals_dict = get_safe_globals(allowed_path)
            locals_dict = {}

            exec(source_code, globals_dict, locals_dict)
            result = locals_dict.get('_result', None)

            result_queue.put(
                ('success', result, stdout.getvalue(), stderr.getvalue())
            )

    except Exception as e:
        result_queue.put(('error', None, '', str(e)))


def run_code(source_code, allowed_path="/tmp/restricted", timeout=15):

    result_queue = Queue()

    # Create and start the process
    process = Process(
        target=execute_code,
        args=(source_code, allowed_path, result_queue)
    )

    try:
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return None, '', 'Execution timed out'

        if not result_queue.empty():
            status, result, stdout, stderr = result_queue.get()
            if status == 'success':
                return result, stdout, stderr
            else:
                return None, stdout, stderr

    except Exception as e:
        return None, '', f'Execution error: {str(e)}'

    finally:
        # Ensure process is terminated
        if process.is_alive():
            process.terminate()
            process.join()

    return None, '', 'Execution failed'