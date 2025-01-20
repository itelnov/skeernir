from markdown import Markdown
import re
import html
from jinja2 import Environment, FileSystemLoader


env = Environment(loader=FileSystemLoader("templates"))
code_template = env.get_template('partials/code_block.html')


class CodeBlockPreprocessor:
    
    def __init__(self):
        self.code_blocks = {}
        self.counter = 0
        
    def _get_language_display_name(self, lang):
        """Get a clean display name for the language"""
        lang_map = {
            'py': 'Python',
            'js': 'JavaScript',
            'ts': 'TypeScript',
            'html': 'HTML',
            'css': 'CSS',
            'java': 'Java',
            'cpp': 'C++',
            'c': 'C',
            'cs': 'C#',
            'go': 'Go',
            'rust': 'Rust',
            'rb': 'Ruby',
            'php': 'PHP',
            'swift': 'Swift',
            'kotlin': 'Kotlin',
            'plaintext': 'Plain Text',
            'txt': 'Plain Text',
            'shell': 'Shell',
            'bash': 'Bash',
            'sql': 'SQL',
            'json': 'JSON',
            'xml': 'XML',
            'yaml': 'YAML',
            'markdown': 'Markdown',
            'md': 'Markdown'
        }
        return lang_map.get(lang.lower(), lang.capitalize())
    
    def _process_code_block(self, match):
        lang = match.group(1)  # Language is now in group 1
        code = match.group(2)  # Code is now in group 2
        # code = match.group(3)
        # lang = match.group(2) if match.group(2) else 'plaintext'
        # lang_display = self._get_language_display_name(lang)
        
        data = {
            "lang_display": lang, # self._get_language_display_name(lang),
            "block_id": f'code-block-{self.counter}',
            "escaped_content": html.escape(code.strip()),
            "lang": f"language-{lang}",
            "indicator": f"#copy-indicator-{self.counter}"
        }
        
        placeholder = f'CODEBLOCK_{self.counter}'
        self.code_blocks[placeholder] = code_template.render(**data)
        self.counter += 1
        
        return placeholder


class MarkdownConverter:
    """
    A class to handle Markdown to HTML conversion with extended features.
    """
        
    def convert(self, markdown_text):
        """
        Convert Markdown to HTML with all extensions enabled.
        
        Args:
            markdown_text (str): Input markdown text
            
        Returns:
            str: HTML output with all extensions processed
        """
        # Initialize preprocessor for code blocks
        preprocessor = CodeBlockPreprocessor()
        
        # Process code blocks before markdown conversion
        # with plain text (3 groups)
        # pattern = '```((\w+)\n)?(.*?)(?<=\n)```(\n|$)'
        # without plain text (2 groups)
        pattern = r'```(\w+)\n(.*?)(?<=\n)```(?:\n|$)'

        processed_text = re.sub(
            pattern, 
            preprocessor._process_code_block, 
            markdown_text, 
            flags=re.DOTALL)

        # Convert markdown to HTML with all extensions
        md = Markdown(extensions=['extra', 'sane_lists'])

        # processed_text = html.escape(processed_text)
        html_output = md.convert(processed_text)

        # TODO insirt as a template !!! 
        for placeholder, formatted_code in preprocessor.code_blocks.items():
            html_output = html_output.replace(f"{placeholder}", formatted_code)
        return html_output