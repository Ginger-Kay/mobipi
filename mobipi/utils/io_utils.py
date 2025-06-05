"""
IO Utilities.

@yjy0625
"""
import re


class DualStream:
    def __init__(self, file_stream, console_stream):
        self.file_stream = file_stream
        self.console_stream = console_stream

    def write(self, message):
        self.file_stream.write(message)
        self.console_stream.write(message)
        self.file_stream.flush()
        self.console_stream.flush()

    def flush(self):
        self.file_stream.flush()
        self.console_stream.flush()

    def isatty(self):
        # Delegate to console stream
        return self.console_stream.isatty()


def camel_to_snake_case(camel_str):
    """
    Converts a CamelCase string to snake_case.

    Args:
        camel_str (str): The CamelCase string.

    Returns:
        str: The snake_case string.
    """
    # Insert underscores between words and convert to lowercase
    snake_str = re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()
    return snake_str


