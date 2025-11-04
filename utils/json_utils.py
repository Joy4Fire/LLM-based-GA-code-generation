import json
import os
from typing import Any, Optional


class JsonFileHandler:
    """
    JSON file read/write utility class
    Provides basic reading and saving functionality for JSON files
    """

    @staticmethod
    def read_json(file_path: str, encoding: str = 'utf-8') -> Optional[Any]:
        """
        Read JSON file

        Args:
            file_path: JSON file path
            encoding: file encoding, defaults to 'utf-8'

        Returns:
            JSON data (dict, list, etc.), returns None if reading fails
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, 'r', encoding=encoding) as file:
                return json.load(file)

        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return None

    @staticmethod
    def save_json(data: Any, file_path: str,
                  encoding: str = 'utf-8',
                  indent: int = 4,
                  ensure_ascii: bool = False) -> bool:
        """
        Save data as JSON file

        Args:
            data: data to save
            file_path: target file path
            encoding: file encoding, defaults to 'utf-8'
            indent: indentation spaces, defaults to 4
            ensure_ascii: whether to ensure ASCII encoding, defaults to False

        Returns:
            bool: whether saving was successful
        """
        try:
            # Ensure directory exists
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            with open(file_path, 'w', encoding=encoding) as file:
                json.dump(data, file, indent=indent, ensure_ascii=ensure_ascii)

            return True

        except Exception as e:
            print(f"Error saving JSON file: {e}")
            return False