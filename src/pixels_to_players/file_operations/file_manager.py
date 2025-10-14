from pathlib import Path

class FileManager:
    @staticmethod
    def write_to_file(filepath: Path, content: str):
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding="utf-8")
        except Exception as e:
            raise IOError(f"Failed to write to file {filepath}: {e}")

    @staticmethod
    def read_file(filepath: Path) -> str:
        try:
            return filepath.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filepath} not found.")
        except Exception as e:
            raise IOError(f"Failed to read file {filepath}: {e}")