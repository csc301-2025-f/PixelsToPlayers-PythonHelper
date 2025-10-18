import platform
from typing import Optional

def get_os_version() -> Optional[tuple[str, int]]:

    system : str = platform.system().lower()

    match system:
        case 'windows':
            release : str = platform.release()
            version_number : int

            try:
                version_number = int(release)
                return system, version_number
            except (ValueError, TypeError) as e:
                print(release, e)
                return system, 0
        case 'linux':
            return system, 0
        case 'macos':
            return system, 0
        case _:
            return None

