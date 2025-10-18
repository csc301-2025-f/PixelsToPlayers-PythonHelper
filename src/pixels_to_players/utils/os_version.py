from __future__ import annotations

import platform
from typing import Optional


class OSVersion:
    name : str
    version_number = 0

    @staticmethod
    def get_os_version() -> Optional[OSVersion]:
        system : str = platform.system()

        match system:
            case 'Windows':
                release : str = platform.release()
                version_number : int

                try:
                    version_number = int(release)
                except (ValueError, TypeError) as e:
                    version_number = 0
                    print(release, e)
                return WindowsVersion(version_number)
            case 'Darwin':
                mac_ver = platform.mac_ver()[0]
                if mac_ver:
                    parts = [int(p) for p in mac_ver.split('.') if p.isdigit()]
                    # normalize to (major, minor, patch)
                    while len(parts) < 3:
                        parts.append(0)
                    return MacOSVersion((parts[0], parts[1], parts[2]))
                return MacOSVersion((0, 0, 0))

            case 'Linux':
                return LinuxVersion(platform.release())
            case _:
                return None

class WindowsVersion(OSVersion):
    def __init__(self, version_number: int) -> None:
        self.name = 'Windows'
        self.version_number = version_number

    def __repr__(self) -> str:
        return f"WindowsVersion(version_number={self.version_number})"


class MacOSVersion(OSVersion):
    def __init__(self, version_number: tuple[int, int, int]) -> None:
        self.name = 'macOS'
        self.version_number:tuple[int, int, int] = version_number

    def __repr__(self) -> str:
        return f"MacOSVersion(version_number={self.version_number})"

class LinuxVersion(OSVersion):
    def __init__(self, release_string: str) -> None:
        self.name = 'Linux'
        self.release = release_string

    def __repr__(self) -> str:
        return f"LinuxVersion(release={self.release!r})"