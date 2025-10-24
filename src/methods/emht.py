"""Code and config to run emht"""

import dataclasses

import pathlib
import os

from byotrack.implementation.linker import icy_emht


@dataclasses.dataclass
class EMHTConfig(icy_emht.EMHTParameters):
    """Configuration for eMHT algorithm"""

    icy_path = pathlib.Path(os.environ.get("ICY", ""))

    def build(self, _optflow=None, _features=None) -> icy_emht.IcyEMHTLinker:
        return icy_emht.IcyEMHTLinker(self.icy_path, self, timeout=2000)  # Prevent infinite loops
