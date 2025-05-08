"""
PlatformPluginBase: Abstract base class for platform connectors/plugins.
All new platforms should inherit from this for plug-and-play expansion.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class PlatformPluginBase(ABC):
    @abstractmethod
    def fetch_trending(self) -> Dict[str, Any]:
        """Fetch trending topics or content for this platform."""
        pass

    @abstractmethod
    def post_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Post content to this platform."""
        pass

    @abstractmethod
    def fetch_metrics(self, content_id: str) -> Dict[str, Any]:
        """Fetch metrics (engagement, reach, etc.) for a given content ID."""
        pass

    @abstractmethod
    def get_platform_name(self) -> str:
        """Return the name of the platform."""
        pass
