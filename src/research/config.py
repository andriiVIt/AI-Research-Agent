"""Shared provider configuration. Keys never become model input or log fields."""
import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from dotenv import find_dotenv, load_dotenv

# Read a local, git-ignored .env from the working directory. Variables already set in the
# shell take precedence over the file.
load_dotenv(find_dotenv(usecwd=True), override=False)


@dataclass(frozen=True)
class Settings:
    model: str
    base_url: str
    api_key: str = field(repr=False)
    temperature: float = 0.2
    timeout: float = 60.0
    openalex_api_key: str = field(default="", repr=False)
    # The reflection agent's model (same provider and key). Empty here so the dataclass stays
    # valid; require_critic_model() makes it mandatory for every run that uses reflection.
    critic_model: str = ""

    @classmethod
    def load(cls, *, model=None, base_url=None, api_key=None, temperature=0.2, timeout=60.0,
             openalex_api_key=None, critic_model=None):
        model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        settings = cls(
            model=model,
            critic_model=critic_model or os.getenv("CRITIC_MODEL", ""),
            base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=api_key if api_key is not None else os.getenv("LLM_API_KEY", ""),
            temperature=temperature,
            timeout=timeout,
            openalex_api_key=(openalex_api_key if openalex_api_key is not None
                              else os.getenv("OPENALEX_API_KEY")
                              or os.getenv("OPEN_ALEX_API_KEY", "")),
        )
        url = urlsplit(settings.base_url)
        local = url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
        if (not url.hostname or url.username or url.password or url.query or url.fragment
                or not (url.scheme == "https" or local)):
            raise ValueError("Use a plain HTTPS API base URL (HTTP only for localhost).")
        if not settings.api_key:
            raise ValueError("Set LLM_API_KEY before calling the provider.")
        if timeout <= 0 or not 0 <= temperature <= 2:
            raise ValueError("Choose a positive timeout and a temperature between 0 and 2.")
        return settings

    def require_openalex_key(self) -> str:
        """Call this from the OpenAlex client, so only code that uses OpenAlex needs the key."""
        if not self.openalex_api_key:
            raise ValueError("Set OPENALEX_API_KEY before calling OpenAlex.")
        return self.openalex_api_key

    def require_critic_model(self) -> str:
        """Only runs with reflection need a critic model; --no-reflection works without it."""
        if not self.critic_model:
            raise ValueError("Set CRITIC_MODEL (e.g. openai/gpt-oss-20b), or run with "
                             "--no-reflection.")
        return self.critic_model
