"""Multi-provider dispatcher with auto-discovery and least-loaded balancing.

Supports 'provider/model' routing: if the model string contains a '/',
the prefix is used to select a specific named provider. Without a prefix,
the dispatcher picks the least-loaded healthy provider that serves the model.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from config import settings
from providers import Provider, ProviderConfig, create_provider, parse_model_string

log = logging.getLogger("rlm_proxy.dispatcher")


@dataclass
class ProviderState:
    """Runtime state for a provider instance."""
    provider: Provider
    models: set[str] = field(default_factory=set)
    active_requests: int = 0
    healthy: bool = True
    last_refresh: float = 0.0
    total_requests: int = 0
    total_errors: int = 0


class Dispatcher:
    """Routes model requests to the least-loaded available provider."""

    def __init__(self) -> None:
        self._providers: dict[str, ProviderState] = {}  # keyed by provider name
        self._model_to_providers: dict[str, list[str]] = {}  # model_name -> [provider_names]
        self._lock = asyncio.Lock()
        self._refresh_task: asyncio.Task | None = None

    def _init_from_config(self) -> None:
        """Initialize providers from settings (called before async is available)."""
        if self._providers:
            return

        import json
        from config import CONFIG_FILE
        provider_configs = []

        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
                if "providers" in data:
                    for p in data["providers"]:
                        provider_configs.append(ProviderConfig(**p))
            except Exception:
                pass

        # Fall back to env-based config
        if not provider_configs:
            hosts_str = settings.ollama_hosts.strip()
            if hosts_str:
                urls = [h.strip().rstrip("/") for h in hosts_str.split(",") if h.strip()]
                for i, url in enumerate(urls):
                    name = f"ollama{i + 1}" if len(urls) > 1 else "local"
                    provider_configs.append(ProviderConfig(name=name, api_type="ollama", url=url))
            else:
                provider_configs.append(ProviderConfig(
                    name="local", api_type="ollama",
                    url=settings.ollama_base_url.rstrip("/"),
                ))

        for cfg in provider_configs:
            provider = create_provider(cfg)
            self._providers[cfg.name] = ProviderState(provider=provider)

        log.info(
            "Dispatcher initialized with %d provider(s): %s",
            len(self._providers),
            ", ".join(f"{name}({ps.provider.api_type}:{ps.provider.url})" for name, ps in self._providers.items()),
        )

    @property
    def host_count(self) -> int:
        self._init_from_config()
        return len(self._providers)

    @property
    def is_multi_host(self) -> bool:
        self._init_from_config()
        return len(self._providers) > 1

    def resolve_model(self, model_string: str) -> tuple[str | None, str]:
        """Parse 'provider/model' and return (provider_name, model_name)."""
        return parse_model_string(model_string)

    def pick_provider(self, model_string: str, pin_url: str | None = None) -> tuple[Provider, str]:
        """Pick the best provider for a model string.

        Handles 'provider/model' syntax and auto-dispatch.

        Returns (provider, actual_model_name) — the model name with prefix stripped.
        """
        self._init_from_config()

        # Pin by URL (for backward compat / root_provider_url)
        if pin_url:
            for ps in self._providers.values():
                if ps.provider.url == pin_url:
                    _, model_name = parse_model_string(model_string)
                    return ps.provider, model_name
            raise ValueError(f"Pinned provider URL '{pin_url}' not found")

        provider_name, model_name = parse_model_string(model_string)

        # Explicit provider prefix
        if provider_name:
            if provider_name not in self._providers:
                available = ", ".join(self._providers.keys())
                raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")
            return self._providers[provider_name].provider, model_name

        # Auto-dispatch: find providers that serve this model
        candidate_names = self._model_to_providers.get(model_name, [])

        if not candidate_names:
            # Before first refresh or unknown model — try all healthy
            candidate_names = [n for n, ps in self._providers.items() if ps.healthy]
            if not candidate_names:
                candidate_names = list(self._providers.keys())

        healthy = [n for n in candidate_names if self._providers[n].healthy]
        if not healthy:
            healthy = candidate_names

        if not healthy:
            raise ValueError(f"No provider available for model '{model_name}'")

        best = min(healthy, key=lambda n: self._providers[n].active_requests)
        return self._providers[best].provider, model_name

    # Backward compat
    def pick_host(self, model: str) -> str:
        provider, _ = self.pick_provider(model)
        return provider.url

    async def get_client(self, host_url: str):
        self._init_from_config()
        for ps in self._providers.values():
            if ps.provider.url == host_url:
                return await ps.provider.get_client()
        raise ValueError(f"No provider for URL '{host_url}'")

    def acquire(self, provider_name_or_url: str) -> None:
        self._init_from_config()
        for name, ps in self._providers.items():
            if name == provider_name_or_url or ps.provider.url == provider_name_or_url:
                ps.active_requests += 1
                ps.total_requests += 1
                return

    def release(self, provider_name_or_url: str) -> None:
        self._init_from_config()
        for name, ps in self._providers.items():
            if name == provider_name_or_url or ps.provider.url == provider_name_or_url:
                ps.active_requests = max(0, ps.active_requests - 1)
                return

    def record_error(self, provider_name_or_url: str) -> None:
        self._init_from_config()
        for name, ps in self._providers.items():
            if name == provider_name_or_url or ps.provider.url == provider_name_or_url:
                ps.total_errors += 1
                return

    async def refresh(self) -> None:
        self._init_from_config()
        new_model_map: dict[str, list[str]] = {}

        async def _probe(name: str, ps: ProviderState):
            try:
                models = await ps.provider.list_models()
                model_names = {m.get("name", "") for m in models if m.get("name")}
                ps.models = model_names
                ps.healthy = True
                ps.last_refresh = time.monotonic()
                return name, model_names
            except Exception as exc:
                log.warning("Failed to probe provider '%s': %s", name, exc)
                ps.healthy = False
                ps.last_refresh = time.monotonic()
                return name, set()

        results = await asyncio.gather(*[_probe(n, ps) for n, ps in self._providers.items()])

        async with self._lock:
            for name, model_names in results:
                for model in model_names:
                    new_model_map.setdefault(model, []).append(name)
            self._model_to_providers = new_model_map

        healthy = sum(1 for ps in self._providers.values() if ps.healthy)
        log.info(
            "Dispatcher refresh: %d/%d providers healthy, %d models discovered",
            healthy, len(self._providers), len(new_model_map),
        )

    async def start_refresh_loop(self) -> None:
        self._init_from_config()
        await self.refresh()
        if settings.dispatcher_refresh_interval <= 0:
            return

        async def _loop():
            while True:
                await asyncio.sleep(settings.dispatcher_refresh_interval)
                try:
                    await self.refresh()
                except Exception as exc:
                    log.error("Dispatcher refresh failed: %s", exc)

        self._refresh_task = asyncio.create_task(_loop())

    async def stop(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        for ps in self._providers.values():
            await ps.provider.close()

    def all_models(self) -> list[dict]:
        """Return deduplicated model list with provider info."""
        self._init_from_config()
        seen = set()
        models = []
        for name, ps in self._providers.items():
            for model in ps.models:
                if model not in seen:
                    seen.add(model)
                    providers = self._model_to_providers.get(model, [])
                    models.append({"name": model, "providers": providers})
        return models

    def snapshot(self) -> dict:
        self._init_from_config()
        providers = {}
        for name, ps in self._providers.items():
            providers[name] = {
                "api_type": ps.provider.api_type,
                "url": ps.provider.url,
                "healthy": ps.healthy,
                "models": len(ps.models),
                "active_requests": ps.active_requests,
                "total_requests": ps.total_requests,
                "total_errors": ps.total_errors,
            }

        return {
            "provider_count": len(self._providers),
            "providers": providers,
            "total_models": len(self._model_to_providers),
            "routing_table": dict(self._model_to_providers),
        }

    async def add_provider(self, config: ProviderConfig) -> dict:
        self._init_from_config()
        if config.name in self._providers:
            return {"error": f"Provider '{config.name}' already exists"}

        provider = create_provider(config)
        ps = ProviderState(provider=provider)
        self._providers[config.name] = ps
        log.info("Provider added: %s (%s @ %s)", config.name, config.api_type, config.url)

        try:
            models = await provider.list_models()
            model_names = {m.get("name", "") for m in models if m.get("name")}
            ps.models = model_names
            ps.healthy = True
            ps.last_refresh = time.monotonic()

            async with self._lock:
                for model in model_names:
                    self._model_to_providers.setdefault(model, []).append(config.name)

            log.info("Provider '%s': %d models discovered", config.name, len(model_names))
        except Exception as exc:
            ps.healthy = False
            log.warning("Provider '%s' added but probe failed: %s", config.name, exc)

        return {
            "name": config.name, "api_type": config.api_type, "url": config.url,
            "healthy": ps.healthy, "models": len(ps.models),
        }

    async def remove_provider(self, name: str) -> dict:
        self._init_from_config()
        if name not in self._providers:
            # Try matching by URL for backward compat
            for pname, ps in self._providers.items():
                if ps.provider.url == name:
                    name = pname
                    break
            else:
                return {"error": f"Provider '{name}' not found"}

        if len(self._providers) <= 1:
            return {"error": "Cannot remove the last provider"}

        ps = self._providers.pop(name)
        await ps.provider.close()

        async with self._lock:
            for model in list(self._model_to_providers.keys()):
                if name in self._model_to_providers[model]:
                    self._model_to_providers[model].remove(name)
                if not self._model_to_providers[model]:
                    del self._model_to_providers[model]

        log.info("Provider removed: %s", name)
        return {"removed": name}

    # Backward compat aliases
    async def add_host(self, url: str) -> dict:
        name = f"ollama-{len(self._providers) + 1}"
        return await self.add_provider(ProviderConfig(name=name, api_type="ollama", url=url))

    async def remove_host(self, url: str) -> dict:
        return await self.remove_provider(url)

    def get_provider_configs(self) -> list[dict]:
        self._init_from_config()
        return [ps.provider.config.to_dict() for ps in self._providers.values()]


# Singleton
dispatcher = Dispatcher()
