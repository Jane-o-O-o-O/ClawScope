# ClawScope

**Unified AI Agent Platform** - Combining AgentScope and Nanobot capabilities

ClawScope is an enterprise-grade AI agent platform that integrates the production-level agent framework capabilities of AgentScope with the multi-channel chat integration advantages of Nanobot.

## Features

### From AgentScope
- **Production-grade Agent Framework** - ReActAgent, UserAgent, RealtimeAgent, A2AAgent
- **Multi-Agent Orchestration** - Pipeline, MsgHub, ChatRoom
- **RAG Integration** - Document processing, vector stores
- **OpenTelemetry Tracing** - Full observability support
- **Agent Fine-tuning** - Trinity-RFT integration

### From Nanobot
- **Multi-Channel Support** - 13+ chat platforms (Telegram, Discord, Slack, Feishu, DingTalk, etc.)
- **Message Bus Architecture** - Decoupled channel-agent communication
- **Cron & Heartbeat** - Scheduled tasks and proactive wake-up
- **Skills Marketplace** - Extensible skill system
- **30+ LLM Providers** - Via LiteLLM routing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ClawScope Platform                       │
├─────────────────────────────────────────────────────────────┤
│  Presentation Layer (Channels)                               │
│  Telegram | Discord | Slack | Feishu | DingTalk | CLI       │
├─────────────────────────────────────────────────────────────┤
│  MessageBus (Inbound ↔ Outbound)                            │
├─────────────────────────────────────────────────────────────┤
│  Orchestration Layer                                         │
│  Pipeline | MsgHub | SessionRouter | AgentOrchestrator      │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer                                                 │
│  ReActAgent | UserAgent | A2AAgent | RealtimeAgent          │
├─────────────────────────────────────────────────────────────┤
│  Foundation Layer                                            │
│  Model | Memory | Tool | Session | Tracing | RAG | Cron     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Basic installation
pip install clawscope

# With all features
pip install clawscope[all]

# With specific channels
pip install clawscope[telegram,discord,feishu]
```

## Quick Start

```python
import asyncio
from clawscope import ClawScope

async def main():
    # Initialize from config
    app = ClawScope.from_config("config.yaml")

    # Start the platform
    await app.start()

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

```yaml
# config.yaml
model:
  provider: openai
  api_key: ${OPENAI_API_KEY}
  default_model: gpt-4

agent:
  type: react
  max_iterations: 40

channels:
  telegram:
    enabled: true
    bot_token: ${TELEGRAM_BOT_TOKEN}

memory:
  working: in_memory
  session: jsonl
  long_term: memory_md
```

## Documentation

- [Migration Plan](docs/MIGRATION_PLAN.md)
- [API Reference](docs/api/)
- [Channel Guides](docs/guides/channels/)
- [Agent Development](docs/guides/agents/)

## License

MIT License

## Acknowledgments

- [AgentScope](https://github.com/agentscope-ai/agentscope) - Alibaba DAMO Academy
- [Nanobot](https://github.com/HKUDS/nanobot) - HKUDS
