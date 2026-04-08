<div align="center">

# ClawScope

**Unified AI Agent Platform**

*Connect any channel. Orchestrate any agent. Extend with any tool.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-287%20passing-brightgreen)](#)
[![MCP](https://img.shields.io/badge/MCP-Client%20%2B%20Server-purple)](https://modelcontextprotocol.io)
[![Status](https://img.shields.io/badge/Status-Alpha%20v0.1.0-orange)](#)

<br/>

```
用户在飞书发消息 → Orchestrator 理解意图 → 动态调度 Agent → 实时回传进度
```

</div>

---

## 它能做什么

用户发来一条消息，ClawScope 的主控 Orchestrator 会自动分析意图、调度 Agent、实时反馈进度：

```
用户:  帮我分析这份合同，找出风险点，然后写一份摘要邮件

     ⚙️  [legal_analyst]  正在分析合同条款…
     ✓   [legal_analyst]  发现 3 处风险条款
     ⚙️  [writer]         正在起草摘要邮件…
     ✓   [writer]         邮件草稿完成

     ────────────────────────────────────
     [ legal_analyst → writer ]

     您好，以下是本合同的风险摘要…
```

用户还可以用**自然语言**直接管理 Agent：

```
用户:  给 legal_analyst 换成更保守的分析风格
用户:  再加一个翻译 Agent，输出英文版本
用户:  把 writer 删掉，我自己来写邮件
```

---

## 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ClawScope Platform                          │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│   Channels   │   MCP 生态   │  Agent 编排  │      基础设施          │
├──────────────┼──────────────┼──────────────┼────────────────────────┤
│ 飞书  钉钉   │ MCPClient    │Orchestrator  │ MessageBus             │
│ Telegram     │     ↕        │ ├ ask_agent  │ SessionRouter          │
│ Discord      │ MCPServer    │ ├ spawn_agent│ ToolRegistry           │
│ Slack        │ MCPSkill     │ ├ create_    │ SkillRegistry          │
│ CLI / HTTP   │              │ └ list_      │ Memory / RAG           │
└──────────────┴──────────────┴──────────────┴────────────────────────┘
         所有渠道通过统一 MessageBus 与 Agent 层完全解耦
```

### 消息流

```
IM 渠道
  │  InboundMessage
  ▼
MessageBus
  │
  ▼
SessionRouter
  │
  ▼
OrchestratorAgent  ←─ LLM 驱动 ReAct 循环
  ├── ask_researcher("搜索 X")           →  ResearcherAgent
  ├── spawn_agent("你是法律专家", task)  →  临时 Agent（用完即弃）
  └── create_agent("translator", role)   →  注册为持久子 Agent
  │  OutboundMessage（含实时进度）
  ▼
MessageBus
  │
  ▼
IM 渠道
```

---

## 快速开始

### 安装

```bash
pip install clawscope

# 含 MCP 支持
pip install clawscope[mcp]

# 含飞书/钉钉
pip install clawscope[feishu,dingtalk]

# 全部
pip install clawscope[all]
```

### 5 分钟跑起来

```python
import asyncio
from clawscope import ClawScope
from clawscope.agent import ReActAgent
from clawscope.memory import InMemoryMemory

async def main():
    app = ClawScope.create(
        model_provider="anthropic",
        default_model="claude-sonnet-4-6",
    )

    # 预注册子 Agent（Orchestrator 也能在对话中动态创建）
    researcher = ReActAgent(
        name="researcher",
        sys_prompt="你是一名深度研究专家，擅长信息搜集和分析。",
        model=app.model_registry.get_model(),
        memory=InMemoryMemory(),
    )
    app.register_sub_agent(researcher)

    await app.start()
    await app.run_forever()

asyncio.run(main())
```

### 接入飞书

```yaml
# config.yaml
model:
  provider: anthropic
  api_key: ${ANTHROPIC_API_KEY}
  default_model: claude-sonnet-4-6

channels:
  feishu:
    enabled: true
    app_id: ${FEISHU_APP_ID}
    app_secret: ${FEISHU_APP_SECRET}
    use_websocket: true   # 长连接，无需公网 IP
```

```bash
clawscope --config config.yaml
```

### 接入 MCP 工具服务器

```python
from clawscope.mcp import MCPClient, StdioServerConfig

client = MCPClient(StdioServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
    name="filesystem",
))
await client.connect()
await client.register_tools(app.tool_registry)
# Agent 现在可直接使用 filesystem_read_file、filesystem_write_file 等工具
```

### 把 ClawScope 暴露给 Claude Desktop

```bash
clawscope-mcp
```

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "clawscope": {
      "command": "clawscope-mcp"
    }
  }
}
```

---

## Agent 动态编排

Orchestrator 的 LLM 在对话过程中可以自由调度和管理 Agent：

| 工具 | 描述 | 生命周期 |
|------|------|----------|
| `ask_<name>(message)` | 调用预注册子 Agent | 持久 |
| `spawn_agent(role, task)` | 创建临时专家，用完即弃 | 单次任务 |
| `create_agent(name, role)` | 创建并注册，后续可复用 | 会话内持久 |
| `update_agent(name, new_role)` | 用自然语言修改 Agent 角色 | — |
| `remove_agent(name)` | 注销 Agent | — |
| `list_agents()` | 查看当前所有已注册 Agent | — |

**示例对话：**

```
用户: 我需要一个写技术文档的 Agent

  → create_agent("tech_writer", "你是资深技术文档工程师…")
  ✅ Agent 'tech_writer' 已创建，可用 ask_tech_writer 调用

用户: 风格改成更口语化

  → update_agent("tech_writer", "…风格轻松口语…")
  ✅ 已更新

用户: 帮我写一篇 Redis 入门教程

  ⚙️  [tech_writer]  撰写中…
  ✓   [tech_writer]  完成

  [ tech_writer ]

  # Redis 入门：5 分钟搞懂缓存…
```

---

## MCP 双向集成

```
外部 MCP 服务器                ClawScope                  MCP 客户端
(filesystem/DB/API)                                  (Claude Desktop/Cursor)

  MCP Server  ──→  MCPClient                MCPServer  ←──  Claude Desktop
                       ↓ 导入工具                ↑ 暴露工具
                  ToolRegistry           ToolRegistry + Skills
                       ↓
                  OrchestratorAgent
```

**MCPSkill：把 MCP 工具包装成触发词 Skill**

```python
from clawscope.mcp import MCPClient, MCPSkillBundle, StdioServerConfig

client = MCPClient(StdioServerConfig(command="npx", args=["-y", "@mcp/server-filesystem"]))
await client.connect()

bundle = await MCPSkillBundle.from_client(client)
await bundle.register_all(skill_registry)
# 触发词匹配自动路由，无需消耗推理 token
```

---

## 渠道支持

| 渠道 | 状态 | 备注 |
|------|------|------|
| 飞书 (Feishu) | ✅ 完整 | WebSocket 长连接，无需公网 IP |
| 钉钉 (DingTalk) | ✅ 完整 | Stream 协议，LRU 回调队列 |
| HTTP / CLI | ✅ 完整 | FastAPI + SSE 流式输出 |
| Telegram | 🔧 框架就位 | SDK 接入待完成 |
| Discord | 🔧 框架就位 | SDK 接入待完成 |
| Slack | 🔧 框架就位 | SDK 接入待完成 |

---

## 项目结构

```
clawscope/
├── agent/
│   ├── orchestrator.py   # 主控 Agent，动态编排入口
│   ├── react.py          # ReActAgent（工具调用循环）
│   └── a2a.py            # Agent-to-Agent 路由
├── channels/
│   ├── feishu.py         # 飞书 WebSocket + REST
│   └── dingtalk.py       # 钉钉 Stream
├── mcp/
│   ├── client.py         # 连接外部 MCP 服务器
│   ├── server.py         # 暴露为 MCP 服务器
│   └── skill.py          # MCP 工具 → Skill
├── orchestration/
│   ├── chatroom.py       # 多 Agent 辩论室（4 种发言策略）
│   ├── msghub.py         # 广播式协作
│   └── pipeline.py       # 串行 / 并行 Pipeline
├── bus/                  # MessageBus 解耦层
├── memory/               # 会话 + 长期记忆
├── tool/                 # ToolRegistry + 内置工具
├── skills/               # SkillRegistry + Marketplace
├── rag/                  # 知识库 + 向量检索
└── server.py             # FastAPI HTTP + SSE
```

---

## 开发进度

```
核心架构          ████████████████████  完成
MCP 集成          ████████████████████  完成
飞书 / 钉钉       ████████████████████  完成
OrchestratorAgent ████████████████████  完成
RAG               ████████░░░░░░░░░░░░  框架完成，待接入 Agent 工具链
持久化记忆        ████████░░░░░░░░░░░░  JSONL 完成，Redis/SQLite 待实现
Telegram 等       ████░░░░░░░░░░░░░░░░  框架就位，SDK 待填充
测试覆盖          ████████████████████  287 cases，全绿
```

---

## 致谢

- [AgentScope](https://github.com/modelscope/agentscope) — Alibaba DAMO Academy
- [Model Context Protocol](https://modelcontextprotocol.io) — Anthropic
- [Nanobot](https://github.com/HKUDS/nanobot) — HKUDS

---

<div align="center">

MIT License · [Issues](https://github.com/Jane-o-O-o-O/ClawScope/issues) · [Discussions](https://github.com/Jane-o-O-o-O/ClawScope/discussions)

</div>
