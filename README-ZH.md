<p align="center">
  <a href="https://asqav.com">
    <img src="https://asqav.com/logo-text-white.png" alt="asqav" width="200">
  </a>
</p>
<p align="center">
  AI Agent 治理平台 - 审计追踪、策略执行、合规报告
</p>
<p align="center">
  <a href="https://github.com/jagmarques/asqav-sdk">English</a> | 中文文档
</p>

# asqav SDK

轻量级 Python SDK，用于 AI Agent 治理。所有 ML-DSA 密码学运算在服务端执行，零原生依赖。

## 安装

```bash
pip install asqav
```

```python
import asqav

asqav.init(api_key="sk_...")
agent = asqav.Agent.create("my-agent")
sig = agent.sign("api:call", {"model": "gpt-4"})
```

每个 Agent 操作都会获得量子安全的密码学签名（ML-DSA-65，FIPS 204）。

## 为什么需要

| 没有治理 | 使用 asqav |
|---|---|
| 无法追踪 Agent 行为 | 每个操作都有 ML-DSA 签名 |
| Agent 可以执行任何操作 | 策略实时阻止危险操作 |
| 单人审批所有操作 | 多方授权关键操作 |
| 手动合规报告 | 自动生成 EU AI Act、DORA 报告 |
| 量子计算威胁 | 从第一天起就是量子安全的 |

## 框架集成

```bash
pip install asqav[langchain]
pip install asqav[crewai]
pip install asqav[litellm]
pip install asqav[haystack]
pip install asqav[openai-agents]
pip install asqav[llamaindex]
pip install asqav[smolagents]
pip install asqav[dspy]
pip install asqav-pydantic          # PydanticAI
```

### LangChain

```python
from asqav.extras.langchain import AsqavCallbackHandler

handler = AsqavCallbackHandler(api_key="sk_...")
chain.invoke(input, config={"callbacks": [handler]})
```

### CrewAI

```python
from asqav.extras.crewai import AsqavCrewHook

hook = AsqavCrewHook(api_key="sk_...")
task = Task(description="研究竞争对手", callbacks=[hook.task_callback])
```

## 生态系统

| 项目 | 说明 |
|------|------|
| [asqav](https://pypi.org/project/asqav/) | Python SDK |
| [asqav-mcp](https://github.com/jagmarques/asqav-mcp) | Claude Desktop MCP 服务器 |
| [asqav-compliance](https://github.com/jagmarques/asqav-compliance) | CI/CD 合规扫描器 |

## 免费套餐

免费开始使用。包含 Agent 创建、签名操作、审计导出和框架集成。

## 许可证

MIT
