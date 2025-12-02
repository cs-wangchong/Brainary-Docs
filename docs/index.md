# Welcome to Brainary Documentation

Brainary is a programmable intelligence kernel that transforms how you build AI systems. Instead of chaining prompts through black-box agents, you compose cognitive primitives—`perceive`, `think`, `analyze`, `plan`, `decide`, `remember`—inside a transparent execution loop with working memory, semantic memory, and metacognition built in.

## What is Brainary?

Brainary is a **cognitive kernel** that routes every primitive call through experience cache, knowledge rules, heuristics, and a semantic fallback to pick the best executor (Direct LLM, ReAct, LangGraph, etc.). Think of it as a logic board for reasoning systems: wire primitives together, pick execution modes, and let the kernel learn which strategies work best.

### Key Features

- **Cognitive Primitives**: Built-in operations for perception, thinking, analysis, planning, decision-making, and memory
- **Execution Contexts**: Program metadata (domain, quality target, token budget, execution mode) guides kernel behavior
- **Memory Tiers**: Working memory for short-term context (7±2 slots) and optional semantic memory for durable knowledge
- **Template Agents**: Building blocks that wrap custom `process(...)` with persistent memory and metacognition
- **Transparent Execution**: Full visibility into reasoning processes and strategy selection

## Getting Started

Ready to dive in? Here's how to get started with Brainary:

1. **[Quickstart Guide](QUICKSTART.md)** - Get up and running in minutes with installation instructions and basic examples
2. **[User Manual](USER_MANUAL.md)** - Learn how to use Brainary's features and capabilities
3. **[API Reference](API_REFERENCE.md)** - Detailed API documentation for all primitives and methods

## Quick Example

```python
from brainary.sdk import Brainary

brain = Brainary(
    enable_learning=True,
    memory_capacity=7,
    quality_threshold=0.85,
)

# Use cognitive primitives
result = brain.think("How can I reduce cold-start latency?")
print(result.content)
print(result.confidence.overall)

# Store and retrieve memories
brain.remember(
    content="Paris is the capital of France",
    importance=0.8,
    tags=["geography", "capital"],
)
```

## Documentation Structure

- **[Quickstart](QUICKSTART.md)**: Installation, configuration, and your first Brainary program
- **[User Manual](USER_MANUAL.md)**: Comprehensive guide to using Brainary
- **[API Reference](API_REFERENCE.md)**: Complete API documentation
- **[SDK Guide](SDK_GUIDE.md)**: Advanced SDK usage and patterns
- **[An Example: Java Vulnerability Detection](https://github.com/cs-wangchong/Brainary-JavaVulnDetector/)**: Intelligent Java vulnerability detection powered by Brainary's multi-agent architecture


## Requirements

- Python 3.10 or newer
- Access to an LLM provider (OpenAI, Anthropic, etc.)
- macOS/Linux (Windows via WSL)

## Installation

```bash
pip install brainary
```

Export your LLM API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## Community & Support

- **GitHub**: [cs-wangchong/Brainary](https://github.com/cs-wangchong/Brainary)
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join the conversation in GitHub Discussions

## License

Brainary is open source software. Check the repository for license details.

---

**Ready to build intelligent systems?** Start with the [Quickstart Guide](QUICKSTART.md) →
