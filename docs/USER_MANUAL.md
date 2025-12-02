# Brainary User Manual

This manual walks through every major capability shipped in the `brainary.sdk`
package: installation, core workflows, memory strategies, template agents, and
troubleshooting tips. Use it as the practical companion to the API reference.

## Contents

1. [Introduction](#introduction)
2. [Installation & Verification](#installation--verification)
3. [First Workflow](#first-workflow)
4. [Primitives in Practice](#primitives-in-practice)
5. [Memory & Context Management](#memory--context-management)
6. [Template Agents & Knowledge](#template-agents--knowledge)
7. [Resource & Cost Controls](#resource--cost-controls)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

Brainary makes cognitive systems programmable. Instead of hard-coding prompts or
depending on opaque agent loops, you orchestrate *primitives* (perceive, think,
analyze, decide, etc.) over a configurable kernel that understands execution
context, working memory, semantic knowledge, and metacognitive monitors.

Key pillars:

- **Transparent execution** – inspect the `PrimitiveResult` from each call.
- **Composable control** – chain primitives manually or through template agents.
- **Memory-aware reasoning** – keep short-term context and long-term knowledge
    directly in the SDK.

---

## Installation & Verification

```bash
git clone https://github.com/cs-wangchong/Brainary brainary
cd brainary
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Export an API key for your LLM provider (OpenAI shown):

```bash
export OPENAI_API_KEY="sk-..."
export BRAINARY_LOG_LEVEL=INFO
```

Optionally run unit tests to confirm the environment:

```bash
pytest tests -q
```

---

## First Workflow

```python
from brainary.sdk import Brainary

brain = Brainary(quality_threshold=0.85, memory_capacity=7)

problem = "Outline a lightweight incident response playbook"

# 1. Think deeply about the request
draft = brain.think(problem)

# 2. Store the final response for future recall
brain.remember(draft.content, importance=0.9, tags=["runbook", "incident"])

# 3. Retrieve the memory later
memories = brain.recall(query="incident playbook", limit=1)
print(memories.content[0]["summary"])
```

### Inspecting Results

Every primitive returns `PrimitiveResult`:

```python
print(draft.success)
print(draft.confidence.overall)
print(draft.cost.tokens)
print(draft.metadata)  # primitive-specific extras
```

Use these fields to log telemetry or apply acceptance criteria before sending
responses to downstream systems.

---

## Primitives in Practice

### Core Set

| Primitive | Use Case | Brainary method |
|-----------|----------|-----------------|
| `perceive` | Normalize/understand arbitrary input | `brain.perceive(input_data, attention_focus=[...])` |
| `think` | Deep reasoning & synthesis | `brain.think(query, reasoning_mode="deep")` |
| `remember` | Persist context/knowledge | `brain.remember(content, importance=0.8)` |
| `recall` | Retrieve memories | `brain.recall(query="contract terms", limit=5)` |
| `associate` | Link or discover concepts | `brain.associate("kubernetes", discover_mode=True)` |

### Composite & Specialized Primitives

Use `brain.execute(...)` or the function API to access the full primitive
catalog:

```python
plan = brain.execute("plan", goal="Launch Q4 update", constraints=["no downtime"])
review = brain.analyze(code_snippet, analysis_type="security")
decision = brain.decide(options=["Strategy A", "Strategy B"], criteria=["risk", "ROI"])
```

### Function-Based Shortcuts

```python
from brainary.sdk import configure, think, analyze

configure(memory_capacity=9)
think("Why did latency spike yesterday?")
analyze(log_blob, analysis_type="root_cause")
```

Behind the scenes these helpers reuse a singleton `Brainary` client.

---

## Memory & Context Management

### Context Builders

```python
from brainary.core.context import create_execution_context, ExecutionMode

context = create_execution_context(
        program_name="support_bot",
        domain="customer_support",
        execution_mode=ExecutionMode.ADAPTIVE,
        quality_threshold=0.9,
        token_budget=8000,
)

brain.think("Resolve this ticket", context=context)
```

Use `brain.context(...)` as a context manager when you only need overrides for a
few calls.

### Working Memory Hygiene

- Keep `memory_capacity` between 5 and 9 for best attention focus.
- Call `brain.clear_memory()` before switching to a wildly different workload.
- When long-term knowledge is required, switch to template agents and enable
    semantic memory (see below).

### Inspecting Memory Stats

```python
stats = brain.get_stats()
print(stats["kernel"]["success_rate"], stats["kernel"]["memory"])  # depends on kernel payload
```

---

## Template Agents & Knowledge

Template agents offer reusable building blocks with scoped memories and
metacognition baked in.

```python
from brainary.sdk.template_agent import TemplateAgent
from brainary.primitive.base import PrimitiveResult

class SupportAgent(TemplateAgent):
        def process(self, input_data, context, **kwargs) -> PrimitiveResult:
                summary = self.kernel.execute(
                        "think", context=context, question=input_data
                )
                return self.kernel.execute(
                        "introspect", context=context, state=summary.content
                )

agent = SupportAgent(name="triage", domain="support")
agent.remember("Escalations require pager duty", tags=["policy"])
result = agent.run("Handle a Sev2 outage report")
print(result.content)
```

Tips:

- Use `AgentConfig` to adjust working-memory capacity, semantic memory
    activation, monitoring level, and learning preferences per agent.
- Share agent instances for long-running services so their semantic memory grows
    with experience.
- `agent.get_stats()` reveals cumulative runs, success rates, and kernel stats.

---

## Resource & Cost Controls

| Lever | How to adjust |
|-------|---------------|
| Token budget | `Brainary(..., token_budget=6000)` or pass `token_budget` when creating contexts. |
| Quality vs. speed | Set `quality_threshold` (higher = more System 2 routing). |
| Execution mode | Choose `fast/system1`, `deep/system2`, `adaptive`, or `cached` when calling `brain.context(mode="fast")`. |
| Learning system | Toggle `enable_learning` in the client or template agent configuration to skip cache/rule updates. |

Monitor `brain.get_stats()` and `brain.get_learning_insights()` regularly to see
how budgets, cache hits, or heuristics evolve.

---

## Troubleshooting

| Symptom | Resolution |
|---------|------------|
| **Repeated hallucinations** | Reduce `memory_capacity`, clear working memory, or force `mode="deep"` for critical calls. |
| **Slow responses** | Lower `quality_threshold`, set `mode="fast"`, or reduce `token_budget`. |
| **Context bleed** | Call `brain.clear_memory()` or create a fresh `Brainary` instance for isolation. |
| **Template agent crash** | Wrap `agent.run()` in try/except to inspect `PrimitiveResult.error`, then check `agent.get_stats()` and semantic memory contents. |
| **Unknown primitive name** | Use `brain.execute("primitive_name", ...)` and confirm that primitive is registered inside `brainary.primitive`. |

Enable verbose logging for deeper inspection:

```bash
export BRAINARY_LOG_LEVEL=DEBUG
```

The kernel will emit routing decisions, LLM payload statistics, and learning
updates directly to the console or your configured logging sink.
### Prerequisites

- Python 3.8 or higher
- pip package manager

### Basic Installation

```bash
pip install brainary
```

### Development Installation

```bash
git clone https://github.com/your-org/brainary.git
cd brainary
pip install -e .
```

### Verify Installation

```python
import brainary
print(brainary.__version__)
```

---

## Quick Start

### Your First Brainary Program

```python
from brainary import BrainaryClient

# Initialize client
client = BrainaryClient()

# Think about something
result = client.think("What is the capital of France?")
print(result)  # Output: Paris

# Remember information
client.remember("Paris is the capital of France", importance=0.8)

# Recall later
memories = client.recall("capital of France")
print(memories)
```

### Example: Simple Q&A Agent

```python
from brainary import BrainaryClient

# Create client with configuration
client = BrainaryClient(
    llm_provider="openai",
    model="gpt-4o-mini",
    memory_capacity=10
)

# Ask a question
question = "What are the main components of a computer?"
response = client.think(question)

# Remember the answer
client.remember(
    content=f"Q: {question}\nA: {response}",
    importance=0.7,
    tags=["computer", "hardware"]
)

# Recall related information
related = client.recall("computer hardware")
for memory in related:
    print(memory)
```

---

## Core Concepts

### 1. Execution Context

The execution context defines how operations are executed:

```python
from brainary import create_execution_context, ExecutionMode

context = create_execution_context(
    program_name="my_app",
    execution_mode=ExecutionMode.ADAPTIVE,  # FAST, BALANCED, or DEEP
    quality_threshold=0.7,
    criticality=0.8,
    time_pressure=0.3,
    token_budget=10000,
    domain="general"
)
```

**Execution Modes:**
- `FAST`: Prioritize speed, use simpler models
- `BALANCED`: Balance quality and speed
- `DEEP`: Prioritize quality, use advanced reasoning
- `ADAPTIVE`: Dynamically adjust based on context

### 2. Cognitive Kernel

The kernel orchestrates primitive execution:

```python
from brainary import CognitiveKernel, ExecutionContext

kernel = CognitiveKernel()

# Execute a primitive
result = kernel.execute(
    primitive_name="think",
    context=context,
    prompt="Analyze this problem..."
)
```

### 3. Working Memory

Manages short-term information storage:

```python
from brainary import WorkingMemory

memory = WorkingMemory(
    capacity=7,  # L1 capacity (Miller's Law)
    l2_capacity=100  # L2 capacity
)

# Store information
item_id = memory.store(
    content="Important fact",
    importance=0.9,
    tags=["fact", "important"]
)

# Retrieve information
results = memory.retrieve(
    tags=["important"],
    top_k=5
)
```

---

## Working with Primitives

### Core Primitives

Brainary provides 6 core primitives that form the foundation of cognitive operations:

#### 1. Perceive

Gather and parse information from the environment:

```python
from brainary import BrainaryClient

client = BrainaryClient()

# Perceive text
result = client.perceive(
    content="The weather is sunny today.",
    source="user_input"
)

# Perceive from file
result = client.perceive(
    file_path="document.txt",
    source="file"
)
```

#### 2. Think

Perform reasoning and problem-solving:

```python
# Basic thinking
result = client.think(
    prompt="What is 2+2?",
    depth="shallow"  # shallow, medium, or deep
)

# Deep thinking with reasoning
result = client.think(
    prompt="Explain quantum computing",
    depth="deep",
    require_reasoning=True
)
```

#### 3. Act

Execute actions or generate outputs:

```python
# Generate text
result = client.act(
    action_type="generate",
    prompt="Write a haiku about AI"
)

# Execute function
result = client.act(
    action_type="execute",
    function=my_function,
    args={"param": "value"}
)
```

#### 4. Reflect

Evaluate and critique previous results:

```python
# Reflect on thinking
result = client.think("Solution to problem X")

reflection = client.reflect(
    target=result,
    criteria=["correctness", "completeness"]
)
```

#### 5. Recall

Retrieve information from memory:

```python
# Recall by tags
memories = client.recall(
    tags=["important", "facts"],
    top_k=5
)

# Recall by similarity
memories = client.recall(
    query="What did I learn about AI?",
    similarity_threshold=0.7
)
```

#### 6. Associate

Find connections between concepts:

```python
# Find associations
associations = client.associate(
    concept="machine learning",
    depth=2,  # How many hops
    min_strength=0.5
)

for assoc in associations:
    print(f"{assoc.source} -> {assoc.target}: {assoc.strength}")
```

### Composite Primitives

Built from core primitives for complex operations:

#### Analyze

```python
result = client.analyze(
    content="Long article text...",
    analysis_type="sentiment"  # sentiment, entities, topics, summary
)
```

#### Plan

```python
plan = client.plan(
    goal="Build a web application",
    constraints=["2 weeks", "small team"],
    resources=["Python", "Flask"]
)
```

#### Solve

```python
solution = client.solve(
    problem="How to optimize database queries?",
    approach="systematic"  # trial_error, systematic, creative
)
```

#### Learn

```python
client.learn(
    content="New information about topic X",
    category="knowledge",
    importance=0.8
)
```

### Metacognitive Primitives

Higher-order thinking operations:

#### Monitor

```python
# Monitor execution
with client.monitor(rules=[], critiria) as monitor:
    result = client.think("Complex problem")
    client.reflect
    xxx
    remember()

    
print(f"Quality: {monitor.quality_score}")
print(f"Token usage: {monitor.tokens_used}")
```

#### Adapt

```python
# Adapt strategy based on performance
client.adapt(
    feedback={"success": False, "reason": "timeout"},
    adjustment="increase_timeout"
)
```

---

## Memory System

### Three-Tier Architecture

Brainary uses a three-tier memory system inspired by human cognition:

```
L1: Working Memory (7±2 items) - Active reasoning
L2: Episodic Memory (~100 items) - Recent experiences  
L3: Semantic Memory (unlimited) - Long-term knowledge
```

### Working with L1 (Working Memory)

```python
from brainary import WorkingMemory

memory = WorkingMemory(capacity=7)

# Store in L1
item_id = memory.store(
    content={"fact": "value"},
    importance=0.9,
    tags=["current", "active"]
)

# Retrieve from L1
items = memory.retrieve(tags=["active"], top_k=3)

# Check capacity
if memory.is_full():
    memory.consolidate()  # Move to L2
```

### Working with L2 (Episodic Memory)

```python
# Store episode
memory.store_episode(
    content="User asked about weather",
    context={"location": "Paris", "time": "morning"},
    importance=0.7
)

# Retrieve episodes
episodes = memory.retrieve_episodes(
    query="weather questions",
    timeframe="recent"  # recent, today, week, month
)
```

### Working with L3 (Semantic Memory)

```python
# Store knowledge
memory.store_knowledge(
    content="Python is a programming language",
    category="programming",
    confidence=0.95
)

# Query knowledge
knowledge = memory.query_knowledge(
    topic="programming languages",
    min_confidence=0.8
)
```

### Memory Consolidation

```python
# Manual consolidation
memory.consolidate(
    from_tier="L1",
    to_tier="L2",
    criteria="importance"  # importance, recency, frequency
)

# Automatic consolidation
memory = WorkingMemory(
    capacity=7,
    auto_consolidate=True,
    consolidation_threshold=0.8
)
```

### Memory Prefetching

```python
from brainary import PrefetchRequest

# Request prefetch
prefetch = PrefetchRequest(
    tags=["relevant", "important"],
    max_items=5,
    min_importance=0.7
)

memory.prefetch(prefetch)
```

---

## Resource Management

### Resource Allocation

```python
from brainary import ResourceManager, ResourceQuota

# Create resource manager
manager = ResourceManager()

# Set quotas
quota = ResourceQuota(
    max_tokens=100000,
    max_memory_mb=512,
    max_time_seconds=300,
    max_cost_usd=1.0
)

manager.set_quota(quota)

# Check availability
if manager.check_availability(
    tokens=1000,
    memory_mb=50,
    cost_usd=0.01
):
    # Allocate resources
    allocation = manager.allocate(
        tokens=1000,
        memory_mb=50,
        cost_usd=0.01
    )
```

### Monitoring Resource Usage

```python
# Get usage statistics
stats = manager.get_usage_stats()
print(f"Tokens used: {stats['tokens_used']}/{stats['tokens_total']}")
print(f"Cost: ${stats['cost_used']:.4f}")
print(f"Memory: {stats['memory_used_mb']}MB")

# Check if quota exceeded
if manager.is_quota_exceeded():
    print("Resource quota exceeded!")
    manager.reset_quota()
```

### Cost Tracking

```python
from brainary import CostTracker

tracker = CostTracker()

# Track LLM call
tracker.track_llm_call(
    provider="openai",
    model="gpt-4o-mini",
    input_tokens=100,
    output_tokens=200
)

# Get cost report
report = tracker.generate_report()
print(f"Total cost: ${report['total_cost']:.4f}")
print(f"By provider: {report['by_provider']}")
print(f"By model: {report['by_model']}")
```

---

## LLM Integration

### Supported Providers

- OpenAI (GPT-4, GPT-3.5, GPT-4o)
- Anthropic (Claude)
- Google (Gemini)
- Local models (Ollama)

### Configuration

```python
from brainary import LLMConfig

# OpenAI configuration
config = LLMConfig(
    provider="openai",
    api_key="your-api-key",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000
)

client = BrainaryClient(llm_config=config)
```

### Using Multiple Providers

```python
# Route based on task
client = BrainaryClient(
    routing_strategy="adaptive",
    providers={
        "fast": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "balanced": {"provider": "openai", "model": "gpt-4o-mini"},
        "deep": {"provider": "anthropic", "model": "claude-3-opus"}
    }
)

# Fast task
result = client.think("Quick question", mode="fast")

# Deep task
result = client.think("Complex analysis", mode="deep")
```

### Cost Optimization

```python
# Use cheaper models for simple tasks
client = BrainaryClient(
    cost_optimization=True,
    max_cost_per_request=0.01
)

# Automatic model selection based on complexity
result = client.think(
    "Simple arithmetic",
    auto_select_model=True
)
```

### Streaming Responses

```python
# Stream long responses
for chunk in client.think_stream("Write a long essay"):
    print(chunk, end="", flush=True)
```

---

## Advanced Features

### 1. Intelligent Routing

```python
from brainary import Router, RoutingStrategy

router = Router(strategy=RoutingStrategy.ADAPTIVE)

# Route based on complexity
primitive = router.route(
    task="Solve complex problem",
    context=context
)
```

### 2. Experience Cache

```python
from brainary import ExperienceCache

cache = ExperienceCache()

# Cache result
cache.store(
    query="What is 2+2?",
    result="4",
    confidence=1.0
)

# Retrieve from cache
cached = cache.retrieve("What is 2+2?")
if cached:
    print(f"Cached: {cached.result}")
```

### 3. Attention Mechanism

```python
from brainary import AttentionManager

attention = AttentionManager()

# Focus attention
attention.focus(
    items=memory_items,
    query="most important facts",
    top_k=3
)

# Get focused items
focused = attention.get_focused()
```

### 4. Simulation

```python
# Simulate outcomes
from brainary import Simulator

simulator = Simulator()

outcomes = simulator.simulate(
    action="deploy_model",
    scenarios=["success", "failure", "partial"],
    num_simulations=100
)

for outcome in outcomes:
    print(f"{outcome.scenario}: {outcome.probability:.2%}")
```

### 5. Planning with Constraints

```python
from brainary import Planner, Constraint

planner = Planner()

plan = planner.create_plan(
    goal="Launch product",
    constraints=[
        Constraint("time", "max", "3 months"),
        Constraint("budget", "max", 50000),
        Constraint("team_size", "max", 5)
    ],
    optimization_target="time"  # time, cost, quality
)

for step in plan.steps:
    print(f"{step.order}. {step.action} ({step.duration})")
```

---

## Best Practices

### 1. Memory Management

```python
# DO: Set appropriate memory limits
memory = WorkingMemory(capacity=7, l2_capacity=100)

# DO: Tag memories for easy retrieval
memory.store(content="fact", tags=["important", "category"])

# DO: Consolidate regularly
if memory.is_full():
    memory.consolidate()

# DON'T: Store everything in L1
# DON'T: Forget to tag important information
```

### 2. Resource Optimization

```python
# DO: Set resource budgets
manager.set_quota(ResourceQuota(max_tokens=10000))

# DO: Monitor usage
stats = manager.get_usage_stats()

# DO: Use cheaper models when possible
client.think("simple task", mode="fast")

# DON'T: Use expensive models for simple tasks
# DON'T: Ignore resource limits
```

### 3. Error Handling

```python
from brainary import BrainaryError, ResourceExhaustedError

try:
    result = client.think("complex problem")
except ResourceExhaustedError:
    # Handle resource exhaustion
    manager.reset_quota()
    result = client.think("complex problem", mode="fast")
except BrainaryError as e:
    # Handle other errors
    logger.error(f"Brainary error: {e}")
```

### 4. Logging and Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Use monitoring
with client.monitor() as monitor:
    result = client.think("problem")

print(f"Execution time: {monitor.execution_time}ms")
print(f"Tokens used: {monitor.tokens_used}")
print(f"Cost: ${monitor.cost:.4f}")
```

### 5. Testing

```python
import pytest
from brainary import BrainaryClient

@pytest.fixture
def client():
    return BrainaryClient(
        llm_provider="mock",  # Use mock for testing
        memory_capacity=5
    )

def test_thinking(client):
    result = client.think("test question")
    assert result is not None
    assert len(result) > 0
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Problem**: `MemoryError: Working memory capacity exceeded`

**Solution**:
```python
# Increase capacity or consolidate
memory = WorkingMemory(capacity=10)
memory.consolidate()
```

#### 2. Resource Quota Exceeded

**Problem**: `ResourceExhaustedError: Token quota exceeded`

**Solution**:
```python
# Check and reset quota
if manager.is_quota_exceeded():
    manager.reset_quota()
    
# Or increase quota
manager.set_quota(ResourceQuota(max_tokens=200000))
```

#### 3. LLM API Errors

**Problem**: `LLMError: API key invalid`

**Solution**:
```python
# Set API key via environment variable
import os
os.environ["OPENAI_API_KEY"] = "your-key"

# Or pass directly
config = LLMConfig(api_key="your-key")
```

#### 4. Slow Performance

**Problem**: Operations taking too long

**Solution**:
```python
# Use faster execution mode
context = create_execution_context(
    execution_mode=ExecutionMode.FAST
)

# Enable caching
client = BrainaryClient(enable_cache=True)

# Use simpler models
config = LLMConfig(model="gpt-3.5-turbo")
```

#### 5. Memory Retrieval Issues

**Problem**: Can't find stored memories

**Solution**:
```python
# Use more specific tags
memory.store(content="data", tags=["specific", "category", "type"])

# Increase retrieval limit
results = memory.retrieve(tags=["category"], top_k=10)

# Lower similarity threshold
results = memory.retrieve(query="search", similarity_threshold=0.5)
```

### Debug Mode

```python
# Enable comprehensive debugging
client = BrainaryClient(debug=True)

# Check system status
status = client.get_status()
print(f"Memory usage: {status.memory}")
print(f"Active primitives: {status.active_primitives}")
print(f"Cache hit rate: {status.cache_hit_rate}")
```

### Getting Help

- **Documentation**: https://brainary.readthedocs.io
- **GitHub Issues**: https://github.com/your-org/brainary/issues
- **Community**: https://discord.gg/brainary
- **Email**: support@brainary.ai

---

## Appendix

### Configuration Reference

```python
# Complete configuration example
config = {
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 2000,
        "api_key": "your-key"
    },
    "memory": {
        "l1_capacity": 7,
        "l2_capacity": 100,
        "auto_consolidate": True,
        "consolidation_threshold": 0.8
    },
    "resources": {
        "max_tokens": 100000,
        "max_cost_usd": 10.0,
        "max_time_seconds": 300
    },
    "execution": {
        "mode": "adaptive",
        "quality_threshold": 0.7,
        "enable_cache": True,
        "enable_monitoring": True
    }
}

client = BrainaryClient.from_config(config)
```

### Environment Variables

```bash
# LLM Configuration
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# Brainary Configuration
export BRAINARY_LOG_LEVEL="INFO"
export BRAINARY_CACHE_DIR="/path/to/cache"
export BRAINARY_MAX_TOKENS="100000"

# Testing
export BRAINARY_TEST_MODE="1"
```

### Performance Tips

1. **Use caching** for repeated queries
2. **Batch operations** when possible
3. **Choose appropriate execution modes** (FAST vs DEEP)
4. **Monitor resource usage** regularly
5. **Consolidate memory** proactively
6. **Use cheaper models** for simple tasks
7. **Enable streaming** for long responses
8. **Implement retry logic** for API failures

---

*Last updated: November 21, 2025*
*Version: 0.1.0*
