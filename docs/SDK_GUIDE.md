# Brainary SDK Guide

This guide explains how to build applications with the modern Brainary SDK
(`brainary.sdk`). It complements the API reference by focusing on
problem-solving workflows, recommended patterns, and architecture highlights.

## Contents

1. [Architecture & Concepts](#architecture--concepts)
2. [Installation & Environment](#installation--environment)
3. [Brainary Client Basics](#brainary-client-basics)
4. [Contexts & Memory](#contexts--memory)
5. [Function-Based API](#function-based-api)
6. [Template Agents](#template-agents)
7. [Diagnostics & Troubleshooting](#diagnostics--troubleshooting)

---

## Architecture & Concepts

- **Cognitive Kernel**: The orchestrator that routes each primitive call through
    experience cache, rules, heuristics, and LLM semantic fallbacks.
- **Primitives**: Named skills such as `think`, `analyze`, `plan`, `decide`, and
    metacognitive controls like `introspect` or `self_correct`.
- **Working + Semantic Memory**: Kernel-scoped stores that capture short-term
    context (`WorkingMemory`) and long-term knowledge (`SemanticMemory`).
- **Execution Context**: A structured payload (program name, domain, quality,
    token budget, execution mode) that shapes routing strategies and monitoring.

The SDK exposes these components through two ergonomics layers:

1. `Brainary` client class (imperative, stateful).
2. Function-based primitives (stateless wrappers with a shared singleton).

Template agents build on top of the kernel to package memory, metacognition, and
custom behaviors.

---

## Installation & Environment

```bash
git clone https://github.com/cs-wangchong/Brainary brainary
cd brainary
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set your preferred LLM credentials (OpenAI shown):

```bash
export OPENAI_API_KEY="sk-..."
export BRAINARY_LOG_LEVEL=INFO
```

Run `pytest` or `python run_tests.py` to validate the environment before wiring
it into other systems.

---

## Brainary Client Basics

```python
from brainary.sdk import Brainary

brain = Brainary(
        enable_learning=True,
        memory_capacity=7,
        quality_threshold=0.85,
        domain="support",
)

result = brain.think("Draft an onboarding email for a new SRE")
print(result.content)
```

### Core Operations

| Capability | Method | Example |
|------------|--------|---------|
| Interpret inputs | `perceive` | `brain.perceive(ticket_payload, attention_focus=["risk"] )` |
| Deep reasoning | `think` | `brain.think("How do we reduce MTTR?", reasoning_mode="deep")` |
| Persistent storage | `remember` | `brain.remember(summary, importance=0.9, tags=["runbook"])` |
| Retrieval | `recall` | `brain.recall(query="runbook", limit=3)` |
| Association | `associate` | `brain.associate("indexes", discover_mode=True)` |
| Composite analysis | `analyze` | `brain.analyze(code, analysis_type="security")` |
| Problem solving | `solve` | `brain.solve("Optimize ingestion", constraints=["<50ms"])` |
| Decisions | `decide` | `brain.decide(options=["MySQL", "Postgres"], criteria=["latency"])` |

All calls return `PrimitiveResult` with:

- `content`: Structured payload or textual explanation
- `success`: Boolean outcome flag
- `confidence`: Multi-dimensional confidence scores
- `cost`: Token/latency/accounting details
- `metadata`: Primitive-specific extras (issues found, plan steps, etc.)

### Execution Contexts On-The-Fly

```python
with brain.context(domain="finance", quality=0.95, mode="deep"):
        assessment = brain.analyze(report, analysis_type="risk")

alt = brain.execute(
        "plan",
        goal="Ship Q4 release",
        constraints=["no downtime"],
)
```

Use `brain.execute("plan", ...)` (or other primitive names) when no dedicated
helper exists on the client.

---

## Contexts & Memory

### Manual Context Creation

```python
from brainary.core.context import create_execution_context, ExecutionMode

context = create_execution_context(
        program_name="incident_bot",
        domain="reliability",
        execution_mode=ExecutionMode.SYSTEM2,
        quality_threshold=0.9,
        token_budget=6_000,
)

brain.think("Prioritize this backlog", context=context)
```

### Working Memory Hygiene

- Keep `memory_capacity` between 5 and 9 for best routing behavior.
- Call `brain.clear_memory()` between completely unrelated workloads to avoid
    cross-talk.
- Inspect `brain.get_stats()` to monitor success rate and memory utilization:

```python
stats = brain.get_stats()
print(stats["client"]["executions"], stats["kernel"]["success_rate"])
```

---

## Function-Based API

The module `brainary.sdk.primitives` exposes stateless helpers. Ideal for short
scripts, notebooks, or when you only need one primitive.

```python
from brainary.sdk import configure, think, analyze, plan

configure(memory_capacity=9, quality_threshold=0.9)

result = think("Why invest in automatic remediation?")
analysis = analyze(repo_text, analysis_type="security")
roadmap = plan("De-risk launch", constraints=["red team review"])
```

Behind the scenes a singleton `Brainary` client is lazily created. Invoke
`clear_memory()` to reset its working memory or `get_stats()` for telemetry.

---

## Template Agents

Template agents package kernel, memories, and metacognitive monitoring into an
extensible base class.

```python
from brainary.sdk.template_agent import TemplateAgent
from brainary.primitive.base import PrimitiveResult

class ComplianceAgent(TemplateAgent):
        def process(self, input_data, context, **kwargs) -> PrimitiveResult:
                findings = self.kernel.execute(
                        "analyze", context=context, data=input_data, analysis_type="compliance"
                )
                return self.kernel.execute(
                        "reflect", context=context, target=findings.content
                )

agent = ComplianceAgent(name="auditor", domain="governance")
report = agent.run("Audit this vendor questionnaire")
print(report.content)
```

Tips:

- Use `AgentConfig` to tune working-memory capacity, semantic memory, learning,
    and monitoring level per agent instance.
- Share an agent across tasks to accumulate knowledge in semantic memory.
- Call `agent.get_stats()` for run counts, success rates, and memory metrics.

---

## Diagnostics & Troubleshooting

- **Token/Latency Drift**: Check `brain.get_stats()` and adjust
    `quality_threshold` or execution mode. Lowering quality often reduces routing
    through deep (System 2) executors.
- **Noisy Results**: Call `brain.clear_memory()` to remove stale context or
    reduce `memory_capacity` to tighten focus.
- **Custom Primitive Errors**: Wrap calls with `brain.execute("name", ...)` and
    inspect `PrimitiveResult.error` for stack traces.
- **Template Agent Failures**: Review application logs emitted from
    `TemplateAgent.run()` (per-agent logger) and inspect `agent.get_stats()`.

When in doubt, enable DEBUG logging (`BRAINARY_LOG_LEVEL=DEBUG`) to capture full
kernel traces, routing decisions, and LLM payload summaries.
### BrainaryClient Class

The main interface for SDK operations.

#### Initialization

```python
from brainary import BrainaryClient

client = BrainaryClient(
    # LLM Configuration
    llm_provider="openai",
    model="gpt-4o-mini",
    api_key=None,  # Uses env variable if None
    temperature=0.7,
    
    # Memory Configuration
    memory_capacity=7,  # L1 capacity
    l2_capacity=100,    # L2 capacity
    auto_consolidate=True,
    
    # Performance
    enable_cache=True,
    max_tokens=100000,
    max_cost_usd=10.0,
    
    # Debugging
    debug=False,
    log_level="INFO"
)
```

#### Configuration Methods

```python
# Update LLM configuration
client.configure_llm(
    provider="anthropic",
    model="claude-3-opus"
)

# Set execution mode
client.set_mode("deep")

# Update memory settings
client.configure_memory(
    capacity=10,
    auto_consolidate=True
)

# Set resource limits
client.set_limits(
    max_tokens=50000,
    max_cost_usd=5.0
)
```

#### Status and Monitoring

```python
# Get system status
status = client.get_status()
print(f"Memory usage: {status.memory_usage}")
print(f"Token usage: {status.tokens_used}")
print(f"Cost: ${status.cost_used:.4f}")
print(f"Cache hit rate: {status.cache_hit_rate:.2%}")

# Get statistics
stats = client.get_statistics()
print(f"Total operations: {stats.total_operations}")
print(f"Average latency: {stats.avg_latency_ms}ms")
print(f"Success rate: {stats.success_rate:.2%}")
```

---

## Primitive Operations

### Core Primitives

#### 1. Perceive

```python
# Perceive text
result = client.perceive(
    content="Text to perceive",
    source="user_input"
)

# Perceive from file
result = client.perceive(
    file_path="document.txt",
    source="file"
)

# Perceive structured data
result = client.perceive(
    content={"key": "value"},
    source="api"
)
```

#### 2. Think

```python
# Simple thinking
answer = client.think("What is 2+2?")

# Deep thinking with reasoning
answer = client.think(
    "Explain quantum computing",
    depth="deep",
    require_reasoning=True
)

# Thinking with context
answer = client.think(
    "Continue the story",
    context=previous_content
)
```

#### 3. Act

```python
# Generate text
text = client.act(
    action_type="generate",
    prompt="Write a poem about AI"
)

# Execute function
result = client.act(
    action_type="execute",
    function=my_function,
    args={"param": "value"}
)

# Transform data
transformed = client.act(
    action_type="transform",
    data=input_data,
    transformation="summarize"
)
```

#### 4. Reflect

```python
# Reflect on result
reflection = client.reflect(
    target=result,
    criteria=["accuracy", "completeness", "clarity"]
)

print(f"Accuracy: {reflection['scores']['accuracy']}")
print(f"Issues: {reflection['issues']}")
print(f"Suggestions: {reflection['suggestions']}")
```

#### 5. Remember

```python
# Simple remember
client.remember("Paris is the capital of France")

# Remember with metadata
client.remember(
    content="Important fact",
    importance=0.9,
    tags=["fact", "important"],
    associations={"related_concept": 0.8}
)

# Remember in specific tier
client.remember(
    content="Long-term knowledge",
    tier="L3",
    category="geography"
)
```

#### 6. Recall

```python
# Recall by tags
memories = client.recall(tags=["important"])

# Recall by query
memories = client.recall(
    query="What did I learn about AI?",
    top_k=5
)

# Recall with filters
memories = client.recall(
    query="recent events",
    tags=["events"],
    min_importance=0.7,
    similarity_threshold=0.8
)
```

### Composite Primitives

#### Analyze

```python
# Sentiment analysis
result = client.analyze(
    content="This product is amazing!",
    analysis_type="sentiment"
)

# Entity extraction
result = client.analyze(
    content="Apple released iPhone in California",
    analysis_type="entities"
)

# Topic modeling
result = client.analyze(
    content=long_document,
    analysis_type="topics"
)

# Summarization
result = client.analyze(
    content=long_document,
    analysis_type="summary",
    max_length=100
)
```

#### Plan

```python
# Create a plan
plan = client.plan(
    goal="Build a web application",
    constraints=["2 weeks", "Python", "small team"],
    resources=["Flask", "PostgreSQL", "AWS"]
)

# Iterate through steps
for step in plan.steps:
    print(f"{step.order}. {step.description}")
    print(f"   Duration: {step.duration}")
    print(f"   Resources: {step.resources}")
    print(f"   Dependencies: {step.dependencies}")
```

#### Solve

```python
# Solve a problem
solution = client.solve(
    problem="How to optimize database performance?",
    approach="systematic"
)

print(f"Solution: {solution.description}")
print(f"Steps: {solution.steps}")
print(f"Expected outcome: {solution.expected_outcome}")
print(f"Confidence: {solution.confidence}")
```

#### Learn

```python
# Learn new information
knowledge_id = client.learn(
    content="REST APIs use HTTP methods",
    category="web_development",
    importance=0.8
)

# Learn from examples
client.learn(
    content=[
        {"input": "2+2", "output": "4"},
        {"input": "3+3", "output": "6"}
    ],
    category="math",
    learning_type="examples"
)
```

#### Associate

```python
# Find associations
associations = client.associate(
    concept="machine learning",
    depth=2,
    min_strength=0.6
)

# Visualize associations
for assoc in associations:
    print(f"{assoc.source} --({assoc.strength})--> {assoc.target}")
    print(f"  Relation: {assoc.relation_type}")
```

---

## Memory System

### Working with Memory

#### Storing Information

```python
# Store with importance
memory_id = client.remember(
    content="Critical information",
    importance=0.95,
    tags=["critical", "security"]
)

# Store with associations
client.remember(
    content="Python is a language",
    associations={
        "programming": 0.9,
        "coding": 0.8,
        "development": 0.7
    }
)

# Store in specific tier
client.remember(
    content="Temporary note",
    tier="L1",  # Working memory
    expire_after=3600  # Expire in 1 hour
)
```

#### Retrieving Information

```python
# Retrieve by tags
memories = client.recall(
    tags=["security"],
    top_k=5
)

# Retrieve by similarity
memories = client.recall(
    query="security best practices",
    similarity_threshold=0.7
)

# Retrieve with attention
memories = client.recall(
    query="important facts",
    attention_weight=1.5,  # Boost importance
    top_k=3
)

# Retrieve from specific tier
memories = client.recall(
    query="recent events",
    tier="L2"  # Episodic memory
)
```

#### Memory Management

```python
# Check memory status
status = client.memory.get_status()
print(f"L1: {status.l1_used}/{status.l1_capacity}")
print(f"L2: {status.l2_used}/{status.l2_capacity}")
print(f"L3: {status.l3_used} items")

# Consolidate memory
if client.memory.is_full("L1"):
    client.memory.consolidate(
        from_tier="L1",
        to_tier="L2",
        criteria="importance"
    )

# Clear memory
client.clear_memory("L1")  # Clear working memory
client.clear_memory()      # Clear all memory
```

### Memory Strategies

#### Episodic Memory

```python
# Store episode
client.memory.store_episode(
    event="User asked about weather",
    context={
        "location": "Paris",
        "time": "morning",
        "user_id": "user123"
    },
    outcome="Provided forecast",
    importance=0.7
)

# Retrieve episodes
episodes = client.memory.retrieve_episodes(
    query="weather questions",
    timeframe="today",
    context_match={"location": "Paris"}
)
```

#### Semantic Memory

```python
# Store knowledge
client.memory.store_knowledge(
    fact="Water boils at 100°C",
    category="physics",
    confidence=1.0,
    source="textbook"
)

# Query knowledge
knowledge = client.memory.query_knowledge(
    topic="water properties",
    min_confidence=0.8
)
```

---

## Advanced Features

### Streaming Responses

```python
# Stream thinking process
for chunk in client.think_stream("Write a long essay"):
    print(chunk, end="", flush=True)

# Stream with callback
def on_chunk(chunk):
    print(chunk, end="")
    
client.think_stream(
    "Long response",
    callback=on_chunk
)
```

### Batch Operations

```python
# Batch thinking
questions = [
    "What is AI?",
    "What is ML?",
    "What is DL?"
]

answers = client.think_batch(questions)

for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a}\n")
```

### Custom Primitives

```python
from brainary.sdk import register_primitive
from brainary.primitive import CorePrimitive

class MyPrimitive(CorePrimitive):
    def __init__(self):
        super().__init__()
        self._name = "my_custom"
    
    def execute(self, context, working_memory, **kwargs):
        # Implementation
        return result

# Register
register_primitive("my_custom", MyPrimitive)

# Use
result = client.execute_primitive("my_custom", param="value")
```

### Plugins

```python
from brainary.sdk import install_plugin

# Install plugin
install_plugin("brainary-plugin-name")

# Use plugin features
client.use_plugin("plugin_name", feature="feature_name")
```

### Event Hooks

```python
# Register hooks
@client.on_event("before_think")
def log_thinking(event):
    print(f"Thinking about: {event.prompt}")

@client.on_event("after_remember")
def track_memory(event):
    print(f"Stored: {event.memory_id}")

@client.on_event("on_error")
def handle_error(event):
    print(f"Error: {event.error}")
```

---

## Best Practices

### 1. Error Handling

```python
from brainary import BrainaryError, ResourceExhaustedError

try:
    result = client.think("question")
except ResourceExhaustedError:
    # Handle resource exhaustion
    client.reset_resources()
    result = client.think("question", mode="fast")
except BrainaryError as e:
    # Handle other errors
    logger.error(f"Error: {e}")
    result = None
```

### 2. Resource Management

```python
# Set budgets
client.set_limits(
    max_tokens=50000,
    max_cost_usd=5.0,
    max_time_seconds=300
)

# Monitor usage
with client.monitor() as monitor:
    result = client.think("question")
    
if monitor.cost > 0.01:
    logger.warning(f"High cost: ${monitor.cost}")
```

### 3. Memory Optimization

```python
# Regular consolidation
if client.memory.utilization("L1") > 0.8:
    client.memory.consolidate()

# Periodic cleanup
client.memory.cleanup(
    older_than_days=30,
    min_importance=0.5
)

# Efficient tagging
client.remember(
    content="data",
    tags=["specific", "category", "subcategory"]
)
```

### 4. Performance Optimization

```python
# Enable caching
client.enable_cache()

# Use appropriate modes
client.think("simple", mode="fast")
client.think("complex", mode="deep")

# Batch operations
results = client.think_batch(questions)

# Async execution
import asyncio
result = await client.think_async("question")
```

### 5. Testing

```python
import pytest
from brainary import BrainaryClient

@pytest.fixture
def client():
    return BrainaryClient(
        llm_provider="mock",
        memory_capacity=5
    )

def test_sdk_operations(client):
    # Test thinking
    result = client.think("test")
    assert result is not None
    
    # Test memory
    client.remember("test", tags=["test"])
    memories = client.recall(tags=["test"])
    assert len(memories) > 0
```

---

## Integration Patterns

### Pattern 1: RESTful API Integration

```python
from flask import Flask, request, jsonify
from brainary import BrainaryClient

app = Flask(__name__)
client = BrainaryClient()

@app.route('/api/think', methods=['POST'])
def think():
    data = request.json
    result = client.think(data['prompt'])
    return jsonify({"result": result})

@app.route('/api/remember', methods=['POST'])
def remember():
    data = request.json
    memory_id = client.remember(
        data['content'],
        importance=data.get('importance', 0.5),
        tags=data.get('tags', [])
    )
    return jsonify({"memory_id": memory_id})
```

### Pattern 2: Async Task Processing

```python
from celery import Celery
from brainary import BrainaryClient

app = Celery('tasks')
client = BrainaryClient()

@app.task
def process_document(document_path):
    with open(document_path) as f:
        content = f.read()
    
    # Analyze
    analysis = client.analyze(content, analysis_type="summary")
    
    # Remember
    client.remember(
        analysis["summary"],
        tags=["document", "summary"],
        importance=0.8
    )
    
    return analysis
```

### Pattern 3: Streaming Chat Interface

```python
import asyncio
from brainary import BrainaryClient

client = BrainaryClient()

async def chat_stream(user_message):
    # Remember user message
    client.remember(
        f"User: {user_message}",
        tags=["conversation"]
    )
    
    # Stream response
    response = ""
    async for chunk in client.think_stream_async(user_message):
        response += chunk
        yield chunk
    
    # Remember assistant response
    client.remember(
        f"Assistant: {response}",
        tags=["conversation"]
    )
```

### Pattern 4: Multi-Agent System

```python
from brainary import BrainaryClient

class Agent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.client = BrainaryClient()
    
    def process(self, task):
        result = self.client.think(
            f"As a {self.role}, {task}"
        )
        return result

# Create agents
researcher = Agent("Alice", "researcher")
writer = Agent("Bob", "writer")
reviewer = Agent("Carol", "reviewer")

# Collaborative work
topic = "AI in healthcare"
research = researcher.process(f"research {topic}")
draft = writer.process(f"write article about {research}")
review = reviewer.process(f"review this article: {draft}")
```

---

## Summary

The Brainary SDK provides:

✅ Simple, intuitive API
✅ Powerful cognitive primitives  
✅ Automatic memory management
✅ Cost optimization
✅ Multiple LLM support
✅ Extensible architecture
✅ Production-ready features

For more information:
- [User Manual](USER_MANUAL.md)
- [API Reference](API_REFERENCE.md)
- [Examples](../examples/)

---

*Last updated: November 21, 2025*
