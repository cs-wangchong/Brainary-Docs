# Brainary SDK Quickstart


```python
from brainary import BrainaryClient

## 1. Requirements

- Python 3.10 or newer
- Access to an LLM provider supported by your kernel configuration
- macOS/Linux (Windows works via WSL)
  
> **Note**: The package is not yet published to PyPI. Install from source.

## 2. Install & Configure

```bash
pip install brainary
```

Export at least one LLM key before running examples:

```bash
export OPENAI_API_KEY="sk-..."
```

## 3. Hello Brainary (Client API)

```python
from brainary.sdk import Brainary

brain = Brainary(
    enable_learning=True,
    memory_capacity=7,
    quality_threshold=0.85,
)

result = brain.think("How can I reduce cold-start latency?")
print(result.content)
print(result.confidence.overall)
```

### Remember & Recall

```python
brain.remember(
    content="Paris is the capital of France",
    importance=0.8,
    tags=["geography", "capital"],
)

memories = brain.recall(query="capital of France", limit=3)
for item in memories.content:
    print(item["summary"])
```

### Switching Execution Contexts

```python
from brainary.core.context import ExecutionMode

with brain.context(domain="security", quality=0.92, mode="deep"):
    audit = brain.analyze(
        "Scan this codebase for auth issues",
        analysis_type="security",
    )
    print(audit.metadata["issues_found"])
```

## 4. Function-Based Primitives

Prefer simple, stateless helpers? Import primitives directly—they share a
singleton `Brainary` client internally.

```python
from brainary.sdk import configure, think, analyze, plan

configure(quality_threshold=0.9, memory_capacity=9)

think("Why do teams adopt feature flags?")
plan("Roll out observability", constraints=["separate staging"],)
analyze("Review this PR", analysis_type="code")
```

Available primitives mirror the kernel: `perceive`, `think`, `remember`,
`recall`, `associate`, `analyze`, `solve`, `plan`, `decide`, `create`,
`evaluate`, `introspect`, `self_assess`, `select_strategy`, and more.

## 5. Template Agents

Use `TemplateAgent` for reusable, metacognitive agents with scoped memories.

```python
from brainary.sdk.template_agent import TemplateAgent
from brainary.primitive.base import PrimitiveResult

class ResearchAgent(TemplateAgent):
    def process(self, input_data, context, **kwargs) -> PrimitiveResult:
        outline = self.kernel.execute(
            "plan", context=context, goal=input_data, constraints=["short"]
        )
        return self.kernel.execute(
            "synthesize", context=context, components=[outline.content]
        )

agent = ResearchAgent(name="analyst", domain="strategy")
report = agent.run("Summarize the state of LLM tooling")
print(report.content)
```

Template agents automatically:

- Provision working and semantic memory scoped to the kernel
- Propagate execution context (domain, quality, token budgets)
- Integrate metacognitive monitoring (enable/disable per config)

## 6. Diagnostics & Observability

- Call `brain.get_stats()` after batches to inspect success rates, token
  budgets, and routing heuristics.
- Use `brain.get_learning_insights()` to retrieve cache hits, rule updates,
  and strategy recommendations.
- When testing template agents, track `agent.get_stats()` for per-run success
  rates and memory usage.

## 7. What’s Next?

- Read `docs/SDK_GUIDE.md` for architectural details and advanced usage.
- Browse `docs/API_REFERENCE.md` for exhaustive method and primitive docs.
- Explore `examples/` for runnable demos covering teams, semantic memory, and
  control primitives.

Happy building!
    content = f.read()

# Analyze
analysis = client.analyze(
    content=content,
    analysis_type="summary"
)

print("Summary:", analysis["summary"])
print("Key topics:", analysis["topics"])
print("Sentiment:", analysis["sentiment"])
```

### Pattern 3: Problem Solving

```python
# Define problem
problem = "How to optimize database queries?"

# Solve
solution = client.solve(
    problem=problem,
    approach="systematic"
)

print("Solution:", solution.description)
print("Steps:")
for step in solution.steps:
    print(f"  - {step}")
```

### Pattern 4: Learning Assistant

```python
# Learn new information
client.learn(
    content="REST APIs use HTTP methods like GET, POST, PUT, DELETE",
    category="web_development",
    importance=0.9
)

# Associate concepts
associations = client.associate(
    concept="REST API",
    depth=2
)

# Recall related knowledge
knowledge = client.recall(
    query="How do REST APIs work?",
    tags=["web_development"]
)
```

### Pattern 5: Code Analysis

```python
from brainary.domains.security import (
    VulnerabilityDetector,
    PerceiveCode,
    AnalyzeVulnerabilities
)

# Read code
with open("app.py") as f:
    code = f.read()

# Detect vulnerabilities
detector = VulnerabilityDetector()
vulnerabilities = detector.detect_vulnerabilities(code, "app.py")

# Report
for vuln in vulnerabilities:
    print(f"{vuln.cwe_id}: {vuln.description}")
    print(f"  Severity: {vuln.severity}")
    print(f"  Line: {vuln.line_number}")
    print(f"  Fix: {vuln.mitigation}")
```

## Configuration

### Environment Variables

```bash
# LLM API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Brainary Settings
export BRAINARY_LOG_LEVEL="INFO"
export BRAINARY_CACHE_DIR="~/.brainary/cache"
```

### Code Configuration

```python
from brainary import BrainaryClient, LLMConfig

# Custom configuration
config = LLMConfig(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=2000
)

client = BrainaryClient(
    llm_config=config,
    memory_capacity=10,
    enable_cache=True
)
```

## Advanced Features

### Monitoring Performance

```python
# Monitor execution
with client.monitor() as monitor:
    result = client.think("Complex question")

print(f"Time: {monitor.execution_time}ms")
print(f"Tokens: {monitor.tokens_used}")
print(f"Cost: ${monitor.cost:.4f}")
```

### Resource Management

```python
from brainary import ResourceManager, ResourceQuota

manager = ResourceManager()
manager.set_quota(ResourceQuota(
    max_tokens=100000,
    max_cost_usd=5.0
))

# Check before execution
if manager.check_availability(tokens=1000):
    result = client.think("question")
```

### Memory Management

```python
# Check memory status
if client.memory.is_full():
    client.memory.consolidate()

# Clear old memories
client.clear_memory("L1")

# Manual consolidation
client.memory.consolidate(
    from_tier="L1",
    to_tier="L2",
    criteria="importance"
)
```

## Examples by Use Case

### Use Case 1: Customer Support Bot

```python
from brainary import BrainaryClient

client = BrainaryClient()

def handle_customer_query(query: str) -> str:
    # Recall similar past interactions
    similar = client.recall(
        query=query,
        tags=["customer_support"],
        top_k=3
    )
    
    # Think with context
    response = client.think(
        f"Query: {query}\nSimilar cases: {similar}"
    )
    
    # Remember this interaction
    client.remember(
        f"Q: {query}\nA: {response}",
        tags=["customer_support"],
        importance=0.7
    )
    
    return response

# Usage
response = handle_customer_query("How do I reset my password?")
print(response)
```

### Use Case 2: Research Assistant

```python
def research_topic(topic: str) -> dict:
    # Plan research
    plan = client.plan(
        goal=f"Research {topic}",
        resources=["internet", "papers"]
    )
    
    # Gather information
    findings = []
    for step in plan.steps:
        result = client.think(step.description)
        findings.append(result)
    
    # Synthesize
    synthesis = client.analyze(
        content="\n".join(findings),
        analysis_type="summary"
    )
    
    # Store knowledge
    client.learn(
        content=synthesis["summary"],
        category=topic,
        importance=0.9
    )
    
    return synthesis

# Usage
research = research_topic("quantum computing")
print(research["summary"])
```

### Use Case 3: Code Review Assistant

```python
from brainary.domains.security import VulnerabilityDetector

def review_code(code: str, file_path: str) -> dict:
    detector = VulnerabilityDetector()
    
    # Detect vulnerabilities
    vulnerabilities = detector.detect_vulnerabilities(code, file_path)
    
    # Generate summary
    summary = detector.generate_summary(vulnerabilities)
    
    # Prioritize critical issues
    critical = [v for v in vulnerabilities if v.severity == "CRITICAL"]
    
    return {
        "total": summary["total_vulnerabilities"],
        "critical": len(critical),
        "summary": summary["summary"],
        "issues": [v.to_dict() for v in critical]
    }

# Usage
with open("app.py") as f:
    code = f.read()

review = review_code(code, "app.py")
print(f"Found {review['total']} issues, {review['critical']} critical")
```

### Use Case 4: Educational Tutor

```python
def tutor_session(topic: str, question: str) -> dict:
    # Recall relevant knowledge
    knowledge = client.recall(
        query=topic,
        tags=["education", topic],
        top_k=5
    )
    
    # Explain concept
    explanation = client.think(
        f"Explain {topic} to a student who asks: {question}",
        depth="deep"
    )
    
    # Reflect on explanation quality
    reflection = client.reflect(
        target=explanation,
        criteria=["clarity", "completeness", "accuracy"]
    )
    
    # Store for future
    client.remember(
        f"Topic: {topic}\nQ: {question}\nA: {explanation}",
        tags=["education", topic],
        importance=0.8
    )
    
    return {
        "explanation": explanation,
        "quality": reflection["scores"],
        "related_topics": [k.content for k in knowledge[:3]]
    }

# Usage
session = tutor_session("Python", "What are decorators?")
print(session["explanation"])
```

### Use Case 5: Task Automation

```python
def automate_task(task_description: str) -> dict:
    # Plan the task
    plan = client.plan(
        goal=task_description,
        resources=["scripts", "APIs"]
    )
    
    # Execute steps
    results = []
    for step in plan.steps:
        # Act on each step
        result = client.act(
            action_type="execute",
            prompt=step.description
        )
        results.append(result)
        
        # Monitor progress
        client.remember(
            f"Step {step.order}: {step.description} - {result}",
            tags=["automation", "progress"]
        )
    
    # Generate report
    report = client.analyze(
        content="\n".join(str(r) for r in results),
        analysis_type="summary"
    )
    
    return {
        "plan": plan,
        "results": results,
        "report": report
    }

# Usage
automation = automate_task("Set up CI/CD pipeline")
print(automation["report"]["summary"])
```

## Testing

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

def test_memory(client):
    # Store
    client.remember("test content", tags=["test"])
    
    # Retrieve
    memories = client.recall(tags=["test"])
    assert len(memories) > 0
    assert "test content" in str(memories[0].content)
```

## Troubleshooting

### Issue: API Key Not Found

```python
# Solution 1: Environment variable
import os
os.environ["OPENAI_API_KEY"] = "your-key"

# Solution 2: Direct configuration
config = LLMConfig(api_key="your-key")
client = BrainaryClient(llm_config=config)
```

### Issue: Out of Memory

```python
# Check memory status
if client.memory.is_full():
    # Consolidate
    client.memory.consolidate()
    
    # Or clear
    client.clear_memory("L1")

# Increase capacity
client = BrainaryClient(memory_capacity=15)
```

### Issue: High Costs

```python
# Use cheaper models
config = LLMConfig(
    provider="openai",
    model="gpt-3.5-turbo"  # Cheaper than gpt-4
)

# Set budget
manager = ResourceManager()
manager.set_quota(ResourceQuota(max_cost_usd=1.0))

# Monitor costs
stats = manager.get_usage_stats()
print(f"Cost: ${stats['cost_used']:.4f}")
```

## Next Steps

1. **Read the [User Manual](USER_MANUAL.md)** for comprehensive documentation
2. **Check the [API Reference](API_REFERENCE.md)** for detailed API documentation

---

*Happy building with Brainary! 🧠*
