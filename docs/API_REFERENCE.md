# Brainary API Reference

This document covers the public APIs surfaced through `brainary.sdk`. It is
organized by entry point so you can quickly find constructor arguments, method
signatures, and return types.

## Contents

1. [Client API (`Brainary`)](#client-api-brainary)
2. [Context Helpers](#context-helpers)
3. [Function-Based Primitives](#function-based-primitives)
4. [Template Agents](#template-agents)
5. [Memory & Diagnostics](#memory--diagnostics)

---

## Client API (`Brainary`)

```python
from brainary.sdk import Brainary
```

### Constructor

```python
Brainary(
        *,
        enable_learning: bool = True,
        memory_capacity: int = 7,
        quality_threshold: float = 0.8,
        token_budget: int = 10_000,
        program_name: str = "brainary_app",
        **context_overrides,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `enable_learning` | `bool` | Enable kernel learning system (experience cache + rule updates). |
| `memory_capacity` | `int` | Working-memory slot count (≈ Miller’s Law). |
| `quality_threshold` | `float` | Minimum acceptable quality for automatic routing. |
| `token_budget` | `int` | Maximum tokens per primitive execution. |
| `program_name` | `str` | Identifier stamped onto execution contexts/traces. |
| `**context_overrides` | `dict` | Passed to `create_execution_context` (e.g., `domain`, `criticality`). |

Internally the client wires the `CognitiveKernel`, creates a `WorkingMemory`,
and registers core primitives exactly once per process.

### Methods

All methods return `PrimitiveResult` instances unless otherwise stated.

| Method | Signature | Notes |
|--------|-----------|-------|
| `think` | `think(query: str, *, reasoning_mode: str = "adaptive", **kwargs)` | Deep reasoning with automatic routing. `reasoning_mode` accepts `fast`, `deep`, `cached`, `adaptive`. |
| `perceive` | `perceive(input_data: Any, *, attention_focus: Optional[List[str]] = None, **kwargs)` | Parse or summarize arbitrary input payloads. |
| `remember` | `remember(content: Any, *, importance: float = 0.5, tags: Optional[List[str]] = None, **kwargs)` | Stores content in working memory with association building. |
| `recall` | `recall(query: Optional[str] = None, *, tags: Optional[List[str]] = None, limit: int = 5, **kwargs)` | Retrieves items via attention + spreading activation. Results surface in `PrimitiveResult.content`. |
| `associate` | `associate(concept1: str, concept2: Optional[str] = None, *, strength: Optional[float] = None, discover_mode: bool = False, **kwargs)` | Creates or discovers semantic links. |
| `analyze` | `analyze(data: Any, *, analysis_type: str = "general", **kwargs)` | Runs composite reasoning (security review, performance audit, etc.). |
| `solve` | `solve(problem: str, *, constraints: Optional[List[str]] = None, **kwargs)` | Goal-oriented planning / solution synthesis. |
| `decide` | `decide(options: List[Any], *, criteria: Optional[List[str]] = None, **kwargs)` | Multi-criteria decision making. |
| `execute` | `execute(primitive_name: str, **kwargs)` | Escape hatch to call any registered primitive (e.g., `plan`, `create`, `verify`). |
| `get_stats` | `get_stats() -> Dict[str, Any]` | Merges execution counts with kernel statistics (success rate, routing mix, cache hits). |
| `get_learning_insights` | `get_learning_insights() -> Dict[str, Any]` | Summaries from the learning subsystem (updated heuristics, rule proposals). |
| `clear_memory` | `clear_memory() -> None` | Resets L1 working memory while keeping semantic memory intact. |
| `context` | `context(domain: Optional[str] = None, quality: Optional[float] = None, mode: Optional[str] = None, **kwargs)` | Returns a context manager that temporarily overrides execution parameters. |

### Usage Notes

- Every method automatically provisions an `ExecutionContext` unless you pass one
    explicitly.
- `Brainary` is lightweight—create multiple instances if you need isolated
    memories or program identifiers.
- Prefer the method helpers (`think`, `analyze`, etc.) over `execute` unless you
    are experimenting with custom primitives.

---

## Context Helpers

```python
from brainary.core.context import (
        ExecutionContext,
        ExecutionMode,
        create_execution_context,
)
from brainary.sdk.context import ContextBuilder, ContextManager
```

- `create_execution_context(**config)` mirrors the keyword arguments accepted by
    the client constructor (`program_name`, `quality_threshold`, `token_budget`,
    `domain`, `criticality`, etc.).
- `ExecutionMode` enum values: `SYSTEM1` (`fast`), `SYSTEM2` (`deep`), `ADAPTIVE`,
    `CACHED`.
- `ContextBuilder` provides a fluent API for building reusable contexts:

```python
builder = ContextBuilder()
default_context = (
        builder
        .with_program_name("analysis_tool")
        .with_quality(0.9)
        .with_mode("deep")
        .build()
)
```

---

## Function-Based Primitives

Module: `brainary.sdk.primitives`

All functions share a lazily-created global `Brainary` client. Call `configure`
once (optional) to adjust defaults.

### Configuration

```python
from brainary.sdk import configure

configure(enable_learning=True, memory_capacity=9, quality_threshold=0.9)
```

### Core Primitives

| Function | Description |
|----------|-------------|
| `perceive(input_data, attention_focus=None, **kwargs)` | Parse/structure raw inputs. |
| `think(query, reasoning_mode="adaptive", **kwargs)` | Deep reasoning. |
| `remember(content, importance=0.5, tags=None, **kwargs)` | Store information. |
| `recall(query=None, tags=None, limit=5, **kwargs)` | Retrieve memories. |
| `associate(concept1, concept2=None, strength=None, discover_mode=False, **kwargs)` | Link or discover concepts. |
| `action(action_type, parameters=None, **kwargs)` | Execute general-purpose actions. |

### Composite Primitives

| Function | Description |
|----------|-------------|
| `analyze(data, analysis_type="general", **kwargs)` | Multi-step code/document analysis. |
| `solve(problem, constraints=None, **kwargs)` | Constraint-aware solutioning. |
| `plan(goal, constraints=None, **kwargs)` | Structured plans (callable even though `Brainary` client does not expose a dedicated method). |
| `decide(options, criteria=None, **kwargs)` | Multi-criteria decisions. |
| `create`, `decompose`, `synthesize`, `evaluate`, `verify`, `explain` | Higher-level control and reasoning primitives tied to kernel implementations. |

### Metacognitive Primitives

| Function | Description |
|----------|-------------|
| `introspect()` | Inspect current cognitive state. |
| `self_assess(task, performance=None, **kwargs)` | Capability/performance review. |
| `select_strategy(problem, available_strategies=None, **kwargs)` | Choose strategies before executing tasks. |
| `self_correct(error, hypothesis=None, **kwargs)` | Perform corrective actions after failures. |

> Every function returns the same `PrimitiveResult` object as the client methods,
> so metadata (`confidence`, `cost`, `trace`) is preserved.

---

## Template Agents

Module: `brainary.sdk.template_agent`

### `AgentConfig`

```python
AgentConfig(
        name: str,
        description: str = "",
        domain: str = "general",
        working_memory_capacity: int = 10,
        enable_semantic_memory: bool = True,
        enable_metacognition: bool = True,
        monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD,
        enable_learning: bool = True,
        quality_threshold: float = 0.8,
        default_execution_mode: str = "adaptive",
        max_token_budget: int = 10_000,
        metadata: Dict[str, Any] = field(default_factory=dict),
)
```

### `TemplateAgent`

Abstract base class with kernel-scoped memories and metacognitive monitoring.

Key methods:

| Method | Description |
|--------|-------------|
| `process(self, input_data, context, **kwargs)` | Abstract hook—implement your agent’s logic by chaining primitives through `self.kernel.execute`. |
| `run(input_data, context=None, **kwargs)` | Creates/propagates contexts, tracks stats, and calls `process`. |
| `add_knowledge(knowledge)` | Stores conceptual/factual/procedural knowledge in semantic memory. |
| `search_knowledge(query, knowledge_types=None, top_k=5)` | Queries semantic memory. |
| `remember(content, importance=0.7, tags=None)` / `recall(query=None, top_k=5)` | Convenience wrappers around the underlying working memory. |
| `get_stats()` | Agent-level metrics (runs, success rate, memory usage, kernel stats). |
| `reset_working_memory()` | Clears L1 without touching semantic memory. |

### `SimpleAgent`

Concrete implementation that routes every request through the `think` primitive.
Ideal for prototyping before creating a custom `TemplateAgent` subclass.

```python
from brainary.sdk.template_agent import SimpleAgent

agent = SimpleAgent(name="assistant", domain="support")
result = agent.run("Diagnose this incident")
```

---

## Memory & Diagnostics

- `brainary.sdk.memory.MemoryManager`: utility for orchestrating working memory
    snapshots, consolidation, and capacity adjustments from SDK code.
- `brainary.sdk.get_stats()`: identical to `Brainary.get_stats()` but available
    even when you only use function-based primitives.
- `brainary.sdk.clear_memory()`: resets the singleton client’s working memory to
    the configured capacity.

Use these helpers in integration tests or long-running services to ensure the
SDK stays within budget and memory constraints.

---

For deeper architectural details (execution loop, primitive hierarchy, learning
system) read `doc/KERNEL_EXECUTION_LOOP.md` and `doc/METACOGNITIVE_ARCHITECTURE.md`.
#### perceive()

Gather and parse information from the environment.

```python
perceive(
    content: Optional[str] = None,
    file_path: Optional[str] = None,
    source: str = "unknown",
    **kwargs
) -> PrimitiveResult
```

**Parameters:**
- `content`: Text content to perceive
- `file_path`: Path to file to read and perceive
- `source`: Source identifier
- `**kwargs`: Additional parameters

**Returns:** PrimitiveResult with parsed content

**Example:**
```python
result = client.perceive(
    content="The sky is blue",
    source="observation"
)
print(result.content)
```

---

#### think()

Perform reasoning and problem-solving.

```python
think(
    prompt: str,
    depth: str = "medium",
    require_reasoning: bool = False,
    **kwargs
) -> str
```

**Parameters:**
- `prompt`: Question or problem to think about
- `depth`: Thinking depth ("shallow", "medium", "deep")
- `require_reasoning`: Include reasoning trace
- `**kwargs`: Additional parameters

**Returns:** Thinking result as string

**Example:**
```python
result = client.think(
    "What is the square root of 144?",
    depth="shallow"
)
print(result)  # "12"
```

---

#### act()

Execute actions or generate outputs.

```python
act(
    action_type: str,
    prompt: Optional[str] = None,
    function: Optional[Callable] = None,
    args: Optional[Dict] = None,
    **kwargs
) -> Any
```

**Parameters:**
- `action_type`: Type of action ("generate", "execute", "transform")
- `prompt`: Prompt for generation
- `function`: Function to execute
- `args`: Arguments for function
- `**kwargs`: Additional parameters

**Returns:** Action result (type depends on action_type)

**Example:**
```python
# Generate text
result = client.act(
    action_type="generate",
    prompt="Write a haiku about AI"
)

# Execute function
result = client.act(
    action_type="execute",
    function=my_func,
    args={"x": 10}
)
```

---

#### reflect()

Evaluate and critique previous results.

```python
reflect(
    target: Any,
    criteria: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**
- `target`: Object to reflect on
- `criteria`: Evaluation criteria
- `**kwargs`: Additional parameters

**Returns:** Reflection results dictionary

**Example:**
```python
result = client.think("Solution to problem")
reflection = client.reflect(
    target=result,
    criteria=["correctness", "clarity"]
)
print(reflection["scores"])
```

---

#### remember()

Store information in memory.

```python
remember(
    content: Any,
    importance: float = 0.5,
    tags: Optional[List[str]] = None,
    associations: Optional[Dict[str, float]] = None,
    **kwargs
) -> str
```

**Parameters:**
- `content`: Content to remember
- `importance`: Importance score (0.0 to 1.0)
- `tags`: Tags for categorization
- `associations`: Associated concepts with strengths
- `**kwargs`: Additional parameters

**Returns:** Memory item ID

**Example:**
```python
memory_id = client.remember(
    content="Python is a programming language",
    importance=0.8,
    tags=["programming", "languages"],
    associations={"coding": 0.9, "development": 0.8}
)
```

---

#### recall()

Retrieve information from memory.

```python
recall(
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    top_k: int = 5,
    similarity_threshold: float = 0.7,
    attention_weight: float = 1.0,
    **kwargs
) -> List[MemoryItem]
```

**Parameters:**
- `query`: Search query string
- `tags`: Filter by tags
- `top_k`: Maximum results to return
- `similarity_threshold`: Minimum similarity score
- `attention_weight`: Attention mechanism weight
- `**kwargs`: Additional parameters

**Returns:** List of MemoryItem objects

**Example:**
```python
memories = client.recall(
    query="programming languages",
    tags=["programming"],
    top_k=3
)
for mem in memories:
    print(f"{mem.content} (score: {mem.score})")
```

---

#### associate()

Find connections between concepts.

```python
associate(
    concept: str,
    depth: int = 1,
    min_strength: float = 0.5,
    max_associations: int = 10,
    **kwargs
) -> List[Association]
```

**Parameters:**
- `concept`: Starting concept
- `depth`: How many hops (1-3)
- `min_strength`: Minimum association strength
- `max_associations`: Maximum associations to return
- `**kwargs`: Additional parameters

**Returns:** List of Association objects

**Example:**
```python
associations = client.associate(
    concept="machine learning",
    depth=2,
    min_strength=0.6
)
for assoc in associations:
    print(f"{assoc.source} -> {assoc.target}: {assoc.strength}")
```

---

### Composite Methods

#### analyze()

Analyze content with various analysis types.

```python
analyze(
    content: str,
    analysis_type: str = "general",
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**
- `content`: Content to analyze
- `analysis_type`: Type ("sentiment", "entities", "topics", "summary", "general")
- `**kwargs`: Additional parameters

**Returns:** Analysis results dictionary

**Example:**
```python
result = client.analyze(
    content="This product is amazing!",
    analysis_type="sentiment"
)
print(result["sentiment"])  # "positive"
print(result["confidence"])  # 0.95
```

---

#### plan()

Create a plan to achieve a goal.

```python
plan(
    goal: str,
    constraints: Optional[List[str]] = None,
    resources: Optional[List[str]] = None,
    **kwargs
) -> Plan
```

**Parameters:**
- `goal`: Goal to achieve
- `constraints`: List of constraints
- `resources`: Available resources
- `**kwargs`: Additional parameters

**Returns:** Plan object with steps

**Example:**
```python
plan = client.plan(
    goal="Build a web application",
    constraints=["2 weeks", "Python only"],
    resources=["Flask", "SQLite"]
)
for step in plan.steps:
    print(f"{step.order}. {step.description}")
```

---

#### solve()

Solve a problem using various approaches.

```python
solve(
    problem: str,
    approach: str = "systematic",
    **kwargs
) -> Solution
```

**Parameters:**
- `problem`: Problem to solve
- `approach`: Approach ("trial_error", "systematic", "creative", "analytical")
- `**kwargs`: Additional parameters

**Returns:** Solution object

**Example:**
```python
solution = client.solve(
    problem="Optimize database queries",
    approach="systematic"
)
print(solution.description)
print(f"Confidence: {solution.confidence}")
```

---

#### learn()

Learn and store new information.

```python
learn(
    content: str,
    category: str = "general",
    importance: float = 0.5,
    **kwargs
) -> str
```

**Parameters:**
- `content`: Content to learn
- `category`: Knowledge category
- `importance`: Importance score
- `**kwargs`: Additional parameters

**Returns:** Knowledge ID

**Example:**
```python
knowledge_id = client.learn(
    content="REST APIs use HTTP methods",
    category="web_development",
    importance=0.8
)
```

---

### Metacognitive Methods

#### monitor()

Monitor execution and collect metrics.

```python
monitor() -> ContextManager[Monitor]
```

**Returns:** Monitor context manager

**Example:**
```python
with client.monitor() as monitor:
    result = client.think("Complex problem")
    
print(f"Time: {monitor.execution_time}ms")
print(f"Tokens: {monitor.tokens_used}")
print(f"Cost: ${monitor.cost:.4f}")
print(f"Quality: {monitor.quality_score}")
```

---

#### adapt()

Adapt strategy based on feedback.

```python
adapt(
    feedback: Dict[str, Any],
    adjustment: str,
    **kwargs
) -> None
```

**Parameters:**
- `feedback`: Feedback dictionary
- `adjustment`: Type of adjustment
- `**kwargs`: Additional parameters

**Example:**
```python
client.adapt(
    feedback={"success": False, "reason": "timeout"},
    adjustment="increase_timeout"
)
```

---

### Utility Methods

#### get_status()

Get current system status.

```python
get_status() -> SystemStatus
```

**Returns:** SystemStatus object

**Example:**
```python
status = client.get_status()
print(f"Memory: {status.memory_usage}")
print(f"Tokens: {status.tokens_used}")
print(f"Cache hits: {status.cache_hit_rate}")
```

---

#### clear_memory()

Clear memory tiers.

```python
clear_memory(
    tier: Optional[str] = None
) -> None
```

**Parameters:**
- `tier`: Tier to clear ("L1", "L2", "L3", None for all)

**Example:**
```python
client.clear_memory("L1")  # Clear working memory
client.clear_memory()      # Clear all memory
```

---

#### reset()

Reset client to initial state.

```python
reset() -> None
```

**Example:**
```python
client.reset()
```

---

## Core API

### ExecutionContext

Context for executing primitives.

```python
from brainary import ExecutionContext, create_execution_context
```

#### create_execution_context()

```python
create_execution_context(
    program_name: str,
    execution_mode: ExecutionMode = ExecutionMode.BALANCED,
    quality_threshold: float = 0.7,
    criticality: float = 0.5,
    time_pressure: float = 0.5,
    token_budget: int = 10000,
    domain: str = "general",
    **kwargs
) -> ExecutionContext
```

**Parameters:**
- `program_name`: Name of the program
- `execution_mode`: Execution mode (FAST, BALANCED, DEEP, ADAPTIVE)
- `quality_threshold`: Minimum quality threshold
- `criticality`: Task criticality (0.0 to 1.0)
- `time_pressure`: Time pressure (0.0 to 1.0)
- `token_budget`: Maximum tokens to use
- `domain`: Domain identifier
- `**kwargs`: Additional context parameters

**Returns:** ExecutionContext

**Example:**
```python
context = create_execution_context(
    program_name="my_app",
    execution_mode=ExecutionMode.ADAPTIVE,
    quality_threshold=0.8,
    token_budget=50000
)
```

---

### CognitiveKernel

Core execution engine.

```python
from brainary import CognitiveKernel
```

#### Constructor

```python
CognitiveKernel() -> CognitiveKernel
```

#### execute()

```python
execute(
    primitive_name: str,
    context: ExecutionContext,
    **kwargs
) -> PrimitiveResult
```

**Parameters:**
- `primitive_name`: Name of primitive to execute
- `context`: Execution context
- `**kwargs`: Primitive-specific parameters

**Returns:** PrimitiveResult

**Example:**
```python
kernel = CognitiveKernel()
result = kernel.execute(
    primitive_name="think",
    context=context,
    prompt="What is 2+2?"
)
```

---

### ExecutionMode

Execution mode enumeration.

```python
from brainary import ExecutionMode
```

**Values:**
- `ExecutionMode.FAST`: Fast execution (lower quality, lower cost)
- `ExecutionMode.BALANCED`: Balanced execution
- `ExecutionMode.DEEP`: Deep execution (higher quality, higher cost)
- `ExecutionMode.ADAPTIVE`: Adaptive execution (context-dependent)

---

## Primitives API

### Primitive Base Class

```python
from brainary.primitive import Primitive, CorePrimitive
```

#### CorePrimitive

Base class for implementing custom primitives.

```python
class MyPrimitive(CorePrimitive):
    def __init__(self):
        super().__init__()
        self._name = "my_primitive"
        self._hint = "Description of what this primitive does"
    
    def validate_inputs(self, **kwargs) -> None:
        """Validate input parameters."""
        if "required_param" not in kwargs:
            raise ValueError("required_param is required")
    
    def estimate_cost(self, **kwargs) -> ResourceEstimate:
        """Estimate resource cost."""
        return ResourceEstimate(
            tokens=100,
            latency_ms=50,
            memory_slots=1
        )
    
    def execute(
        self,
        context: ExecutionContext,
        working_memory: WorkingMemory,
        **kwargs
    ) -> PrimitiveResult:
        """Execute the primitive."""
        # Implementation
        return PrimitiveResult(
            success=True,
            content=result,
            confidence=ConfidenceMetrics(overall=0.9),
            cost=CostMetrics(tokens=100, latency_ms=50),
            execution_mode=context.execution_mode,
            primitive_name=self._name
        )
    
    def rollback(self, context: ExecutionContext) -> None:
        """Rollback changes if needed."""
        pass
```

---

### PrimitiveResult

Result from primitive execution.

```python
from brainary import PrimitiveResult
```

**Attributes:**
- `success` (bool): Whether execution succeeded
- `content` (Any): Result content
- `error` (Optional[str]): Error message if failed
- `confidence` (ConfidenceMetrics): Confidence metrics
- `cost` (CostMetrics): Resource costs
- `execution_mode` (ExecutionMode): Mode used
- `primitive_name` (str): Primitive name
- `metadata` (Dict): Additional metadata

**Example:**
```python
if result.success:
    print(f"Result: {result.content}")
    print(f"Confidence: {result.confidence.overall}")
    print(f"Cost: {result.cost.tokens} tokens")
else:
    print(f"Error: {result.error}")
```

---

### ResourceEstimate

Resource cost estimation.

```python
from brainary import ResourceEstimate
```

**Constructor:**
```python
ResourceEstimate(
    tokens: int,
    latency_ms: float,
    memory_slots: int = 0,
    cost_usd: float = 0.0
)
```

**Attributes:**
- `tokens`: Estimated token usage
- `latency_ms`: Estimated latency in milliseconds
- `memory_slots`: Memory slots needed
- `cost_usd`: Estimated cost in USD

---

### ConfidenceMetrics

Confidence scores.

```python
from brainary import ConfidenceMetrics
```

**Constructor:**
```python
ConfidenceMetrics(
    overall: float,
    components: Optional[Dict[str, float]] = None
)
```

**Attributes:**
- `overall`: Overall confidence (0.0 to 1.0)
- `components`: Component-wise confidence scores

---

### CostMetrics

Resource cost metrics.

```python
from brainary import CostMetrics
```

**Constructor:**
```python
CostMetrics(
    tokens: int = 0,
    latency_ms: float = 0.0,
    memory_slots: int = 0,
    provider_cost_usd: float = 0.0
)
```

**Attributes:**
- `tokens`: Tokens used
- `latency_ms`: Execution time in milliseconds
- `memory_slots`: Memory slots used
- `provider_cost_usd`: Provider cost in USD

---

## Memory API

### WorkingMemory

L1/L2 working memory implementation.

```python
from brainary import WorkingMemory
```

#### Constructor

```python
WorkingMemory(
    capacity: int = 7,
    l2_capacity: int = 100,
    auto_consolidate: bool = True,
    consolidation_threshold: float = 0.8
) -> WorkingMemory
```

**Parameters:**
- `capacity`: L1 capacity (default: 7)
- `l2_capacity`: L2 capacity (default: 100)
- `auto_consolidate`: Auto-consolidate when full
- `consolidation_threshold`: Threshold for consolidation

---

#### store()

Store item in memory.

```python
store(
    content: Any,
    importance: float = 0.5,
    tags: Optional[List[str]] = None,
    tier: MemoryTier = MemoryTier.L1_WORKING,
    **kwargs
) -> str
```

**Parameters:**
- `content`: Content to store
- `importance`: Importance score (0.0 to 1.0)
- `tags`: Tags for categorization
- `tier`: Memory tier (L1, L2, L3)
- `**kwargs`: Additional metadata

**Returns:** Item ID

**Example:**
```python
item_id = memory.store(
    content={"fact": "value"},
    importance=0.9,
    tags=["important", "fact"]
)
```

---

#### retrieve()

Retrieve items from memory.

```python
retrieve(
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    top_k: int = 5,
    min_importance: float = 0.0,
    similarity_threshold: float = 0.7,
    tier: Optional[MemoryTier] = None
) -> List[MemoryItem]
```

**Parameters:**
- `query`: Search query
- `tags`: Filter by tags
- `top_k`: Maximum results
- `min_importance`: Minimum importance
- `similarity_threshold`: Minimum similarity
- `tier`: Specific tier to search

**Returns:** List of MemoryItem objects

**Example:**
```python
items = memory.retrieve(
    tags=["important"],
    top_k=3,
    min_importance=0.7
)
```

---

#### consolidate()

Move items between memory tiers.

```python
consolidate(
    from_tier: str = "L1",
    to_tier: str = "L2",
    criteria: str = "importance",
    num_items: Optional[int] = None
) -> int
```

**Parameters:**
- `from_tier`: Source tier ("L1", "L2")
- `to_tier`: Target tier ("L2", "L3")
- `criteria`: Selection criteria ("importance", "recency", "frequency")
- `num_items`: Number of items to consolidate

**Returns:** Number of items consolidated

**Example:**
```python
consolidated = memory.consolidate(
    from_tier="L1",
    to_tier="L2",
    criteria="importance"
)
```

---

#### is_full()

Check if memory is full.

```python
is_full(tier: str = "L1") -> bool
```

**Parameters:**
- `tier`: Tier to check ("L1", "L2")

**Returns:** True if full

---

#### clear()

Clear memory tier.

```python
clear(tier: Optional[str] = None) -> None
```

**Parameters:**
- `tier`: Tier to clear (None for all)

---

### MemoryTier

Memory tier enumeration.

```python
from brainary import MemoryTier
```

**Values:**
- `MemoryTier.L1_WORKING`: Working memory (active)
- `MemoryTier.L2_EPISODIC`: Episodic memory (recent)
- `MemoryTier.L3_SEMANTIC`: Semantic memory (long-term)

---

### MemoryItem

Memory item representation.

```python
from brainary import MemoryItem
```

**Attributes:**
- `id` (str): Item ID
- `content` (Any): Item content
- `importance` (float): Importance score
- `tags` (List[str]): Tags
- `tier` (MemoryTier): Memory tier
- `timestamp` (datetime): Creation time
- `access_count` (int): Access count
- `last_access` (datetime): Last access time
- `metadata` (Dict): Additional metadata

---

### PrefetchRequest

Memory prefetch request.

```python
from brainary import PrefetchRequest
```

**Constructor:**
```python
PrefetchRequest(
    tags: Optional[List[str]] = None,
    max_items: int = 5,
    min_importance: float = 0.5,
    priority: str = "normal"
)
```

**Attributes:**
- `tags`: Tags to prefetch
- `max_items`: Maximum items to prefetch
- `min_importance`: Minimum importance
- `priority`: Priority ("low", "normal", "high")

---

## LLM API

### LLMManager

Manage LLM interactions.

```python
from brainary.llm import LLMManager, get_llm_manager
```

#### get_llm_manager()

Get singleton LLM manager.

```python
get_llm_manager() -> LLMManager
```

**Returns:** LLMManager instance

---

#### invoke()

Invoke LLM.

```python
invoke(
    request: LLMRequest,
    context: Optional[ExecutionContext] = None
) -> LLMResponse
```

**Parameters:**
- `request`: LLM request
- `context`: Execution context

**Returns:** LLMResponse

**Example:**
```python
manager = get_llm_manager()
request = LLMRequest.from_string(
    prompt="What is 2+2?",
    model="gpt-4o-mini"
)
response = manager.invoke(request)
print(response.content)
```

---

### LLMRequest

LLM request specification.

```python
from brainary.llm import LLMRequest
```

#### from_string()

Create request from string.

```python
LLMRequest.from_string(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    **kwargs
) -> LLMRequest
```

**Parameters:**
- `prompt`: Prompt string
- `model`: Model name
- `temperature`: Temperature (0.0 to 2.0)
- `max_tokens`: Maximum tokens
- `**kwargs`: Additional parameters

**Returns:** LLMRequest

---

### LLMResponse

LLM response.

```python
from brainary.llm import LLMResponse
```

**Attributes:**
- `content` (str): Response content
- `model` (str): Model used
- `tokens_used` (int): Tokens used
- `cost_usd` (float): Cost in USD
- `latency_ms` (float): Response time
- `metadata` (Dict): Additional metadata

---

### LLMConfig

LLM configuration.

```python
from brainary.llm import LLMConfig
```

**Constructor:**
```python
LLMConfig(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    **kwargs
)
```

**Attributes:**
- `provider`: Provider name
- `model`: Model name
- `api_key`: API key
- `temperature`: Temperature
- `max_tokens`: Max tokens
- Additional provider-specific settings

---

## Resource Management API

### ResourceManager

Manage computational resources.

```python
from brainary import ResourceManager
```

#### Constructor

```python
ResourceManager() -> ResourceManager
```

---

#### set_quota()

Set resource quota.

```python
set_quota(quota: ResourceQuota) -> None
```

**Parameters:**
- `quota`: ResourceQuota object

**Example:**
```python
manager = ResourceManager()
manager.set_quota(ResourceQuota(
    max_tokens=100000,
    max_memory_mb=512,
    max_time_seconds=300,
    max_cost_usd=10.0
))
```

---

#### check_availability()

Check if resources are available.

```python
check_availability(
    tokens: int = 0,
    memory_mb: int = 0,
    cost_usd: float = 0.0
) -> bool
```

**Parameters:**
- `tokens`: Tokens needed
- `memory_mb`: Memory needed
- `cost_usd`: Cost needed

**Returns:** True if available

---

#### allocate()

Allocate resources.

```python
allocate(
    tokens: int = 0,
    memory_mb: int = 0,
    cost_usd: float = 0.0
) -> ResourceAllocation
```

**Parameters:**
- `tokens`: Tokens to allocate
- `memory_mb`: Memory to allocate
- `cost_usd`: Cost to allocate

**Returns:** ResourceAllocation

**Raises:** ResourceExhaustedError if quota exceeded

---

#### get_usage_stats()

Get resource usage statistics.

```python
get_usage_stats() -> Dict[str, Any]
```

**Returns:** Usage statistics dictionary

**Example:**
```python
stats = manager.get_usage_stats()
print(f"Tokens: {stats['tokens_used']}/{stats['tokens_total']}")
print(f"Cost: ${stats['cost_used']:.4f}")
```

---

#### reset_quota()

Reset resource quota counters.

```python
reset_quota() -> None
```

---

### ResourceQuota

Resource quota specification.

```python
from brainary import ResourceQuota
```

**Constructor:**
```python
ResourceQuota(
    max_tokens: int = 100000,
    max_memory_mb: int = 1024,
    max_time_seconds: int = 600,
    max_cost_usd: float = 10.0
)
```

**Attributes:**
- `max_tokens`: Maximum tokens
- `max_memory_mb`: Maximum memory in MB
- `max_time_seconds`: Maximum time in seconds
- `max_cost_usd`: Maximum cost in USD

---

## Utility API

### Exceptions

```python
from brainary import (
    BrainaryError,
    ResourceExhaustedError,
    MemoryError,
    LLMError,
    PrimitiveError
)
```

**Exception Hierarchy:**
```
BrainaryError (base)
├── ResourceExhaustedError
├── MemoryError
├── LLMError
└── PrimitiveError
```

**Example:**
```python
try:
    result = client.think("problem")
except ResourceExhaustedError as e:
    print(f"Out of resources: {e}")
except BrainaryError as e:
    print(f"Brainary error: {e}")
```

---

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Brainary logger
logger = logging.getLogger('brainary')
logger.setLevel(logging.DEBUG)
```

---

### Type Hints

```python
from typing import Optional, List, Dict, Any, Callable
from brainary import (
    ExecutionContext,
    WorkingMemory,
    PrimitiveResult,
    MemoryItem,
    LLMRequest,
    LLMResponse
)
```

---

## Version Information

```python
import brainary

print(f"Brainary version: {brainary.__version__}")
print(f"API version: {brainary.__api_version__}")
```

---

## Complete Example

```python
from brainary import (
    BrainaryClient,
    create_execution_context,
    ExecutionMode,
    WorkingMemory,
    ResourceManager,
    ResourceQuota
)

# Setup
context = create_execution_context(
    program_name="my_app",
    execution_mode=ExecutionMode.ADAPTIVE,
    quality_threshold=0.8,
    token_budget=50000
)

memory = WorkingMemory(capacity=10)

resource_manager = ResourceManager()
resource_manager.set_quota(ResourceQuota(
    max_tokens=100000,
    max_cost_usd=5.0
))

client = BrainaryClient(
    llm_provider="openai",
    model="gpt-4o-mini",
    memory_capacity=10
)

# Use primitives
question = "What is machine learning?"

# Think
with client.monitor() as monitor:
    answer = client.think(question, depth="medium")

# Remember
client.remember(
    content=f"Q: {question}\nA: {answer}",
    importance=0.8,
    tags=["ml", "education"]
)

# Recall
memories = client.recall(tags=["ml"], top_k=3)

# Monitor
print(f"Execution time: {monitor.execution_time}ms")
print(f"Tokens used: {monitor.tokens_used}")
print(f"Cost: ${monitor.cost:.4f}")

# Check resources
stats = resource_manager.get_usage_stats()
print(f"Resource usage: {stats}")
```

---

*Last updated: November 21, 2025*
*API Version: 0.1.0*
