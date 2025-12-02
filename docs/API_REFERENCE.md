# Brainary API Reference (0.9.x)

This document covers the public APIs surfaced through `brainary.sdk`. It is
organized by entry point so you can quickly find constructor arguments, method
signatures, and return types.

## Contents

1. [SDK Overview](#sdk-overview)
2. [Client API (`Brainary`)](#client-api-brainary)
3. [Context Helpers](#context-helpers)
4. [Function-Based Primitives](#function-based-primitives)
5. [Template Agents](#template-agents)
6. [Agent System](#agent-system)
7. [Memory Management](#memory-management)
8. [Memory & Diagnostics](#memory--diagnostics)

---

## SDK Overview

The Brainary SDK (`brainary.sdk`) provides a comprehensive, user-friendly interface to the Brainary cognitive computing platform. It is designed to abstract away internal complexity while exposing powerful cognitive capabilities through clean, intuitive APIs.

### Architecture

The SDK is organized into several key modules:

```python
from brainary.sdk import (
    # Main client interface
    Brainary,
    
    # Function-based primitives API
    perceive, think, remember, recall, associate,
    analyze, solve, decide, plan,
    introspect, self_assess, select_strategy, self_correct,
    
    # Memory management
    MemoryManager,
    
    # Context management
    ContextBuilder, ContextManager, create_context,
    
    # Agent templates
    Agent, AgentTeam, AgentRole, AgentConfig,
    TemplateAgent, SimpleAgent,
    
    # Configuration utilities
    configure, get_stats, clear_memory,
)
```

### Module Structure

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `brainary.sdk.client` | Main SDK interface | `Brainary` - Primary client class |
| `brainary.sdk.primitives` | Function-based API | Core, composite, and metacognitive primitives |
| `brainary.sdk.memory` | Memory management | `MemoryManager` - High-level memory interface |
| `brainary.sdk.context` | Context management | `ContextBuilder`, `ContextManager` - Context creation and scoping |
| `brainary.sdk.agents` | Agent system | `Agent`, `AgentTeam`, `AgentRole`, `AgentConfig` |
| `brainary.sdk.template_agent` | Agent templates | `TemplateAgent`, `SimpleAgent` - Base templates for custom agents |

### Design Principles

1. **Dual API Approach**: The SDK supports both object-oriented (via `Brainary` client) and functional programming styles (via standalone functions)
2. **Progressive Disclosure**: Simple tasks remain simple while complex scenarios are fully supported
3. **Kernel-Scoped Memory**: Memories persist across executions within the same kernel instance
4. **Automatic Resource Management**: Token budgets, memory capacity, and quality thresholds are managed automatically
5. **Flexible Configuration**: Context can be configured globally, per-client, or per-operation

### Quick Start Examples

**Object-Oriented Style:**
```python
from brainary.sdk import Brainary

# Initialize client
brain = Brainary(
    enable_learning=True,
    memory_capacity=10,
    quality_threshold=0.9
)

# Execute cognitive operations
result = brain.think("Explain quantum computing")
brain.remember(result.content, importance=0.8, tags=["physics", "computing"])
memories = brain.recall(query="quantum", limit=5)
```

**Functional Style:**
```python
from brainary.sdk import configure, think, remember, recall

# Configure once (optional)
configure(enable_learning=True, quality_threshold=0.9)

# Use functions directly
result = think("Explain quantum computing")
remember(result.content, importance=0.8, tags=["physics"])
memories = recall(query="quantum", limit=5)
```

**Agent-Based Style:**
```python
from brainary.sdk import Agent, AgentRole

# Create specialized agent
agent = Agent.create("analyst", domain="security")

# Process with agent
result = agent.analyze(code_snippet)
```

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

## Building Custom Agents: Complete Guide

This guide shows you how to build sophisticated custom agents using the `TemplateAgent` system, incorporating working memory, long-term (semantic) memory, metacognition, and custom processing logic.

### Agent Architecture Overview

A `TemplateAgent` provides four key capabilities:

1. **Working Memory (L1)**: Short-term context across executions (7±2 items)
2. **Semantic Memory (L2)**: Long-term knowledge base (unlimited)
3. **Metacognition**: Self-monitoring and adaptive behavior
4. **Custom Process Logic**: Your implementation of the `process()` method

```
┌─────────────────────────────────────────┐
│         Your Custom Agent               │
├─────────────────────────────────────────┤
│  process() - Your Custom Logic          │
│    ↓                                    │
│  Cognitive Kernel                       │
│    ├── Working Memory (L1)              │
│    ├── Semantic Memory (L2)             │
│    ├── Metacognitive Monitor            │
│    └── Primitive Execution              │
└─────────────────────────────────────────┘
```

### Step 1: Define Your Agent Class

Subclass `TemplateAgent` and implement the abstract `process()` method:

```python
from brainary.sdk.template_agent import TemplateAgent, AgentConfig
from brainary.core import ExecutionContext
from brainary.primitive.base import PrimitiveResult
from typing import Any

class MyCustomAgent(TemplateAgent):
    """
    Custom agent with specialized behavior.
    
    The process() method defines your agent's logic using primitives
    and the cognitive kernel.
    """
    
    def process(
        self,
        input_data: Any,
        context: ExecutionContext,
        **kwargs
    ) -> PrimitiveResult:
        """
        Implement your agent's custom processing logic.
        
        Args:
            input_data: The input to process
            context: Execution context (automatically provided)
            **kwargs: Additional parameters
            
        Returns:
            PrimitiveResult from your final operation
        """
        # Your custom logic goes here
        # Use self.kernel.execute() to call primitives
        # Access memories via self.working_memory and self.semantic_memory
        
        result = self.kernel.execute(
            "think",
            context=context,
            working_memory=self.working_memory,
            query=str(input_data)
        )
        
        return result
```

### Step 2: Configure Agent Capabilities

Use `AgentConfig` to enable/disable features and set parameters:

```python
from brainary.sdk.template_agent import AgentConfig
from brainary.core.metacognitive_monitor import MonitoringLevel

# Create configuration
config = AgentConfig(
    name="research_assistant",
    description="Research agent with deep analysis capabilities",
    domain="research",
    
    # Memory settings
    working_memory_capacity=15,      # Increase for more context
    enable_semantic_memory=True,      # Enable long-term knowledge
    
    # Metacognition settings
    enable_metacognition=True,        # Enable self-monitoring
    monitoring_level=MonitoringLevel.DETAILED,  # BASIC, STANDARD, or DETAILED
    
    # Execution settings
    enable_learning=True,             # Enable adaptive learning
    quality_threshold=0.9,            # High quality requirement
    default_execution_mode="deep",    # "fast", "deep", or "adaptive"
    max_token_budget=15000,           # Generous token budget
    
    # Custom metadata
    metadata={
        "specialization": "academic_research",
        "citation_required": True
    }
)

# Create agent with configuration
agent = MyCustomAgent(config=config)
```

### Step 3: Implement Working Memory Integration

Working memory provides short-term context that persists across executions:

```python
class ContextAwareAgent(TemplateAgent):
    def process(self, input_data, context, **kwargs):
        # Store current input in working memory
        self.remember(
            content=f"Current task: {input_data}",
            importance=0.8,
            tags=["current_task", "context"]
        )
        
        # Recall relevant past context
        past_context = self.recall(
            query=str(input_data),
            top_k=3
        )
        
        # Build context-aware query
        context_info = "\n".join([
            f"- {item.content} (importance: {item.importance})"
            for item in past_context
        ])
        
        enhanced_query = f"""
        Current input: {input_data}
        
        Relevant past context:
        {context_info}
        
        Please analyze with full context awareness.
        """
        
        # Execute with context
        result = self.kernel.execute(
            "analyze",
            context=context,
            working_memory=self.working_memory,
            data=enhanced_query
        )
        
        # Store result for future reference
        self.remember(
            content=f"Analysis result: {result.content[:200]}...",
            importance=result.quality,
            tags=["result", "analysis"]
        )
        
        return result
```

### Step 4: Implement Semantic Memory Integration

Semantic memory provides long-term knowledge that enriches your agent's capabilities:

```python
from brainary.memory import (
    ConceptualKnowledge,
    FactualKnowledge,
    ProceduralKnowledge,
    MetacognitiveKnowledge
)

class KnowledgeEnhancedAgent(TemplateAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Pre-populate with domain knowledge
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Load domain-specific knowledge."""
        
        # Add conceptual knowledge
        self.add_knowledge(ConceptualKnowledge(
            concept="machine_learning",
            description="Statistical methods for pattern recognition from data",
            related_concepts=["ai", "statistics", "neural_networks"],
            domain=self.config.domain
        ))
        
        # Add factual knowledge
        self.add_knowledge(FactualKnowledge(
            subject="Python",
            predicate="latest_version",
            object="3.12",
            confidence=0.95,
            domain=self.config.domain
        ))
        
        # Add procedural knowledge
        self.add_knowledge(ProceduralKnowledge(
            task="code_review",
            steps=[
                "Check for security vulnerabilities",
                "Verify code style and conventions",
                "Test edge cases",
                "Review documentation"
            ],
            conditions=["code is complete", "tests exist"],
            expected_outcome="comprehensive review report",
            domain=self.config.domain
        ))
        
        # Add metacognitive knowledge
        self.add_knowledge(MetacognitiveKnowledge(
            strategy_name="adaptive_analysis",
            applicable_contexts=["complex_problems", "multi_step_tasks"],
            effectiveness=0.85,
            conditions=["sufficient time", "detailed requirements"],
            domain=self.config.domain
        ))
    
    def process(self, input_data, context, **kwargs):
        # Search knowledge base for relevant information
        relevant_knowledge = self.search_knowledge(
            query=str(input_data),
            knowledge_types=["conceptual", "factual", "procedural"],
            top_k=5
        )
        
        # Build knowledge-enhanced prompt
        knowledge_context = "\n".join([
            f"- {k.to_dict()}"
            for k in relevant_knowledge
        ])
        
        enhanced_input = f"""
        Input: {input_data}
        
        Relevant knowledge:
        {knowledge_context}
        
        Apply this knowledge to the analysis.
        """
        
        # Execute with knowledge context
        result = self.kernel.execute(
            "analyze",
            context=context,
            working_memory=self.working_memory,
            data=enhanced_input,
            knowledge=relevant_knowledge
        )
        
        return result
```

### Step 5: Implement Metacognition

Metacognition enables your agent to monitor its own performance and adapt:

```python
from brainary.core.metacognitive_monitor import MonitoringLevel

class SelfAwareAgent(TemplateAgent):
    def __init__(self, *args, **kwargs):
        # Enable detailed metacognition
        if 'config' in kwargs:
            kwargs['config'].enable_metacognition = True
            kwargs['config'].monitoring_level = MonitoringLevel.DETAILED
        
        super().__init__(*args, **kwargs)
    
    def process(self, input_data, context, **kwargs):
        # Step 1: Self-assess capability for this task
        assessment = self.kernel.execute(
            "self_assess",
            context=context,
            working_memory=self.working_memory,
            task=str(input_data)
        )
        
        # Step 2: Select strategy based on assessment
        if assessment.confidence < 0.6:
            # Low confidence - use deep reasoning
            strategy = "deep_analysis"
            execution_mode = "deep"
        else:
            # High confidence - use adaptive
            strategy = "adaptive_analysis"
            execution_mode = "adaptive"
        
        # Step 3: Execute with selected strategy
        result = self.kernel.execute(
            "analyze",
            context=context,
            working_memory=self.working_memory,
            data=input_data,
            strategy=strategy,
            execution_mode=execution_mode
        )
        
        # Step 4: Self-correct if quality is low
        if result.quality < self.config.quality_threshold:
            # Introspect to understand the issue
            introspection = self.kernel.execute(
                "introspect",
                context=context,
                working_memory=self.working_memory
            )
            
            # Attempt correction
            result = self.kernel.execute(
                "self_correct",
                context=context,
                working_memory=self.working_memory,
                error=f"Low quality result: {result.quality}",
                hypothesis=introspection.content
            )
        
        # Step 5: Store metacognitive insights
        self.remember(
            content=f"Strategy: {strategy}, Quality: {result.quality}",
            importance=0.7,
            tags=["metacognition", "performance"]
        )
        
        return result
```

### Step 5a: Customizing Metacognitive Monitoring Rules

The metacognitive monitor uses a flexible criteria system that you can customize with domain-specific rules. Each criterion defines what to check, when to check it, and what action to take if the check fails.

#### Understanding Monitoring Criteria

A monitoring criterion consists of:

1. **WHAT to check**: Evaluation logic (e.g., quality threshold, security check)
2. **WHEN to check**: Pre-execution, post-execution, or continuous
3. **WHERE to check**: Which primitives it applies to (or all)
4. **WHAT TO DO**: Action to take if check fails (filter, retry, reject, warn, etc.)

#### Built-in Criteria

The monitor includes three default criteria:

```python
from brainary.core.metacognitive_rules import (
    ContentSecurityCriterion,      # Priority 100: Security filtering
    ConfidenceThresholdCriterion,  # Priority 50: Auto-retry low confidence
    ResourceLimitCriterion,         # Priority 10: Resource warnings
)
```

#### Creating Custom Criteria

Subclass `MonitoringCriterion` to define your own rules:

```python
from brainary.core.metacognitive_rules import (
    MonitoringCriterion,
    CriteriaType,
    CriteriaEvaluation,
    TransitionAction,
    ActionType
)
from brainary.core.context import ExecutionContext
from brainary.primitive.base import PrimitiveResult
from typing import Any, Dict, Optional


class DomainSpecificCriterion(MonitoringCriterion):
    """
    Custom criterion for domain-specific validation.
    
    Example: Ensure medical diagnoses include confidence levels
    and cite sources.
    """
    
    def __init__(self, domain: str, priority: int = 60):
        self.domain = domain
        
        # Define the action to take if criterion fails
        action = TransitionAction(
            action_type=ActionType.AUGMENT,
            reason="Add required domain validation steps",
            metadata={"validation_type": "domain_specific"}
        )
        
        super().__init__(
            criterion_id=f"domain_validation_{domain}",
            criteria_type=CriteriaType.POST_EXECUTION,
            description=f"Validate {domain}-specific requirements",
            applicable_primitives=["analyze", "decide", "think"],
            severity=0.8,  # High severity
            action=action,
            priority=priority
        )
    
    def evaluate(
        self,
        primitive_name: str,
        context: ExecutionContext,
        result: Optional[PrimitiveResult] = None,
        **kwargs
    ) -> CriteriaEvaluation:
        """
        Evaluate if result meets domain requirements.
        
        Args:
            primitive_name: Name of the primitive
            context: Execution context
            result: Result to validate (for post-execution)
            **kwargs: Additional parameters
            
        Returns:
            CriteriaEvaluation with pass/fail and details
        """
        # Only evaluate for matching domain
        if context.metadata.get("domain") != self.domain:
            return CriteriaEvaluation(
                criterion_id=self.criterion_id,
                passed=True,
                severity=0.0,
                details="Not applicable to this domain"
            )
        
        # Check if result exists (post-execution)
        if result is None:
            return CriteriaEvaluation(
                criterion_id=self.criterion_id,
                passed=True,
                severity=0.0,
                details="Pre-execution check - skipped"
            )
        
        issues = []
        
        # Validation 1: Confidence must be explicit
        if result.confidence is None or result.confidence < 0.7:
            issues.append("Insufficient confidence in result")
        
        # Validation 2: Must cite sources for medical domain
        if self.domain == "medical":
            if "source" not in result.metadata and "citation" not in result.metadata:
                issues.append("Missing required citations")
        
        # Validation 3: Check for required metadata
        required_fields = ["reasoning", "evidence"]
        missing_fields = [f for f in required_fields if f not in result.metadata]
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
        
        passed = len(issues) == 0
        
        return CriteriaEvaluation(
            criterion_id=self.criterion_id,
            passed=passed,
            severity=self.severity if not passed else 0.0,
            details="; ".join(issues) if issues else "All domain requirements met",
            metadata={"issues": issues, "domain": self.domain}
        )
```

#### Example: Quality Gate Criterion

Enforce minimum quality standards with automatic escalation:

```python
class QualityGateCriterion(MonitoringCriterion):
    """
    Enforce quality thresholds with automatic escalation.
    
    If initial execution is below threshold, automatically
    retry with deeper reasoning mode.
    """
    
    def __init__(self, min_quality: float = 0.8, max_retries: int = 2):
        self.min_quality = min_quality
        self.max_retries = max_retries
        self.retry_count = {}
        
        action = TransitionAction(
            action_type=ActionType.RETRY,
            modified_params={"execution_mode": "deep"},
            reason="Quality below threshold, escalating to deep mode"
        )
        
        super().__init__(
            criterion_id="quality_gate",
            criteria_type=CriteriaType.POST_EXECUTION,
            description=f"Enforce minimum quality {min_quality}",
            applicable_primitives=None,  # Apply to all
            severity=0.7,
            action=action,
            priority=70
        )
    
    def evaluate(
        self,
        primitive_name: str,
        context: ExecutionContext,
        result: Optional[PrimitiveResult] = None,
        **kwargs
    ) -> CriteriaEvaluation:
        if result is None or result.quality is None:
            return CriteriaEvaluation(
                criterion_id=self.criterion_id,
                passed=True,
                severity=0.0,
                details="No result to evaluate"
            )
        
        # Track retries
        operation_id = context.metadata.get("operation_id", "unknown")
        retries = self.retry_count.get(operation_id, 0)
        
        # Check quality
        passed = result.quality >= self.min_quality
        
        if not passed and retries < self.max_retries:
            self.retry_count[operation_id] = retries + 1
            return CriteriaEvaluation(
                criterion_id=self.criterion_id,
                passed=False,
                severity=self.severity,
                details=f"Quality {result.quality:.2f} below {self.min_quality} (retry {retries + 1}/{self.max_retries})",
                metadata={"retry_count": retries + 1}
            )
        
        # Clean up retry count
        if operation_id in self.retry_count:
            del self.retry_count[operation_id]
        
        return CriteriaEvaluation(
            criterion_id=self.criterion_id,
            passed=passed or retries >= self.max_retries,
            severity=0.0 if passed else 0.3,  # Lower severity if max retries reached
            details=f"Quality {result.quality:.2f}" + (" (max retries reached)" if retries >= self.max_retries else "")
        )
```

#### Example: Resource Budget Criterion

Monitor and enforce resource limits:

```python
class ResourceBudgetCriterion(MonitoringCriterion):
    """
    Monitor token and time budgets with warnings and hard limits.
    """
    
    def __init__(
        self,
        token_warn_threshold: int = 8000,
        token_hard_limit: int = 10000,
        time_warn_threshold_ms: int = 25000,
        time_hard_limit_ms: int = 30000
    ):
        self.token_warn = token_warn_threshold
        self.token_limit = token_hard_limit
        self.time_warn = time_warn_threshold_ms
        self.time_limit = time_hard_limit_ms
        
        action = TransitionAction(
            action_type=ActionType.REJECT,
            reason="Resource limits exceeded"
        )
        
        super().__init__(
            criterion_id="resource_budget",
            criteria_type=CriteriaType.POST_EXECUTION,
            description="Monitor resource consumption",
            severity=0.9,
            action=action,
            priority=80
        )
    
    def evaluate(
        self,
        primitive_name: str,
        context: ExecutionContext,
        result: Optional[PrimitiveResult] = None,
        **kwargs
    ) -> CriteriaEvaluation:
        if result is None:
            return CriteriaEvaluation(
                criterion_id=self.criterion_id,
                passed=True,
                severity=0.0,
                details="Pre-execution"
            )
        
        issues = []
        severity = 0.0
        
        # Check token usage
        tokens_used = result.metadata.get("token_count", 0)
        if tokens_used > self.token_limit:
            issues.append(f"Token limit exceeded: {tokens_used}/{self.token_limit}")
            severity = max(severity, 0.9)
        elif tokens_used > self.token_warn:
            issues.append(f"Token warning: {tokens_used}/{self.token_warn}")
            severity = max(severity, 0.3)
        
        # Check time
        time_ms = result.metadata.get("execution_time_ms", 0)
        if time_ms > self.time_limit:
            issues.append(f"Time limit exceeded: {time_ms}ms/{self.time_limit}ms")
            severity = max(severity, 0.9)
        elif time_ms > self.time_warn:
            issues.append(f"Time warning: {time_ms}ms/{self.time_warn}ms")
            severity = max(severity, 0.3)
        
        passed = severity < 0.5  # Hard limits cause failure
        
        return CriteriaEvaluation(
            criterion_id=self.criterion_id,
            passed=passed,
            severity=severity,
            details="; ".join(issues) if issues else "Within resource budgets",
            metadata={"tokens": tokens_used, "time_ms": time_ms}
        )
```

#### Registering Custom Criteria

Add your custom criteria to an agent's monitor:

```python
class CustomMonitoredAgent(TemplateAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Access the monitor from the kernel
        monitor = self.kernel.metacognitive_monitor
        
        # Register custom criteria
        monitor.register_criterion(
            DomainSpecificCriterion(domain="medical", priority=60)
        )
        
        monitor.register_criterion(
            QualityGateCriterion(min_quality=0.85, max_retries=2)
        )
        
        monitor.register_criterion(
            ResourceBudgetCriterion(
                token_warn_threshold=8000,
                token_hard_limit=10000
            )
        )
```

#### Custom Monitoring with Actions

Define complex behaviors with action types:

```python
class DataValidationCriterion(MonitoringCriterion):
    """
    Validate data inputs with automatic transformation.
    """
    
    def __init__(self):
        def sanitize_input(data):
            """Transform function to sanitize input."""
            if isinstance(data, str):
                # Remove potential injection attacks
                data = data.replace("<script>", "").replace("</script>", "")
                data = data.replace("DROP TABLE", "")
            return data
        
        action = TransitionAction(
            action_type=ActionType.TRANSFORM,
            filter_function=sanitize_input,
            reason="Sanitize potentially unsafe input"
        )
        
        super().__init__(
            criterion_id="input_validation",
            criteria_type=CriteriaType.PRE_EXECUTION,
            description="Validate and sanitize input data",
            applicable_primitives=["perceive", "remember"],
            severity=0.8,
            action=action,
            priority=90
        )
    
    def evaluate(self, primitive_name, context, **kwargs) -> CriteriaEvaluation:
        input_data = kwargs.get("input_data") or kwargs.get("content")
        
        if input_data is None:
            return CriteriaEvaluation(
                criterion_id=self.criterion_id,
                passed=True,
                severity=0.0,
                details="No input to validate"
            )
        
        # Check for suspicious patterns
        suspicious_patterns = ["<script>", "DROP TABLE", "'; DELETE", "OR 1=1"]
        found_patterns = [p for p in suspicious_patterns if p in str(input_data)]
        
        if found_patterns:
            return CriteriaEvaluation(
                criterion_id=self.criterion_id,
                passed=False,
                severity=0.8,
                details=f"Suspicious patterns detected: {found_patterns}",
                metadata={"patterns": found_patterns}
            )
        
        return CriteriaEvaluation(
            criterion_id=self.criterion_id,
            passed=True,
            severity=0.0,
            details="Input validation passed"
        )
```

#### Monitoring Levels

Configure monitoring granularity:

```python
from brainary.core.metacognitive_monitor import MonitoringLevel

# Minimal monitoring (success/failure only)
config = AgentConfig(
    name="fast_agent",
    enable_metacognition=True,
    monitoring_level=MonitoringLevel.MINIMAL
)

# Standard monitoring (quality metrics, default)
config = AgentConfig(
    name="standard_agent",
    enable_metacognition=True,
    monitoring_level=MonitoringLevel.STANDARD
)

# Detailed monitoring (execution traces)
config = AgentConfig(
    name="detailed_agent",
    enable_metacognition=True,
    monitoring_level=MonitoringLevel.DETAILED
)

# Introspective monitoring (deep cognitive analysis)
config = AgentConfig(
    name="introspective_agent",
    enable_metacognition=True,
    monitoring_level=MonitoringLevel.INTROSPECTIVE
)
```

#### Complete Example: Agent with Custom Monitoring

```python
from brainary.sdk.template_agent import TemplateAgent, AgentConfig
from brainary.core.metacognitive_monitor import MonitoringLevel
from brainary.core.metacognitive_rules import MonitoringCriterion, CriteriaType, CriteriaEvaluation, TransitionAction, ActionType


class ComplianceCriterion(MonitoringCriterion):
    """Ensure outputs comply with regulatory requirements."""
    
    def __init__(self, required_disclaimers: list, priority: int = 85):
        self.required_disclaimers = required_disclaimers
        
        action = TransitionAction(
            action_type=ActionType.AUGMENT,
            modified_params={"add_disclaimers": True},
            reason="Add required compliance disclaimers"
        )
        
        super().__init__(
            criterion_id="compliance_check",
            criteria_type=CriteriaType.POST_EXECUTION,
            description="Verify regulatory compliance",
            applicable_primitives=["analyze", "decide", "think"],
            severity=0.95,
            action=action,
            priority=priority
        )
    
    def evaluate(self, primitive_name, context, result=None, **kwargs):
        if result is None:
            return CriteriaEvaluation(
                criterion_id=self.criterion_id,
                passed=True,
                severity=0.0,
                details="Pre-execution"
            )
        
        content = str(result.content).lower()
        missing_disclaimers = [
            d for d in self.required_disclaimers
            if d.lower() not in content
        ]
        
        if missing_disclaimers:
            return CriteriaEvaluation(
                criterion_id=self.criterion_id,
                passed=False,
                severity=0.95,
                details=f"Missing required disclaimers: {missing_disclaimers}",
                metadata={"missing": missing_disclaimers}
            )
        
        return CriteriaEvaluation(
            criterion_id=self.criterion_id,
            passed=True,
            severity=0.0,
            details="All compliance requirements met"
        )


class ComplianceAgent(TemplateAgent):
    """Agent with strict compliance monitoring."""
    
    def __init__(self, name: str, domain: str, **kwargs):
        config = AgentConfig(
            name=name,
            domain=domain,
            enable_metacognition=True,
            monitoring_level=MonitoringLevel.DETAILED,
            quality_threshold=0.9
        )
        
        super().__init__(name=name, config=config, **kwargs)
        
        # Register compliance criteria
        self.kernel.metacognitive_monitor.register_criterion(
            ComplianceCriterion(
                required_disclaimers=[
                    "for informational purposes only",
                    "not financial advice",
                    "consult a professional"
                ],
                priority=85
            )
        )
        
        # Register quality gate
        self.kernel.metacognitive_monitor.register_criterion(
            QualityGateCriterion(min_quality=0.9, max_retries=2)
        )
    
    def process(self, input_data, context, **kwargs):
        # Monitor will automatically check criteria
        result = self.kernel.execute(
            "analyze",
            context=context,
            working_memory=self.working_memory,
            data=input_data
        )
        
        # Access monitoring assessment
        assessment = self.kernel.metacognitive_monitor.get_assessment()
        
        if assessment and not assessment.detected_issues:
            self.remember(
                content=f"Compliant analysis: {result.content[:100]}",
                importance=0.9,
                tags=["compliant", "verified"]
            )
        
        return result


# Usage
agent = ComplianceAgent(name="compliance_advisor", domain="financial")
result = agent.run("Should I invest in cryptocurrency?")
```

#### Best Practices for Custom Monitoring

1. **Priority Management**: Set priorities to control execution order
   - 90-100: Security and safety checks
   - 70-89: Quality and compliance
   - 50-69: Resource management
   - 10-49: Warnings and logging

2. **Action Types**: Choose appropriate actions
   - `FILTER`: Sanitize content (security)
   - `REJECT`: Block execution (safety)
   - `RETRY`: Improve quality (performance)
   - `AUGMENT`: Add steps (compliance)
   - `WARN`: Log issues (monitoring)
   - `TRANSFORM`: Modify data (validation)

3. **Severity Levels**: Calibrate severity for proper escalation
   - 0.9-1.0: Critical (reject/block)
   - 0.7-0.9: High (retry/augment)
   - 0.4-0.7: Medium (warn/log)
   - 0.0-0.4: Low (informational)

4. **Applicable Primitives**: Target specific operations
   - Use `None` for global criteria
   - Specify list for targeted monitoring
   - Common targets: `["think", "analyze", "decide"]`

5. **Testing**: Validate criteria behavior
   - Test pass and fail conditions
   - Verify actions are triggered correctly
   - Check priority ordering
   - Monitor performance impact

### Step 6: Build Multi-Step Processes

Combine primitives to create sophisticated multi-step workflows:

```python
class ResearchAgent(TemplateAgent):
    """
    Advanced research agent with multi-step processing.
    """
    
    def process(self, input_data, context, **kwargs):
        # Multi-step research process
        
        # Step 1: Decompose the research question
        decomposition = self.kernel.execute(
            "decompose",
            context=context,
            working_memory=self.working_memory,
            problem=str(input_data)
        )
        
        self.remember(
            content=f"Research plan: {decomposition.content}",
            importance=0.9,
            tags=["plan", "research"]
        )
        
        # Step 2: Search knowledge base
        knowledge = self.search_knowledge(
            query=str(input_data),
            top_k=10
        )
        
        # Step 3: Gather information for each sub-question
        findings = []
        for sub_question in decomposition.content.get("sub_questions", []):
            # Think about each sub-question
            finding = self.kernel.execute(
                "think",
                context=context,
                working_memory=self.working_memory,
                query=sub_question,
                reasoning_mode="deep"
            )
            
            findings.append({
                "question": sub_question,
                "answer": finding.content,
                "quality": finding.quality
            })
            
            # Store intermediate finding
            self.remember(
                content=f"Q: {sub_question}\nA: {finding.content}",
                importance=finding.quality,
                tags=["finding", "intermediate"]
            )
        
        # Step 4: Synthesize findings
        synthesis = self.kernel.execute(
            "synthesize",
            context=context,
            working_memory=self.working_memory,
            components=findings,
            goal=str(input_data)
        )
        
        # Step 5: Verify completeness
        verification = self.kernel.execute(
            "verify",
            context=context,
            working_memory=self.working_memory,
            claim=synthesis.content,
            original_question=str(input_data)
        )
        
        # Step 6: Generate final report
        if verification.confidence > 0.8:
            final_result = synthesis
        else:
            # Need more work - iterate
            additional_analysis = self.kernel.execute(
                "analyze",
                context=context,
                working_memory=self.working_memory,
                data=verification.content
            )
            
            final_result = self.kernel.execute(
                "synthesize",
                context=context,
                working_memory=self.working_memory,
                components=[synthesis.content, additional_analysis.content]
            )
        
        # Store final result
        self.remember(
            content=f"Research complete: {final_result.content[:300]}...",
            importance=0.95,
            tags=["result", "final", "research"]
        )
        
        return final_result
```

### Step 7: Complete Example - Security Analyst Agent

Here's a complete, production-ready example combining all concepts:

```python
from brainary.sdk.template_agent import TemplateAgent, AgentConfig
from brainary.core import ExecutionContext
from brainary.core.metacognitive_monitor import MonitoringLevel
from brainary.memory import ConceptualKnowledge, ProceduralKnowledge
from brainary.primitive.base import PrimitiveResult
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class SecurityAnalystAgent(TemplateAgent):
    """
    Advanced security analyst agent that:
    - Uses working memory to track analysis context
    - Leverages semantic memory for security knowledge
    - Employs metacognition for self-assessment
    - Implements multi-step security analysis process
    """
    
    def __init__(self, name: str = "security_analyst", **kwargs):
        # Create optimized configuration
        config = AgentConfig(
            name=name,
            description="Expert security analyst with threat detection capabilities",
            domain="security",
            working_memory_capacity=12,
            enable_semantic_memory=True,
            enable_metacognition=True,
            monitoring_level=MonitoringLevel.DETAILED,
            enable_learning=True,
            quality_threshold=0.9,  # High quality for security
            default_execution_mode="deep",  # Thorough analysis
            max_token_budget=20000,
            metadata={
                "expertise": ["vulnerability_analysis", "threat_detection", "code_security"],
                "compliance_standards": ["OWASP", "CWE", "CVE"]
            }
        )
        
        super().__init__(name=name, config=config, **kwargs)
        
        # Initialize security knowledge base
        self._initialize_security_knowledge()
        
        logger.info(f"Security analyst agent '{name}' initialized")
    
    def _initialize_security_knowledge(self):
        """Populate semantic memory with security knowledge."""
        
        # Security concepts
        concepts = [
            ("sql_injection", "Code injection attack on SQL databases", 
             ["injection", "database", "web_security"]),
            ("xss", "Cross-site scripting attack via untrusted data",
             ["web_security", "injection", "client_side"]),
            ("csrf", "Cross-site request forgery using victim credentials",
             ["web_security", "session", "authentication"]),
            ("buffer_overflow", "Memory corruption via buffer boundary violation",
             ["memory_safety", "c", "cpp"]),
        ]
        
        for concept, desc, related in concepts:
            self.add_knowledge(ConceptualKnowledge(
                concept=concept,
                description=desc,
                related_concepts=related,
                domain="security"
            ))
        
        # Security analysis procedure
        self.add_knowledge(ProceduralKnowledge(
            task="security_code_review",
            steps=[
                "1. Identify input sources and data flow",
                "2. Check input validation and sanitization",
                "3. Review authentication and authorization",
                "4. Analyze cryptographic implementations",
                "5. Check for common vulnerabilities (OWASP Top 10)",
                "6. Verify secure configuration",
                "7. Review error handling and logging",
                "8. Generate security report with severity ratings"
            ],
            conditions=["source_code_available", "context_understood"],
            expected_outcome="comprehensive security assessment",
            domain="security"
        ))
    
    def process(self, input_data: Any, context: ExecutionContext, **kwargs) -> PrimitiveResult:
        """
        Execute comprehensive security analysis.
        
        Process:
        1. Self-assess capability
        2. Recall past findings
        3. Search security knowledge
        4. Decompose analysis task
        5. Perform detailed analysis
        6. Verify findings
        7. Generate report
        """
        
        logger.info(f"Starting security analysis for: {str(input_data)[:100]}")
        
        # Step 1: Self-assess capability for this specific code/system
        assessment = self._assess_capability(input_data, context)
        
        # Step 2: Recall similar past analyses
        past_findings = self._recall_similar_analyses(input_data)
        
        # Step 3: Search security knowledge base
        security_knowledge = self._gather_security_knowledge(input_data)
        
        # Step 4: Decompose into analysis components
        analysis_plan = self._create_analysis_plan(input_data, context)
        
        # Step 5: Execute detailed analysis
        findings = self._execute_analysis(
            input_data,
            context,
            analysis_plan,
            security_knowledge,
            past_findings
        )
        
        # Step 6: Verify and validate findings
        validated_findings = self._verify_findings(findings, context)
        
        # Step 7: Generate final security report
        report = self._generate_report(validated_findings, context)
        
        # Step 8: Store for future reference
        self._store_analysis_results(input_data, report)
        
        logger.info(f"Security analysis complete. Quality: {report.quality}")
        
        return report
    
    def _assess_capability(self, input_data: Any, context: ExecutionContext) -> PrimitiveResult:
        """Self-assess capability to analyze this input."""
        return self.kernel.execute(
            "self_assess",
            context=context,
            working_memory=self.working_memory,
            task=f"Security analysis of: {str(input_data)[:200]}"
        )
    
    def _recall_similar_analyses(self, input_data: Any) -> List:
        """Recall past security analyses."""
        return self.recall(
            query=f"security analysis {str(input_data)[:100]}",
            top_k=5
        )
    
    def _gather_security_knowledge(self, input_data: Any) -> List:
        """Search security knowledge base."""
        return self.search_knowledge(
            query=str(input_data),
            knowledge_types=["conceptual", "procedural"],
            top_k=10
        )
    
    def _create_analysis_plan(self, input_data: Any, context: ExecutionContext) -> PrimitiveResult:
        """Decompose security analysis into steps."""
        return self.kernel.execute(
            "decompose",
            context=context,
            working_memory=self.working_memory,
            problem=f"Security analysis of: {input_data}"
        )
    
    def _execute_analysis(
        self,
        input_data: Any,
        context: ExecutionContext,
        plan: PrimitiveResult,
        knowledge: List,
        past_findings: List
    ) -> List[Dict]:
        """Execute detailed security analysis."""
        
        findings = []
        
        # Build enhanced context
        knowledge_context = "\n".join([
            f"- {k.to_dict() if hasattr(k, 'to_dict') else str(k)}"
            for k in knowledge[:5]
        ])
        
        past_context = "\n".join([
            f"- {item.content[:100]}..."
            for item in past_findings[:3]
        ])
        
        enhanced_input = f"""
        Security Analysis Request:
        {input_data}
        
        Relevant Security Knowledge:
        {knowledge_context}
        
        Past Similar Findings:
        {past_context}
        
        Perform thorough security analysis following OWASP guidelines.
        """
        
        # Execute deep analysis
        analysis_result = self.kernel.execute(
            "analyze",
            context=context,
            working_memory=self.working_memory,
            data=enhanced_input,
            analysis_type="security"
        )
        
        findings.append({
            "type": "security_analysis",
            "result": analysis_result.content,
            "quality": analysis_result.quality,
            "confidence": analysis_result.confidence
        })
        
        # Store intermediate finding
        self.remember(
            content=f"Security finding: {analysis_result.content[:200]}...",
            importance=analysis_result.quality,
            tags=["security", "finding", "intermediate"]
        )
        
        return findings
    
    def _verify_findings(self, findings: List[Dict], context: ExecutionContext) -> PrimitiveResult:
        """Verify and validate security findings."""
        return self.kernel.execute(
            "verify",
            context=context,
            working_memory=self.working_memory,
            claim=str(findings),
            verification_type="security_assessment"
        )
    
    def _generate_report(self, validated_findings: PrimitiveResult, context: ExecutionContext) -> PrimitiveResult:
        """Generate final security report."""
        return self.kernel.execute(
            "synthesize",
            context=context,
            working_memory=self.working_memory,
            components=[validated_findings.content],
            output_format="security_report"
        )
    
    def _store_analysis_results(self, input_data: Any, report: PrimitiveResult):
        """Store analysis results in memory."""
        self.remember(
            content=f"Security Analysis Complete\nInput: {str(input_data)[:100]}\nReport: {report.content[:300]}",
            importance=0.95,
            tags=["security", "report", "complete", "final"]
        )


# Usage example
if __name__ == "__main__":
    # Create security analyst agent
    agent = SecurityAnalystAgent(name="sec_analyst_001")
    
    # Analyze code
    code_to_analyze = """
    def login(username, password):
        query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
        result = db.execute(query)
        return result
    """
    
    result = agent.run(code_to_analyze)
    
    print("Security Analysis Report:")
    print(result.content)
    print(f"\nQuality: {result.quality}")
    print(f"Confidence: {result.confidence}")
    
    # Check agent stats
    stats = agent.get_stats()
    print(f"\nAgent Stats:")
    print(f"Total runs: {stats['run_count']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
```

### Best Practices for Custom Agents

1. **Memory Management**
   - Store important context in working memory with appropriate importance scores
   - Pre-populate semantic memory with domain knowledge
   - Use tags consistently for efficient retrieval

2. **Metacognition**
   - Always self-assess before complex operations
   - Monitor quality and retry/correct if below threshold
   - Learn from past performance through introspection

3. **Process Design**
   - Break complex tasks into clear steps
   - Validate intermediate results
   - Store progress for continuity across executions

4. **Error Handling**
   - Use try-except around kernel.execute() calls
   - Implement fallback strategies for low-quality results
   - Log important decisions and outcomes

5. **Performance**
   - Adjust token budgets based on task complexity
   - Use appropriate execution modes (fast/deep/adaptive)
   - Monitor stats regularly to detect degradation

6. **Testing**
   - Test with various input types and edge cases
   - Verify memory persistence across multiple runs
   - Check that metacognition adapts appropriately

### Advanced Patterns

**Pattern 1: Chained Agents**
```python
class AnalystAgent(TemplateAgent):
    def process(self, input_data, context, **kwargs):
        return self.kernel.execute("analyze", context=context, data=input_data)

class ReviewerAgent(TemplateAgent):
    def process(self, input_data, context, **kwargs):
        return self.kernel.execute("verify", context=context, claim=input_data)

# Chain them
analyst = AnalystAgent(name="analyst")
reviewer = ReviewerAgent(name="reviewer")

analysis = analyst.run(data)
review = reviewer.run(analysis.content)
```

**Pattern 2: Adaptive Execution Mode**
```python
class AdaptiveAgent(TemplateAgent):
    def process(self, input_data, context, **kwargs):
        # Select mode based on input complexity
        complexity = len(str(input_data))
        mode = "fast" if complexity < 100 else "deep"
        
        return self.kernel.execute(
            "analyze",
            context=context,
            data=input_data,
            execution_mode=mode
        )
```

**Pattern 3: Knowledge Accumulation**
```python
class LearningAgent(TemplateAgent):
    def process(self, input_data, context, **kwargs):
        result = self.kernel.execute("analyze", context=context, data=input_data)
        
        # Extract and store new knowledge
        if result.quality > 0.9:
            self.add_knowledge(FactualKnowledge(
                subject=str(input_data)[:50],
                predicate="analysis_result",
                object=result.content[:200],
                confidence=result.quality,
                domain=self.config.domain
            ))
        
        return result
```

---

## Agent System

Module: `brainary.sdk.agents`

The Agent system provides specialized cognitive agents with role-specific configurations, team coordination, and built-in memory management.

### `AgentRole`

Predefined agent roles with optimized configurations:

```python
from brainary.sdk import AgentRole

class AgentRole(Enum):
    ANALYST = "analyst"      # Data and code analysis
    RESEARCHER = "researcher" # Information gathering and synthesis
    CODER = "coder"          # Code generation and modification
    REVIEWER = "reviewer"     # Code review and quality assessment
    PLANNER = "planner"      # Task planning and decomposition
    WRITER = "writer"        # Content generation
    TEACHER = "teacher"      # Educational explanations
    ASSISTANT = "assistant"  # General-purpose assistance
```

### `AgentConfig`

Configuration dataclass for customizing agent behavior:

```python
@dataclass
class AgentConfig:
    # Identity
    name: str
    role: AgentRole
    domain: str
    description: str
    
    # Cognitive settings
    quality_threshold: float = 0.8
    memory_capacity: int = 7
    enable_learning: bool = True
    
    # Execution preferences
    default_mode: str = "adaptive"
    token_budget: int = 10000
    
    # Behavioral traits
    reasoning_style: str = "analytical"
    attention_focus: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Unique agent identifier |
| `role` | `AgentRole` | required | Agent's primary role |
| `domain` | `str` | required | Domain of expertise |
| `quality_threshold` | `float` | `0.8` | Minimum output quality (0-1) |
| `memory_capacity` | `int` | `7` | Working memory slots |
| `enable_learning` | `bool` | `True` | Enable adaptive learning |
| `default_mode` | `str` | `"adaptive"` | Execution mode |
| `token_budget` | `int` | `10000` | Max tokens per operation |

### `Agent`

Main agent class for creating specialized cognitive agents.

**Creation Methods:**

```python
# From role template
agent = Agent.create("analyst", domain="security")

# From custom config
config = AgentConfig(name="custom", role=AgentRole.ANALYST, domain="finance")
agent = Agent.from_config(config)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `process(task, **kwargs)` | Process task with agent configuration |
| `think(query, **kwargs)` | Execute reasoning |
| `analyze(data, **kwargs)` | Perform analysis |
| `remember(content, **kwargs)` | Store in memory |
| `recall(query, **kwargs)` | Retrieve from memory |
| `get_stats()` | Get performance statistics |

**Example:**
```python
from brainary.sdk import Agent

# Create specialized analyst
analyst = Agent.create(
    "analyst",
    domain="security",
    quality_threshold=0.95
)

# Process with agent context
result = analyst.analyze(source_code)
print(result.content)

# Check stats
stats = analyst.get_stats()
print(f"Tasks completed: {stats['task_count']}")
```

### `AgentTeam`

Coordinate multiple agents for complex tasks with different execution strategies.

**Execution Strategies:**
- `sequential`: Agents process one after another
- `parallel`: Agents process simultaneously  
- `pipeline`: Output chains through agents

**Example:**
```python
from brainary.sdk import Agent, AgentTeam

# Create team
analyst = Agent.create("analyst", domain="security")
coder = Agent.create("coder", domain="python")
reviewer = Agent.create("reviewer", domain="code_quality")

team = AgentTeam("security_team", [analyst, coder, reviewer])

# Execute pipeline
results = team.execute(
    task={"code": source_code, "action": "review"},
    strategy="pipeline"
)
```

---

## Memory Management

Module: `brainary.sdk.memory`

### `MemoryManager`

High-level memory interface with intelligent storage and retrieval.

**Constructor:**
```python
MemoryManager(capacity: int = 7)
```

**Core Methods:**

##### `store()`
```python
store(content: Any, importance: float = 0.5, tags: List[str] = None) -> str
```
Store content with importance scoring and tagging.

##### `retrieve()`
```python
retrieve(query: str = None, tags: List[str] = None, top_k: int = 5) -> List[MemoryItem]
```
Retrieve memories with optional query and tag filters.

##### `search()`
```python
search(query: str, tags: List[str] = None, min_importance: float = 0.0, limit: int = 10) -> List[MemoryItem]
```
Advanced search with multiple filters.

**Example:**
```python
from brainary.sdk import MemoryManager

memory = MemoryManager(capacity=10)

# Store with importance
mem_id = memory.store(
    content="Critical security finding",
    importance=0.9,
    tags=["security", "critical"]
)

# Search with filters
results = memory.search(
    query="security",
    min_importance=0.7,
    limit=5
)

for item in results:
    print(f"[{item.importance}] {item.content}")
```

**Advanced Features:**

- **Consolidation**: Merge similar memories
  ```python
  memory.consolidate(similarity_threshold=0.85)
  ```

- **Decay**: Apply time-based importance decay
  ```python
  memory.apply_decay(decay_rate=0.1)
  ```

- **Snapshots**: Save and restore memory state
  ```python
  snapshot = memory.create_snapshot()
  memory.restore_snapshot(snapshot)
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

---

## SDK Module Reference

This section provides detailed documentation for each SDK module and its components.

### brainary.sdk.client

The main client module provides the `Brainary` class - the primary interface for object-oriented SDK usage.

**Key Features:**
- Automatic kernel and memory initialization
- Built-in primitive registration
- Flexible context management
- Performance tracking and statistics

See [Client API](#client-api-brainary) section for detailed documentation.

### brainary.sdk.primitives

Function-based API that provides standalone functions for all cognitive primitives. Uses a lazily-created global `Brainary` client.

**Module Structure:**
- Core primitives: `perceive`, `think`, `remember`, `recall`, `associate`, `action`
- Composite primitives: `analyze`, `solve`, `decide`, `plan`, `create`, `decompose`, etc.
- Metacognitive primitives: `introspect`, `self_assess`, `select_strategy`, `self_correct`, `reflect`

See [Function-Based Primitives](#function-based-primitives) section for detailed documentation.

### brainary.sdk.memory

Memory management utilities providing high-level interfaces to the memory subsystem.

**Key Class:** `MemoryManager`
- Working memory operations (store, retrieve, search)
- Advanced features (consolidation, decay, snapshots)
- Statistics and monitoring

See [Memory Management](#memory-management) section for detailed documentation.

### brainary.sdk.context

Context management tools for building and scoping execution contexts.

**Key Classes:**
- `ContextBuilder`: Fluent API for building contexts
- `ContextManager`: Context manager for temporary overrides
- `create_context`: Convenience function for context creation

See [Context Helpers](#context-helpers) section for detailed documentation.

### brainary.sdk.agents

Agent system for creating specialized cognitive agents with role-specific configurations.

**Key Classes:**
- `Agent`: Main agent class with role templates
- `AgentConfig`: Configuration dataclass
- `AgentRole`: Predefined role enum
- `AgentTeam`: Multi-agent coordination

See [Agent System](#agent-system) section for detailed documentation.

### brainary.sdk.template_agent

Base templates for creating custom agents with kernel-scoped memories and abstract process hooks.

**Key Classes:**
- `TemplateAgent`: Abstract base class for custom agents
- `SimpleAgent`: Concrete implementation for quick prototyping
- `AgentConfig`: Configuration for template agents

See [Template Agents](#template-agents) section for detailed documentation.

---

## Primitive Function Reference

The following sections detail individual primitive functions available through the SDK.

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

## SDK Usage Patterns

### Pattern 1: Simple Function-Based Usage

Best for: Quick scripts, prototypes, exploratory work

```python
from brainary.sdk import configure, think, remember, recall

# Configure once (optional)
configure(quality_threshold=0.9)

# Use functions directly
answer = think("Explain quantum computing")
remember(answer.content, importance=0.8, tags=["physics"])
memories = recall(query="quantum", limit=5)
```

### Pattern 2: Client-Based Application

Best for: Applications with multiple operations, custom configuration

```python
from brainary.sdk import Brainary

# Create configured client
brain = Brainary(
    enable_learning=True,
    memory_capacity=10,
    quality_threshold=0.9
)

# Use client methods
result = brain.think("Complex problem")
brain.remember(result.content, importance=0.9)

# Monitor performance
stats = brain.get_stats()
print(f"Success rate: {stats['kernel_stats']['success_rate']:.2%}")
```

### Pattern 3: Specialized Agents

Best for: Domain-specific tasks, role-based processing

```python
from brainary.sdk import Agent

# Create specialized agent
analyst = Agent.create(
    "analyst",
    domain="security",
    quality_threshold=0.95
)

# Use agent with domain expertise
result = analyst.analyze(source_code)
print(result.content)
```

### Pattern 4: Multi-Agent Teams

Best for: Complex workflows, multi-stage processing

```python
from brainary.sdk import Agent, AgentTeam

# Create team of specialists
analyst = Agent.create("analyst", domain="security")
coder = Agent.create("coder", domain="python")
reviewer = Agent.create("reviewer", domain="code_quality")

team = AgentTeam("security_team", [analyst, coder, reviewer])

# Execute pipeline
results = team.execute(
    task={"code": source, "action": "security_review"},
    strategy="pipeline"
)
```

### Pattern 5: Custom Template Agents

Best for: Complex custom behaviors, multi-step processes

```python
from brainary.sdk import TemplateAgent
from brainary.core import ExecutionContext
from brainary.primitive.base import PrimitiveResult

class ResearchAgent(TemplateAgent):
    def process(self, input_data, context, **kwargs):
        # Multi-step custom process
        
        # Step 1: Search knowledge base
        knowledge = self.search_knowledge(input_data, top_k=5)
        
        # Step 2: Analyze with context
        analysis = self.kernel.execute(
            "analyze",
            context=context,
            data=input_data,
            knowledge=knowledge
        )
        
        # Step 3: Synthesize findings
        result = self.kernel.execute(
            "synthesize",
            context=context,
            findings=analysis.content
        )
        
        # Step 4: Store in memory
        self.remember(result.content, importance=0.8)
        
        return result

# Use custom agent
agent = ResearchAgent(name="researcher", domain="ml")
result = agent.run("Research latest in transformers")
```

---

## SDK Design Philosophy

### Progressive Complexity

The SDK is designed with layers of abstraction:

1. **Simple Functions** - Zero configuration, immediate use
2. **Client Object** - Configured instances, performance tracking
3. **Agents** - Role-based specialization, memory management
4. **Templates** - Full customization, multi-step workflows

Start simple and graduate to more complex patterns as needs grow.

### Dual API Paradigm

**Functional API:**
- Stateless, global client
- No object management
- Ideal for scripts and notebooks

**Object-Oriented API:**
- Explicit instances
- Configuration control
- Better for applications

Both APIs share the same underlying implementation and can be mixed.

### Memory-Centric Design

All SDK components integrate with Brainary's memory architecture:

- **Working Memory**: Automatic, kernel-scoped, configurable capacity
- **Semantic Memory**: Optional, agent-scoped, unlimited knowledge base
- **Experience Cache**: Automatic, learning-enabled, adaptive optimization

Memory persists across operations within the same kernel/agent instance.

### Quality-First Execution

Every operation includes quality assessment:

- Quality thresholds guide execution mode selection
- Low-quality outputs trigger automatic retries
- Quality scores included in all results
- Learning system optimizes quality over time

---

## Migration Guide

### From Version 0.8.x to 0.9.x

The 0.9.x SDK introduces significant improvements:

**1. Unified Import Pattern**
```python
# Old (0.8.x)
from brainary import BrainaryClient
from brainary.primitives import think, perceive

# New (0.9.x)
from brainary.sdk import Brainary, think, perceive
```

**2. Simplified Client Construction**
```python
# Old (0.8.x)
from brainary import BrainaryClient, ExecutionContext

client = BrainaryClient(llm_provider="openai", model="gpt-4")
context = ExecutionContext(quality_threshold=0.9)

# New (0.9.x)
from brainary.sdk import Brainary

brain = Brainary(quality_threshold=0.9)
```

**3. Enhanced Agent System**
```python
# Old (0.8.x)
# Manual agent configuration required

# New (0.9.x)
from brainary.sdk import Agent

agent = Agent.create("analyst", domain="security")
```

**4. Memory Management**
```python
# Old (0.8.x)
from brainary.memory import WorkingMemory

memory = WorkingMemory()
memory.add_item(...)

# New (0.9.x)
from brainary.sdk import MemoryManager

memory = MemoryManager()
memory.store(content, importance=0.8)
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```python
# Problem: ModuleNotFoundError: No module named 'brainary.sdk'
# Solution: Ensure Brainary 0.9.x or later is installed
pip install --upgrade brainary
```

**2. Memory Capacity Issues**
```python
# Problem: Memory filling up, performance degrading
# Solution: Adjust capacity or clear periodically
from brainary.sdk import Brainary

brain = Brainary(memory_capacity=15)  # Increase capacity
# or
brain.clear_memory()  # Clear periodically
```

**3. Quality Threshold Problems**
```python
# Problem: Operations failing due to quality threshold
# Solution: Adjust threshold or use adaptive mode
brain = Brainary(quality_threshold=0.7)  # Lower threshold
# or
result = brain.think(query, reasoning_mode="adaptive")  # Adaptive mode
```

**4. Token Budget Exhaustion**
```python
# Problem: Token budget exceeded errors
# Solution: Increase budget or optimize operations
brain = Brainary(token_budget=20000)  # Increase budget
```

### Getting Help

- **Documentation**: See `doc/` directory for detailed guides
- **Examples**: Check `examples/` directory for code samples
- **Issues**: Report bugs on GitHub
- **Community**: Join discussions on Discord

---

## Related Documentation

- [Installation Guide](INSTALLATION.md) - Setup and configuration
- [Kernel Execution Loop](KERNEL_EXECUTION_LOOP.md) - Internal execution details
- [Metacognitive Architecture](METACOGNITIVE_ARCHITECTURE.md) - Self-monitoring system
- [LLM Integration](LLM_INTEGRATION.md) - LLM provider configuration
- [Design Document](DESIGN.md) - Overall system architecture

---

*Last updated: December 2, 2025*
*API Version: 0.9.x*
*Brainary SDK Documentation*
