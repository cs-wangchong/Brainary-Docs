# Brainary Introduction

Brainary is a programmable intelligence kernel. Instead of chaining prompts
through a black-box agent, you compose cognitive primitives—`perceive`, `think`,
`analyze`, `plan`, `decide`, `remember`—inside a transparent execution loop with
working memory, semantic memory, and metacognition built in. Think of it as a
logic board for reasoning systems: wire primitives together, pick execution
modes, and let the kernel learn which strategies work best.

## The General Idea

1. **Cognitive Kernel** routes every primitive call through experience cache,
   knowledge rules, heuristics, and a semantic fallback to pick the best
   executor (Direct LLM, ReAct, LangGraph, etc.).
2. **Execution Contexts** carry program metadata (domain, quality target,
   token budget, execution mode) so the kernel knows how cautious or fast to be.
3. **Memory Tiers** keep the system grounded: `WorkingMemory` handles short-term
   context (7±2 slots) while optional `SemanticMemory` stores durable knowledge
   for template agents.
4. **Primitives + Template Agents** form the building blocks. Stay at the SDK
   level (`Brainary` client / primitive functions) or create agents that wrap a
   custom `process(...)` with persistent memory and metacognition.

## Quick Demo (≈5 minutes)

> Run these commands from the repo root to see the kernel in action.

### 1. Environment

```bash
git clone https://github.com/cs-wangchong/Brainary brainary
cd brainary
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY="sk-..."  # or another provider key
```

### 2. Minimal SDK Script

```python
# demo.py
from brainary.sdk import Brainary

brain = Brainary(quality_threshold=0.85, memory_capacity=7)

question = "Outline a lightweight incident response playbook"
result = brain.think(question)

print("Question:", question)
print("Answer:\n", result.content)
print("Confidence:", result.confidence.overall)
```

Run it:

```bash
python demo.py
```

The kernel selects an execution strategy, returns the answer, and reports a
confidence score. Call `brain.get_stats()` afterward to inspect routing and
success rate.

### 3. Function Wrapper (Optional)

```python
from brainary.sdk import configure, plan

configure(memory_capacity=9)
print(plan("Ship Q4 release", constraints=["no downtime"]).content)
```

The function API reuses a singleton `Brainary` client—ideal for notebooks.

### 4. Template Agent Snapshot

```python
from brainary.sdk.template_agent import TemplateAgent
from brainary.primitive.base import PrimitiveResult

class ResearchAgent(TemplateAgent):
    def process(self, input_data, context, **kwargs) -> PrimitiveResult:
        outline = self.kernel.execute("plan", context=context, goal=input_data)
        return self.kernel.execute(
            "synthesize", context=context, components=[outline.content]
        )

agent = ResearchAgent(name="analyst", domain="strategy")
print(agent.run("Summarize LLM tooling").content)
```

Template agents keep their own working/semantic memory, so repeated runs build a
knowledge base automatically.

## Where to Go Next

- **Quickstart** – `docs/QUICKSTART.md` for additional runnable snippets.
- **SDK Guide** – `docs/SDK_GUIDE.md` for architecture deep dives and patterns.
- **API Reference** – `docs/API_REFERENCE.md` for every method signature.
- **User Manual** – `docs/USER_MANUAL.md` for end-to-end scenarios.
- **Template Example** – `tpl/java_security_detector/README.md` for a complete
  multi-agent security pipeline demo.

Brainary is evolving quickly. Share feedback or ideas via issues/PRs so we can
continue refining the primitives, executors, and documentation together.
