# Memory Debugging Guide

## Overview

The Brainary memory system includes built-in debugging capabilities that allow you to inspect memory operations in real-time. This is useful for:

- Understanding memory behavior during development
- Debugging memory-related issues
- Monitoring memory consolidation and eviction
- Tracking item promotion across memory tiers

## Enabling Debug Output

### Global Control

Debug output is controlled at the class level and affects all `WorkingMemory` instances:

```python
from brainary.memory.working_memory import WorkingMemory

# Enable debug output
WorkingMemory.enable_debug()

# Your memory operations will now print debug info
memory = WorkingMemory()
memory.store("Important data", importance=0.9)

# Disable debug output
WorkingMemory.disable_debug()

# Operations are now silent
memory.store("Silent data", importance=0.5)
```

## What Gets Logged

### 1. Store Operations

When storing items, you'll see:
- Item ID
- Content (truncated if long)
- Importance level
- Tags
- Source primitive
- Current tier size

**Example:**
```
[MEMORY UPDATE - L1] Store:
  Item ID: 339440cf-f545-428d-ab1f-63a0265ffc82
  Content: Python is a programming language
  Importance: 0.8
  Tags: ['language']
  Source: None
  L1 Size: 1/5
```

### 2. Retrieve Operations

When retrieving items, you'll see:
- Query parameters
- Number of results found
- Details for each result:
  - Item ID
  - Content preview
  - Which tier it was found in
  - Importance and activation scores
  - Access count
  - Whether it was promoted to L1

**Example:**
```
[MEMORY RETRIEVE] Query:
  Query: None
  Tags: ['language']
  Tier: None
  Top-k: 1
  Min Importance: 0.0
  Results Found: 1
  - Item ID: 770cd492-db8f-4a9c-9cc5-46f4d6769203
    Content: Python is a programming language
    Found in: L1
    Importance: 0.80
    Activation: 0.63
    Access Count: 1
```

### 3. Eviction Operations

When L1 is full and items need to be evicted:
- Which item is being evicted
- Its activation and importance scores
- Where it's being moved (L2 or discarded)

**Example:**
```
[MEMORY EVICTION - L1]:
  Evicting: 265019f1...
  Content: Third low item...
  Activation: 0.21
  Importance: 0.30
  Access Count: 0
  → Moved to L2
  New L1 Size: 2/3
```

### 4. Consolidation Operations

When memory consolidates from L1 to L2/L3:
- How many items are being consolidated
- Destination tier for each item
- Reason for destination (importance, associations)
- Summary of consolidation results

**Example:**
```
[MEMORY CONSOLIDATION] Starting:
  L1 Items: 8
  Items to consolidate: 3
  → L3: 339440cf... (importance=0.90, assocs=2)
     Content: High importance item...
  → L2: 3d95a8f3... (importance=0.50)
     Content: Medium importance item...
  → DISCARD: 265019f1... (importance=0.20)
[MEMORY CONSOLIDATION] Complete:
  To L2: 1
  To L3: 1
  Discarded: 1
  New L1 Size: 5/5
  New L2 Size: 10/100
  New L3 Size: 5
```

### 5. Promotion Operations

When items are promoted from L2/L3 to L1:
- Source tier
- Item details
- Whether eviction was needed

**Example:**
```
[MEMORY PROMOTION] L2 → L1:
  Item ID: cb08a246...
  Content: Episodic memory item...
  Importance: 0.60
  Activation: 0.45
  L1 Full - triggering eviction
  New L1 Size: 5/5
```

### 6. Update Operations

When updating existing items:
- Item ID
- Which fields are being updated
- Old and new values

**Example:**
```
[MEMORY UPDATE] Item:
  Item ID: 339440cf...
  Updates: ['importance', 'tags']
  importance: 0.9 → 0.95
  tags: ['important', 'test'] → ['important', 'test', 'updated']
```

## Memory State Summary

Use `print_memory_state()` to get a comprehensive view of the current memory state:

```python
# Brief summary
memory.print_memory_state()

# Detailed summary with item listings
memory.print_memory_state(verbose=True)
```

**Output:**
```
======================================================================
[MEMORY STATE SUMMARY]
======================================================================

L1 (Working Memory): 3/5 items
  - 770cd492... | Act: 0.63 | Imp: 0.80
    Content: Python is a programming language...
  - cdaaf806... | Act: 0.49 | Imp: 0.70
    Content: Machine learning uses algorithms...
  - c260bc1b... | Act: 0.42 | Imp: 0.60
    Content: Data structures organize information...

L2 (Episodic Memory): 1/100 items
  - 265019f1... | Imp: 0.30 | Access: 0
    Content: Third low item...

L3 (Semantic Memory): 0 items

Statistics:
  Total Stores: 4
  Total Retrievals: 1
  L1 Evictions: 1
  L2 Consolidations: 0
  Cache Hit Rate: 100.0%
======================================================================
```

## Best Practices

### 1. Development vs Production

Enable debug output during development and testing, but disable it in production:

```python
import os

# Enable debug in development
if os.getenv('DEBUG') == '1':
    WorkingMemory.enable_debug()
```

### 2. Selective Debugging

Enable debug only for specific sections of code:

```python
# Normal operations (no debug)
memory.store("data1")
memory.store("data2")

# Debug critical section
WorkingMemory.enable_debug()
memory.consolidate()
memory.print_memory_state(verbose=True)
WorkingMemory.disable_debug()

# Continue without debug
memory.store("data3")
```

### 3. Log File Output

Redirect debug output to a file:

```python
import sys

# Redirect stdout to file
with open('memory_debug.log', 'w') as f:
    sys.stdout = f
    WorkingMemory.enable_debug()
    
    # Your operations
    memory.store("test")
    memory.retrieve(tags=["test"])
    
    sys.stdout = sys.__stdout__  # Restore stdout
```

### 4. Understanding Memory Behavior

Use debug output to understand:

- **Why items are evicted**: Check activation scores
- **Where items go**: Track L1 → L2 → L3 movement
- **Cache efficiency**: Monitor hit rates
- **Memory pressure**: Watch L1 size and eviction frequency

## Complete Example

```python
from brainary.memory.working_memory import WorkingMemory, MemoryTier

# Enable debugging
WorkingMemory.enable_debug()

# Create memory with small capacity to see eviction
memory = WorkingMemory(capacity=3)

print("\n### Storing items ###")
memory.store("First item", importance=0.9, tags=["high"])
memory.store("Second item", importance=0.5, tags=["medium"])
memory.store("Third item", importance=0.3, tags=["low"])

print("\n### Triggering eviction ###")
memory.store("Fourth item", importance=0.8, tags=["new"])

print("\n### Retrieval with promotion ###")
# Store directly to L2
id_l2 = memory.store("L2 item", tier=MemoryTier.L2_EPISODIC, importance=0.6)
# Retrieve it (should promote to L1)
memory.retrieve(query=id_l2)

print("\n### Manual consolidation ###")
memory.consolidate()

print("\n### Final state ###")
memory.print_memory_state(verbose=True)

# Disable when done
WorkingMemory.disable_debug()
```

## Troubleshooting

### Issue: Too much output

**Solution**: Disable debug for routine operations, enable only when needed.

### Issue: Can't see memory changes

**Solution**: Use `print_memory_state(verbose=True)` to see detailed snapshots.

### Issue: Need to debug specific tier

**Solution**: Filter output by looking for tier-specific markers (`[MEMORY UPDATE - L1]`, etc.).

### Issue: Understanding eviction decisions

**Solution**: Check the activation scores shown in eviction messages - lower activation items are evicted first.

## See Also

- [Memory System Architecture](../docs/USER_MANUAL.md#memory-system)
- [Working Memory API](../docs/API_REFERENCE.md#memory-api)
