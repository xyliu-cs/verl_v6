# StateLM Tools Integration Guide

## Overview

This guide explains the integration of StateLM tools from `orchestrator_es.py` into the VERL framework. The implementation maintains full compatibility with the original tool specifications while adapting them to VERL's distributed architecture.

## What's Been Implemented

### 1. Core Components

#### `verl/tools/statelm_tools.py` (846 lines)
Complete implementation of all 12 StateLM tools:
- **Document Tools**: `AnalyzeTextTool`, `LoadDocumentTool`, `BuildIndexTool`
- **Retrieval Tools**: `ReadChunkTool`, `SearchEngineTool`
- **Note Management**: `NoteTool`, `ReadNoteTool`, `UpdateNoteTool`, `MergeNotesTool`
- **Context Tools**: `CheckBudgetTool`, `GetContextStatsTool`
- **Completion**: `FinishTool`

#### `DocStateManager` Class
Centralized state management for:
- Document content and tokenization
- Chunk indexing (local + Elasticsearch)
- Search history
- Elasticsearch connection pooling

### 2. Integration Points

#### Modified: `verl/experimental/agent_loop/tool_agent_loop.py`

**Changes Made**:

1. **Added state_manager to AgentData** (Line ~108):
   ```python
   self.state_manager = None  # StateLM State Manager for document operations
   ```

2. **State Manager Initialization** (Line ~303):
   ```python
   # Initialize StateLM state manager if document_content is provided
   if self.statelm_enabled and document_content:
       from verl.tools.statelm_tools import DocStateManager
       agent_data.state_manager = DocStateManager(self.tokenizer, document_content)
   ```

3. **Enhanced _call_tool Method** (Line ~744-763):
   - Detects StateLM tools by name
   - Passes required context (agent_data, state_manager, tokenizer, etc.) via kwargs
   - Maintains backward compatibility with non-StateLM tools

4. **Cleanup on Completion** (Line ~375):
   ```python
   # Cleanup state manager if needed
   if self.statelm_enabled and agent_data.state_manager:
       agent_data.state_manager.clear_current_document()
   ```

### 3. Configuration

#### `verl/tools/configs/statelm_tools_config.yaml`
Complete tool registry configuration with:
- All 12 tools registered
- Proper schema definitions matching `tools_qwen_full_update.json`
- Native tool type configuration

#### `verl/tools/__init__.py`
Exports all StateLM tools for easy importing:
```python
from verl.tools import AnalyzeTextTool, NoteTool, SearchEngineTool
```

### 4. Documentation

#### `verl/tools/STATELM_TOOLS_README.md`
Comprehensive documentation including:
- Tool catalog with all parameters
- Usage patterns and best practices
- Configuration examples
- Troubleshooting guide
- Performance considerations

#### `examples/test_statelm_tools_simple.py`
Working example demonstrating:
- Tool initialization
- Basic operations
- State management
- Note-taking workflow

## Architecture Comparison

### Original (orchestrator_es.py)

```
Orchestrator
├── StateManager (notes)
└── ToolLibrary (all tools as methods)
    ├── document state
    ├── index state
    └── Elasticsearch connection
```

### New VERL Implementation

```
ToolAgentLoop
└── AgentData
    ├── notes (dict)
    └── state_manager (DocStateManager)
        ├── document content
        ├── index
        └── Elasticsearch connection

Individual Tool Classes
├── AnalyzeTextTool
├── BuildIndexTool
├── NoteTool
└── ... (each inheriting from BaseTool)
```

## Key Design Decisions

### 1. State Management Strategy

**Decision**: Two-tier state management
- **AgentData.notes**: Persistent note storage (was StateManager.notes)
- **AgentData.state_manager**: Document/index operations (was ToolLibrary)

**Rationale**:
- Maintains separation between note-taking and document operations
- Allows notes to persist even without document content
- Enables independent testing of each component

### 2. Tool Communication Pattern

**Decision**: Pass context via `**kwargs` in execute()

**Implementation**:
```python
exec_kwargs = {
    'agent_data': agent_data,
    'state_manager': agent_data.state_manager,
    'tokenizer': self.tokenizer,
    'tool_schemas': self.tool_schemas,
    # ... other context
}
tool.execute(instance_id, parameters, **exec_kwargs)
```

**Rationale**:
- Non-invasive: Doesn't break existing tool interfaces
- Flexible: Easy to add new context parameters
- Optional: Non-StateLM tools ignore extra kwargs

### 3. Elasticsearch Integration

**Decision**: Maintain original ES implementation with optional fallback

**Features**:
- Same ES configuration via environment variables
- TLS support for ES 9.x
- Graceful degradation if ES unavailable (local index only)
- Per-trajectory document IDs prevent collisions

### 4. Tool Categorization

**Stateful Tools** (modify persistent state):
- `note`, `updateNote`, `mergeNotes` → modify AgentData.notes
- `buildIndex` → modifies state_manager.index

**Query Tools** (read-only):
- `analyzeText`, `loadDocument`, `readChunk`, `searchEngine`
- `readNote`, `checkBudget`, `getContextStats`

**Special Tools**:
- `finish` → triggers termination
- `deleteContext` → handled by agent loop (already implemented)

## Migration from orchestrator_es.py

### For Tool Users

**Before (orchestrator_es.py)**:
```python
orchestrator = Orchestrator(config, document, temp, tokenizer)
orchestrator.run(user_query)
```

**After (VERL)**:
```python
agent_loop = ToolAgentLoop(config, tokenizer, processor)
output = await agent_loop.run(
    sampling_params=params,
    raw_prompt=messages,
    document_content=document,  # NEW
    tools_kwargs={}
)
```

### For Tool Developers

**Before**:
```python
class ToolLibrary:
    def buildIndex(self, params):
        # Implementation
        return result
```

**After**:
```python
class BuildIndexTool(BaseTool):
    async def execute(self, instance_id, parameters, **kwargs):
        state_manager = kwargs.get('state_manager')
        # Implementation
        return ToolResponse(text=json.dumps(result)), 0.0, {}
```

## Integration Checklist

### For Existing Projects

- [ ] Update config to include `statelm_enabled: true`
- [ ] Set `tool_config_path` to StateLM config
- [ ] Configure Elasticsearch environment variables
- [ ] Pass `document_content` in rollout kwargs
- [ ] Update system prompts to reference StateLM tools
- [ ] Test with sample document

### For New Projects

- [ ] Copy `verl/tools/configs/statelm_tools_config.yaml`
- [ ] Set up Elasticsearch (or use local-only mode)
- [ ] Configure environment variables
- [ ] Review `STATELM_TOOLS_README.md`
- [ ] Run `examples/test_statelm_tools_simple.py`
- [ ] Adapt for your use case

## Performance Considerations

### Memory Usage

**Per Trajectory**:
- Document tokenization: ~4x document size (with offsets)
- Chunk index: ~1.2x document size
- Notes: Variable (user-controlled)
- Elasticsearch: External (shared)

**Optimization Tips**:
1. Use larger chunk sizes for big documents
2. Consolidate notes with `mergeNotes`
3. Clear old notes with `updateNote(mode="delete")`
4. Enable Elasticsearch for better search performance

### Computational Cost

**One-time Costs**:
- Document tokenization: O(n) - done at initialization
- Index building: O(n) - done on first `buildIndex`

**Per-Query Costs**:
- Search: O(log n) via Elasticsearch BM25
- Read chunk: O(1) index lookup
- Note operations: O(1) dictionary access

### Scaling Recommendations

**Small Documents (<10K tokens)**:
- Local index only (ES optional)
- Chunk size: 2000-3000 tokens
- In-memory storage sufficient

**Medium Documents (10K-100K tokens)**:
- Use Elasticsearch for search
- Chunk size: 3000-4000 tokens
- Monitor memory usage

**Large Documents (>100K tokens)**:
- Elasticsearch required
- Chunk size: 4000-5000 tokens
- Use note-taking aggressively
- Consider document chunking at data prep stage

## Testing

### Unit Tests

Run the simple test:
```bash
python examples/test_statelm_tools_simple.py
```

Expected output:
```
[1] Initializing tokenizer...
[2] Initializing StateLM state manager...
   Document size: 245 tokens
[3] Testing analyzeText tool...
   Result: {"file_name": "attached_document.txt", "total_tokens": 245}
...
All tests completed successfully! ✓
```

### Integration Tests

Test with full agent loop:
```bash
# Set up environment
export ES_HOST="https://localhost:9200"
export ES_API_KEY="your-key"

# Run agent with StateLM tools
python your_agent_script.py \
    --config configs/statelm_config.yaml \
    --document path/to/document.txt
```

### Elasticsearch Health Check

```bash
# Check ES is running
curl -X GET "$ES_HOST" -H "Authorization: ApiKey $ES_API_KEY"

# Check index exists
curl -X GET "$ES_HOST/lc_agent_document" -H "Authorization: ApiKey $ES_API_KEY"
```

## Troubleshooting

### Common Issues

**1. "State manager not available"**

**Symptoms**: Tools fail with this error
**Causes**:
- `statelm_enabled` not set to true
- `document_content` not passed to agent loop
- State manager initialization failed

**Solutions**:
```yaml
# In config
actor_rollout_ref:
  rollout:
    multi_turn:
      statelm_enabled: true  # Must be true
```

```python
# In code
await agent_loop.run(
    document_content=your_document,  # Must be provided
    ...
)
```

**2. Elasticsearch Connection Errors**

**Symptoms**: "Elasticsearch query failed" errors
**Solutions**:
- Verify ES_HOST is correct and accessible
- Check authentication (ES_API_KEY or ES_USER/ES_PASS)
- Test connection: `curl -X GET "$ES_HOST"`
- Check CA certificate path if using TLS
- Fall back to local-only mode (tools will work without ES)

**3. Memory Issues**

**Symptoms**: OOM errors, slow performance
**Solutions**:
- Reduce chunk_size in buildIndex
- Use smaller documents
- Increase swap space
- Clear old notes regularly
- Monitor with `getContextStats`

**4. Tool Not Found**

**Symptoms**: "Tool 'xyz' not found"
**Solutions**:
- Verify tool is in config YAML
- Check class_name is correct
- Ensure config file is loaded
- Check tool name matches schema

## Future Enhancements

### Potential Improvements

1. **Async Elasticsearch**
   - Use async ES client for better performance
   - Non-blocking search operations

2. **Caching Layer**
   - Cache frequently accessed chunks
   - LRU cache for search results

3. **Distributed State**
   - Redis-backed state for multi-worker scenarios
   - Shared note storage across trajectories

4. **Advanced Search**
   - Semantic search with embeddings
   - Hybrid BM25 + vector search
   - Query expansion and reranking

5. **Monitoring**
   - Metrics collection for tool usage
   - Performance dashboards
   - Cost tracking (token usage)

### Contributing

To add new StateLM tools:

1. Create tool class in `statelm_tools.py`
2. Inherit from `BaseTool`
3. Implement required methods
4. Add to `statelm_tools_config.yaml`
5. Update `__init__.py` exports
6. Document in README
7. Add tests
8. Update this guide

## References

- **Original Implementation**: `orchestrator_es.py`
- **Tool Specifications**: `tools_qwen_full_update.json`
- **Base Tool Interface**: `verl/tools/base_tool.py`
- **Example Tool**: `verl/tools/search_tool.py`
- **Agent Loop**: `verl/experimental/agent_loop/tool_agent_loop.py`

## Contact & Support

For issues or questions:
1. Check this guide and README
2. Review example scripts
3. Check existing issues
4. Open new issue with:
   - Error messages
   - Configuration
   - Minimal reproduction

## License

Copyright 2025 Bytedance Ltd. and/or its affiliates
Licensed under the Apache License, Version 2.0

---

**Last Updated**: 2025-11-25
**Version**: 1.0.0
**Status**: Production Ready ✓

