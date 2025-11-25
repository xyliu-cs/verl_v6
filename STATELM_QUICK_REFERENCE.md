# StateLM Tools Quick Reference

## Quick Start

```bash
# 1. Configure Elasticsearch (optional)
export ES_HOST="https://localhost:9200"
export ES_API_KEY="your-key"

# 2. Test the tools
python examples/test_statelm_tools_simple.py

# 3. Use in your config
tool_config_path: verl/tools/configs/statelm_tools_config.yaml
```

## Tool Cheat Sheet

### Document Analysis
```python
# Check document size
analyzeText()
→ {"file_name": "...", "total_tokens": 15000}

# Load full document (use carefully!)
loadDocument()
→ {"document_content": "..."}
```

### Document Indexing
```python
# Build searchable index
buildIndex({"chunk_size": 3000, "overlap": 200})
→ {"total_chunks": 42, "first_chunk_id": 0, "last_chunk_id": 41}

# Read specific chunk
readChunk({"chunk_id": 5})
→ {"retrieved_chunk": [{...}], "chunk_id": 5}

# Search by keywords
searchEngine({"keyword": "neural networks, deep learning"})
→ {"retrieved_chunks": [{...}], "keywords": [...]}
```

### Note Management
```python
# Create note
note({"key": "findings", "summary": "Key points", "content": {...}})
→ {"status": "success", "key": "findings"}

# Read note
readNote({"key": "findings"})
→ {"summary": "...", "full_content": "..."}

# Update note
updateNote({"key": "findings", "mode": "append", "new_content": {...}})
→ {"status": "success", "message": "Note appended"}

# Merge notes
mergeNotes({"keys": ["note1", "note2"], "new_key": "combined"})
→ {"status": "success", "new_key": "combined", "merged_from": [...]}
```

### Context Management
```python
# Check remaining budget
checkBudget()
→ {"conv_rounds": 12, "available_tokens": 4500, "available_rounds": 38}

# Get full statistics
getContextStats()
→ {"total_notes": 5, "index_chunks": 42, "document_size": 15000, ...}
```

### Completion
```python
# Submit final answer
finish({"answer": "The answer is..."})
→ {"final_answer": "The answer is..."}
```

## Common Workflows

### Workflow 1: Search and Answer
```
1. buildIndex({"chunk_size": 3000})
2. searchEngine({"keyword": "your topic"})
3. readChunk({"chunk_id": 5})
4. finish({"answer": "..."})
```

### Workflow 2: Progressive Analysis
```
1. analyzeText()
2. buildIndex({...})
3. Loop:
   a. searchEngine({...})
   b. readChunk({...})
   c. note({...})  # Save findings
4. readNote({...})  # Review findings
5. finish({...})
```

### Workflow 3: Memory Management
```
1. checkBudget()  # Check space
2. note({...})    # Save important info
3. mergeNotes({...})  # Consolidate if needed
4. Continue analysis...
```

## Configuration Templates

### Minimal Config
```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      statelm_enabled: true
      tool_config_path: verl/tools/configs/statelm_tools_config.yaml
```

### Production Config
```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      statelm_enabled: true
      tool_config_path: verl/tools/configs/statelm_tools_config.yaml
      max_assistant_turns: 50
      max_user_turns: 50
      max_tool_response_length: 4000
    max_model_length: 8192
    context_length_penalty: -1.0
```

### Elasticsearch Env Vars
```bash
# Required
export ES_HOST="https://localhost:9200"

# Auth (choose one)
export ES_API_KEY="your-api-key"
# OR
export ES_USER="elastic" ES_PASS="password"

# Optional
export ES_CA_CERT="/path/to/ca.crt"
export ES_INDEX_NAME="my_index"
```

## Parameter Reference

### buildIndex
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| chunk_size | int | 4000 | Tokens per chunk |
| overlap | int | 0 | Overlapping tokens |

### searchEngine
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| keyword | string | required | Keywords (comma-separated) |
| mode | string | "or" | "and" or "or" |
| size | int | 50 | Max results |

### note
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| key | string | required | Unique identifier |
| summary | string | required | Short description |
| content | object | required | Full content |

### updateNote
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| key | string | required | Note to update |
| mode | string | required | "append"/"overwrite"/"delete" |
| new_content | object | - | New content |
| new_summary | string | - | New summary |

## Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| "State manager not available" | StateLM not enabled | Set `statelm_enabled: true` |
| "Index not built" | Search before indexing | Call `buildIndex` first |
| "Chunk_id out of range" | Invalid chunk ID | Check `total_chunks` |
| "Note not found" | Invalid note key | Use `getContextStats` |
| "Elasticsearch query failed" | ES connection issue | Check ES_HOST and auth |

## Best Practices

✅ **DO**:
- Call `buildIndex` before searching
- Use notes to save important findings
- Check budget periodically
- Use overlap for important documents
- Start with broad searches, then narrow

❌ **DON'T**:
- Load full document unless necessary
- Create too many small chunks
- Ignore budget warnings
- Skip error checking
- Forget to call `finish`

## Performance Tips

### For Small Documents (<10K tokens)
```python
buildIndex({"chunk_size": 2000, "overlap": 100})
```

### For Medium Documents (10K-100K tokens)
```python
buildIndex({"chunk_size": 3000, "overlap": 200})
# Use Elasticsearch for search
```

### For Large Documents (>100K tokens)
```python
buildIndex({"chunk_size": 4000, "overlap": 0})
# Aggressive note-taking
# Regular budget checks
```

## Troubleshooting Quick Fixes

### Out of Memory
1. Reduce `chunk_size`
2. Clear old notes
3. Increase system memory

### Slow Search
1. Check Elasticsearch status
2. Reduce `size` parameter
3. Use more specific keywords

### Context Length Exceeded
1. Call `checkBudget` regularly
2. Use `mergeNotes` to consolidate
3. Delete unnecessary notes

## File Locations

```
verl/
├── tools/
│   ├── statelm_tools.py          # Tool implementations
│   ├── configs/
│   │   └── statelm_tools_config.yaml  # Tool registry
│   └── STATELM_TOOLS_README.md   # Full documentation
├── experimental/
│   └── agent_loop/
│       └── tool_agent_loop.py    # Integration point
examples/
└── test_statelm_tools_simple.py  # Test script
```

## Help & Documentation

- **Full Docs**: `verl/tools/STATELM_TOOLS_README.md`
- **Integration Guide**: `STATELM_INTEGRATION_GUIDE.md`
- **Implementation Details**: `STATELM_IMPLEMENTATION_SUMMARY.md`
- **Test Example**: `examples/test_statelm_tools_simple.py`

## Quick Debug Commands

```bash
# Test Elasticsearch
curl -X GET "$ES_HOST"

# Check index
curl -X GET "$ES_HOST/lc_agent_document"

# Enable debug logging
export VERL_LOGGING_LEVEL=DEBUG

# Run simple test
python examples/test_statelm_tools_simple.py
```

---

**For more details, see the full documentation in `STATELM_TOOLS_README.md`**

