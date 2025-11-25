# StateLM Usage Example

## Configuration Setup

### 1. Update Your Config File

```yaml
# config/your_config.yaml

actor_rollout_ref:
  rollout:
    # Context length settings
    prompt_length: 4096
    response_length: 2048
    max_model_length: 8192  # NEW: Maximum context for model
    
    multi_turn:
      # StateLM specific settings
      statelm_enabled: true  # NEW: Enable StateLM features
      context_length_penalty: -1.0  # NEW: Penalty for exceeding context
      
      # Standard settings
      max_user_turns: 10
      max_assistant_turns: 20
      max_parallel_calls: 3
      max_tool_response_length: 2000
      tool_response_truncate_side: "middle"
      format: "function_calling"
      
      # Tool configuration
      tool_config_path: "configs/tools.yaml"
      interaction_config_path: null

data:
  apply_chat_template_kwargs:
    add_generation_prompt: true
```

### 2. Define Tools Configuration

```yaml
# configs/tools.yaml

tools:
  - name: readChunk
    type: custom
    module: verl.tools.document_tools
    class: ReadChunkTool
    
  - name: buildIndex
    type: custom
    module: verl.tools.document_tools
    class: BuildIndexTool
    
  - name: searchDocument
    type: custom
    module: verl.tools.document_tools
    class: SearchDocumentTool
    
  - name: finish
    type: builtin
    description: "Call this when you have completed the task"
    
  # Note: deleteContext is handled internally, no need to define here
```

## Data Preparation

### Dataset Format

```python
# example_dataset.py

def create_dataset_item():
    """Create a dataset item with document content separate from prompt."""
    
    return {
        "raw_prompt": [
            {
                "role": "user",
                "content": "Based on the provided document, answer: What are the key findings?"
            }
        ],
        
        # Document content is NOT in the prompt, but passed separately
        "document_content": """
        Long document text here...
        This can be very long (e.g., 50K tokens).
        The model will interact with it via tools.
        """,
        
        "extra_info": {
            "task_id": "task_001",
            "document_id": "doc_123",
            # Add any interaction kwargs if using interactions
            # "interaction_kwargs": {"name": "interaction_name"}
        },
        
        "tools_kwargs": {
            "readChunk": {
                "create_kwargs": {
                    "chunk_size": 512,
                    "overlap": 50
                }
            }
        }
    }


# Example dataset
dataset = [
    create_dataset_item()
    for _ in range(1000)
]
```

## Execution Flow Example

### Scenario: Document Q&A with Context Management

**Turn 1: Initial Question**
```
User: Based on the provided document, answer: What are the key findings?

<external_memory>
## Available Notes
No notes recorded.
</external_memory>
```

**Turn 2: Model Builds Index**
```
Assistant: [Tool Call: buildIndex]
Arguments: {"document_type": "scientific_paper"}

State:
- msg_id: 0 (assistant message)
- emission_view: [prompt_ids before generation]
- turn_boundaries: [(0, 150)]  # positions in response_mask
- response_mask: [1, 1, 1, ..., 1]  # 150 ones for assistant response
```

**Turn 3: Tool Response**
```
Tool Result: Successfully built index with 1247 chunks.

State:
- msg_id: 1 (tool result)
- msg_id(invoking_assistant): 0
- response_mask: [1, 1, ..., 1, 0, 0, ..., 0]  # tool result masked
```

**Turn 4: Model Reads Chunks**
```
Assistant: [Tool Call: readChunk]
Arguments: {"chunk_ids": [0, 1, 2]}

State:
- msg_id: 2
- emission_view: [prompt_ids snapshot]
- turn_boundaries: [(0, 150), (200, 280)]
- response_mask: [prev..., 1, 1, ..., 1]
```

**Turn 5: Tool Response with Content**
```
Tool Result: 
Chunk 0: "Introduction: This paper presents..."
Chunk 1: "Methods: We conducted experiments..."
Chunk 2: "Results: Our findings show..."

State:
- msg_id: 3
- response_mask: [prev..., 0, 0, ..., 0]  # tool result masked
```

**Turn 6: Model Decides Context is Too Long**
```
Assistant: [Tool Call: deleteContext]
Arguments: {"message_ids": [0, 1, 2, 3]}

State:
- msg_id: 4
- Triggers special handling:
  1. Snapshot created with full trajectory state
  2. Processes deleteContext tool
```

**Turn 7: After Delete Processing**
```
Tool Result: Successfully deleted 4 message(s) from context.

State:
- msg_id: 5
- deleted_msg_ids: {0, 1, 2, 3}
- response_mask: [0, 0, 0, ..., 0]  # ALL previous masked!
- trajectory_snapshots: [snapshot_dict]
- Returns to PENDING (not GENERATING) for re-rendering
```

**Turn 8: Re-render and Continue**
```
The message view is re-rendered with deleted messages shown as stubs:

User: Based on the provided document, answer: What are the key findings?
<external_memory>...</external_memory>