# StateLM Implementation for Tool Agent Loop

## Overview

This document describes the implementation of agentic RL training with `deleteContext` operation in `tool_agent_loop.py`. The implementation supports dynamic conversation history management, trajectory snapshots, context masking, and context length protection.

## Key Features Implemented

### 1. **Conversation History Management**

The system maintains a persistent conversation history (`self.full_history`) that can be modified via the `deleteContext` tool operation.

**Implementation Details:**
- Each message in `full_history` is assigned a unique `msg_id` via `msg_id_counter`
- Deleted message IDs are tracked in `deleted_msg_ids` set
- The `_render_message_view()` method renders messages, replacing deleted content with stubs
- Supports assistant messages with tool calls and tool result messages

**Code Location:** `AgentData` class (lines 84-100)

### 2. **Document Content as Manipulatable Object**

The full document context (`self.document_content`) is stored separately and not directly provided in messages. Instead, the model interacts with it via tool calls (e.g., `buildIndex`, `readChunk`).

**Implementation Details:**
- Document content passed via `kwargs.get("document_content", "")` in `run()` method
- Stored in `AgentData.document_content`
- Available to tools for processing

**Code Location:** Lines 239, 274

### 3. **Trajectory Snapshots on Delete Operations**

When a `deleteContext` operation occurs, the system creates a snapshot of the current trajectory state before processing the deletion.

**Implementation Details:**
- Snapshot includes: `prompt_ids`, `response_ids`, `response_mask`, `response_logprobs`, `full_history`, `deleted_msg_ids`, `emission_views`, `assistant_turn_boundaries`
- Stored in `trajectory_snapshots` list
- Created immediately when `deleteContext` tool is detected

**Code Location:** `_handle_processing_tools_state()` (lines 527-537)

### 4. **Message Masking Strategy**

The implementation supports two types of masking:

#### Normal Masking (No Delete Operation)
- Tool results: `response_mask = 0` (no loss computed)
- Assistant responses: `response_mask = 1` (used for learning)

#### After Delete Operation
- **ALL previous messages**: `response_mask = 0` (no loss computed)
- Tool result confirming deletion: `response_mask = 0`
- Only future assistant responses after delete: `response_mask = 1`

**Implementation Details:**
- After `deleteContext`, all previous tokens in `response_mask` are set to 0
- Ensures no gradient flows through deleted context
- Logged with info message showing number of masked tokens

**Code Location:** `_handle_processing_tools_state()` (lines 538-548)

### 5. **Delete Context Tool Execution**

The `deleteContext` tool is handled specially within the tool execution flow.

**Implementation Details:**
- Extracts `message_ids` from tool arguments
- Returns success response with deleted message IDs in result dict
- No external tool execution needed (handled internally)
- Updates `deleted_msg_ids` set in AgentData
- Triggers snapshot creation and masking

**Code Location:** `_call_tool()` (lines 653-664)

### 6. **Context Length Protection**

Before each generation, the system checks if the trajectory length would exceed the model's maximum context length. If so, it early-stops the rollout with a negative reward.

**Implementation Details:**
- Checks: `len(prompt_ids) + response_length > max_model_length`
- Early terminates with `AgentState.TERMINATED`
- Sets negative reward: `context_length_penalty` (default: -1.0)
- Logs warning and sets metrics: `context_length_exceeded=True`

**Configuration:**
- `max_model_length`: Maximum context length (default: 8192)
- `context_length_penalty`: Penalty reward for exceeding context (default: -1.0)

**Code Location:** `_handle_pending_state()` (lines 355-364)

### 7. **Emission Views and Turn Boundaries**

For causal log-probability computation, the system tracks:
- **Emission Views**: Snapshot of `prompt_ids` when each assistant turn starts
- **Assistant Turn Boundaries**: Start and end indices in `response_mask` for each turn

**Implementation Details:**
- `emission_views`: List of prompt_ids snapshots before each generation
- `assistant_turn_boundaries`: List of (start_idx, end_idx) tuples
- Updated in `_handle_generating_state()` before and after generation

**Code Location:** `_handle_generating_state()` (lines 380-382, 404-407)

## Flow Diagram

### Normal Tool Call Flow
```
PENDING -> GENERATING -> PROCESSING_TOOLS -> GENERATING -> ... -> TERMINATED
   |            |              |
   |            |              +-> Execute tools
   |            |              +-> Add tool results (mask=0)
   |            |              +-> Append to prompt_ids
   |            |
   |            +-> Track emission view
   |            +-> Generate response (mask=1)
   |            +-> Track turn boundaries
   |
   +-> Render message view
   +-> Check context length
```

### Delete Context Flow
```
PENDING -> GENERATING -> PROCESSING_TOOLS -> PENDING -> GENERATING -> ...
   |            |              |                |
   |            |              |                +-> Re-render with deleted msgs
   |            |              |
   |            |              +-> Create snapshot
   |            |              +-> Execute deleteContext
   |            |              +-> Mask ALL previous tokens
   |            |              +-> Update deleted_msg_ids
   |            |              +-> Return to PENDING (not GENERATING)
   |            |
   |            +-> Track emission view
   |            +-> Generate with deleteContext call
```

## Data Structures

### AgentData New Fields

```python
# Conversation history management
self.full_history: list[dict] = []           # Complete message history
self.deleted_msg_ids: set[int] = set()       # IDs of deleted messages
self.msg_id_counter: int = 0                 # Message ID generator

# Document content
self.document_content: Optional[str] = None   # Full document text

# Trajectory tracking
self.emission_views: list[list[int]] = []    # Prompt snapshots per turn
self.assistant_turn_boundaries: list[tuple[int, int]] = []  # Turn ranges

# Snapshot management
self.trajectory_snapshots: list[dict[str, Any]] = []  # Snapshots before deletes
self.had_delete_operation: bool = False      # Flag for delete occurrence
```

### Message Format in full_history

**Assistant Message:**
```python
{
    "role": "assistant",
    "content": [{"text": "response text"}],
    "tool_calls": [
        {
            "id": "call_id",
            "type": "function",
            "function": {"name": "tool_name", "arguments": "json_args"}
        }
    ],
    "msg_id": 0
}
```

**Tool Result Message:**
```python
{
    "role": "tool",
    "content": {"status": "success", "message": "result"},
    "tool_name": "deleteContext",
    "msg_id": 1,
    "msg_id(invoking_assistant)": 0
}
```

## Configuration Parameters

Add these to your config file under `actor_rollout_ref.rollout.multi_turn`:

```yaml
actor_rollout_ref:
  rollout:
    max_model_length: 8192  # Maximum context length for model
    multi_turn:
      statelm_enabled: true  # Enable StateLM features
      context_length_penalty: -1.0  # Penalty for exceeding context
```

## Usage Example

### In Your Rollout Data

```python
data_item = {
    "raw_prompt": [{"role": "user", "content": "Question here"}],
    "document_content": "Full document text...",  # Separate from messages
    "extra_info": {...}
}
```

### DeleteContext Tool Call

The model can call `deleteContext` tool:

```json
{
  "name": "deleteContext",
  "arguments": {
    "message_ids": [0, 1, 2]  // IDs of messages to delete
  }
}
```

Result:
1. Snapshot created with current trajectory state
2. Messages with IDs 0, 1, 2 marked as deleted
3. All previous response_mask values set to 0
4. Message view re-rendered with stubs for deleted messages
5. Trajectory continues with fresh context

## Testing Considerations

1. **Context Length Protection**: Test with trajectories that would exceed `max_model_length`
2. **Delete Operation**: Verify snapshots are created and masking works correctly
3. **Message Rendering**: Ensure deleted messages show as stubs in rendered view
4. **Turn Tracking**: Check emission_views and turn_boundaries are properly recorded
5. **Tool Integration**: Test with real tools that manipulate document_content

## Future Enhancements

1. Support for partial message deletion (delete specific turns, not just full messages)
2. Undo/redo functionality for delete operations using snapshots
3. More sophisticated context compression strategies
4. Dynamic context window management based on importance scores

## Notes

- The `deleteContext` tool is handled internally and doesn't require external tool registration
- When `statelm_enabled=False`, the system behaves as a standard tool agent loop
- Snapshots can be used for debugging or implementing rollback mechanisms
- The masking strategy ensures clean gradient flow only through relevant parts of the trajectory

