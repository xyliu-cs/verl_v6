# Summary of Changes for StateLM Implementation

## Files Modified
- `verl/experimental/agent_loop/tool_agent_loop.py`

## Key Code Changes

### 1. AgentData Class Enhancements

**Added Fields:**
```python
# Line 56: Added document_content parameter to __init__
document_content: Optional[str] = None

# Lines 85-98: New tracking fields
self.full_history: list[dict] = copy.deepcopy(self.messages)
self.deleted_msg_ids: set[int] = set()
self.msg_id_counter: int = 0
self.emission_views: list[list[int]] = []
self.assistant_turn_boundaries: list[tuple[int, int]] = []
self.trajectory_snapshots: list[dict[str, Any]] = []
self.had_delete_operation: bool = False
```

**Existing Method:**
- `_render_message_view()` - Already implemented to render messages with deleted content replaced by stubs

### 2. ToolAgentLoop Class Configuration

**Added Class Variables (lines 230-236):**
```python
cls.statelm_enabled = config.actor_rollout_ref.rollout.multi_turn.get("statelm_enabled", False)
cls.max_model_length = config.actor_rollout_ref.rollout.get("max_model_length", 8192)
cls.context_length_penalty = config.actor_rollout_ref.rollout.multi_turn.get("context_length_penalty", -1.0)
```

### 3. Context Length Protection

**In `_handle_pending_state()` (lines 355-364):**
```python
# Context length protection: check if trajectory exceeds max model length
if self.statelm_enabled and len(agent_data.prompt_ids) + self.response_length > self.max_model_length:
    logger.warning(
        f"Trajectory length {len(agent_data.prompt_ids)} + response_length {self.response_length} "
        f"exceeds max_model_length {self.max_model_length}. Early stopping with penalty."
    )
    agent_data.metrics["context_length_exceeded"] = True
    agent_data.metrics["early_stop_penalty"] = self.context_length_penalty
    agent_data.turn_scores.append(self.context_length_penalty)
    return AgentState.TERMINATED
```

### 4. Emission Views and Turn Boundary Tracking

**In `_handle_generating_state()` (lines 380-382, 404-407):**
```python
# Before generation
if self.statelm_enabled:
    emission_view = copy.deepcopy(agent_data.prompt_ids)
    agent_data.emission_views.append(emission_view)
    turn_start_idx = len(agent_data.response_mask)

# After generation
if self.statelm_enabled:
    turn_end_idx = len(agent_data.response_mask)
    agent_data.assistant_turn_boundaries.append((turn_start_idx, turn_end_idx))
```

### 5. Full History Tracking

**In `_handle_generating_state()` (lines 428-437):**
```python
if self.statelm_enabled:
    # Append assistant message to full_history
    tool_calls_copy = copy.deepcopy(agent_data.tool_calls)
    agent_data.full_history.append({
        "role": "assistant",
        "content": [{"text": text_response}],
        "tool_calls": [tc.model_dump() for tc in tool_calls_copy],
        "msg_id": agent_data.msg_id_counter,
    })
    agent_data.msg_id_counter += 1
```

### 6. Delete Context Tool Handling

**In `_handle_processing_tools_state()` (lines 453-458):**
```python
finish_tool_call = False
editor_tool_call = False
delete_msg_ids = []  # Collect message IDs to delete

for tool_response, tool_reward, tool_result_dict, tool_name in responses:
    # ... message processing ...
    
    if tool_name == 'finish':
        finish_tool_call = True
    elif tool_name == 'deleteContext':
        editor_tool_call = True
        agent_data.had_delete_operation = True
        if tool_result_dict and "deleted_msg_ids" in tool_result_dict:
            delete_msg_ids.extend(tool_result_dict["deleted_msg_ids"])
```

### 7. Snapshot and Masking Logic

**In `_handle_processing_tools_state()` (lines 527-548):**
```python
# Handle deleteContext operation: snapshot, mask, and delete
if self.statelm_enabled and editor_tool_call:
    # Step 1: Create snapshot
    snapshot = {
        "prompt_ids": copy.deepcopy(agent_data.prompt_ids),
        "response_ids": copy.deepcopy(agent_data.response_ids),
        "response_mask": copy.deepcopy(agent_data.response_mask),
        "response_logprobs": copy.deepcopy(agent_data.response_logprobs) if agent_data.response_logprobs else None,
        "full_history": copy.deepcopy(agent_data.full_history),
        "deleted_msg_ids": copy.deepcopy(agent_data.deleted_msg_ids),
        "emission_views": copy.deepcopy(agent_data.emission_views),
        "assistant_turn_boundaries": copy.deepcopy(agent_data.assistant_turn_boundaries),
    }
    agent_data.trajectory_snapshots.append(snapshot)
    
    # Step 2: Mask ALL previous messages
    if len(agent_data.response_mask) > 0:
        current_length = len(agent_data.response_mask)
        for i in range(current_length):
            agent_data.response_mask[i] = 0
    
    # Step 3: Mark messages as deleted
    for msg_id in delete_msg_ids:
        agent_data.deleted_msg_ids.add(msg_id)
    
    logger.info(f"DeleteContext operation: masked {len(agent_data.response_mask)} previous tokens, "
               f"deleted message IDs: {delete_msg_ids}")
```

### 8. Tool Call Method Updates

**In `_call_tool()` signature (line 637):**
```python
async def _call_tool(
    self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
) -> tuple[ToolResponse, float, dict, str]:
```

**In `_call_tool()` - DeleteContext Handling (lines 653-664):**
```python
# Handle deleteContext tool specially
if tool_name == "deleteContext" and self.statelm_enabled:
    # Extract message IDs to delete from arguments
    msg_ids_to_delete = tool_args.get("message_ids", [])
    if not isinstance(msg_ids_to_delete, list):
        msg_ids_to_delete = [msg_ids_to_delete]
    
    # Return success response with deleted message IDs
    return (
        ToolResponse(
            text=f"Successfully deleted {len(msg_ids_to_delete)} message(s) from context.",
        ),
        0.0,  # No immediate reward
        {"deleted_msg_ids": msg_ids_to_delete},
        tool_name,
    )
```

### 9. Tool Result Tracking in Full History

**In `_handle_processing_tools_state()` (lines 477-485):**
```python
# Add tool result to full_history
invoking_assistant_id = agent_data.msg_id_counter - 1  # The last message should be assistant
agent_data.full_history.append({
    "role": "tool",
    "content": message["content"],
    "tool_name": tool_name,
    "msg_id": agent_data.msg_id_counter,
    "msg_id(invoking_assistant)": invoking_assistant_id
})
agent_data.msg_id_counter += 1
```

## Breaking Changes

### Method Signature Changes
1. `_call_tool()`: Now requires `agent_data` parameter and returns 4-tuple instead of 3-tuple
   - Old: `(ToolResponse, float, dict)`
   - New: `(ToolResponse, float, dict, str)` - added tool_name

2. `AgentData.__init__()`: Added `document_content` parameter

## Configuration Changes Required

Add to your config YAML:

```yaml
actor_rollout_ref:
  rollout:
    max_model_length: 8192  # New: maximum context length
    multi_turn:
      statelm_enabled: true  # New: enable StateLM features
      context_length_penalty: -1.0  # New: penalty for exceeding context
      # ... existing multi_turn config ...
```

## Backward Compatibility

- When `statelm_enabled = False`, the agent loop functions as before
- All StateLM features are gated behind `self.statelm_enabled` checks
- Existing functionality is preserved for non-StateLM use cases

## Testing Checklist

- [ ] Test with `statelm_enabled = False` to ensure backward compatibility
- [ ] Test with `statelm_enabled = True` and verify context protection
- [ ] Test deleteContext tool with single and multiple message IDs
- [ ] Test trajectory snapshots are created correctly
- [ ] Test masking logic (normal and after delete)
- [ ] Test context length protection with long trajectories
- [ ] Test emission views and turn boundaries tracking
- [ ] Test document_content parameter passing

## Lines of Code Changed

- **Added:** ~180 lines
- **Modified:** ~30 lines
- **Total Impact:** ~210 lines across 1 file

## Performance Considerations

1. **Snapshot Creation**: Creates deep copies on each deleteContext operation
   - Impact: O(n) where n is trajectory length
   - Mitigation: Only occurs on explicit delete operations

2. **Message Rendering**: `_render_message_view()` processes full history each time
   - Impact: O(m) where m is number of messages
   - Mitigation: Only called in PENDING state before generation

3. **Context Length Check**: Simple comparison before each generation
   - Impact: O(1)
   - Mitigation: None needed, minimal overhead

## Notes

- All StateLM features are opt-in via configuration
- The implementation is thread-safe (no shared state between rollouts)
- Snapshots can be used for debugging or implementing rollback features
- The masking strategy is conservative (masks everything after delete) to ensure clean training

