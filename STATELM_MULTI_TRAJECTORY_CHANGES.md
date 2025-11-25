# StateLM Multi-Trajectory Support Implementation

## Overview

This document describes the changes made to support multiple trajectories per rollout in the agent loop training pipeline, specifically for StateLM with `deleteContext` operations.

## Problem Statement

### Issue 1: Single Trajectory Output
- **Original**: `AgentLoopOutput` could only hold one trajectory
- **Problem**: When `deleteContext` is called, we create snapshots that should also be used for learning, but they were never returned
- **Impact**: Partial trajectories before deletion were not used for training

### Issue 2: Independent Trajectory Processing
- **Original**: Postprocessing assumed single trajectory per sample
- **Problem**: Each trajectory (snapshot or final) may have different rewards (e.g., context length penalty)
- **Impact**: Could not handle pre-computed rewards for individual trajectories

## Solution Architecture

### 1. New Data Structure: `MultiTrajectoryAgentLoopOutput`

**Location**: `verl/experimental/agent_loop/agent_loop.py:143-147`

```python
class MultiTrajectoryAgentLoopOutput(BaseModel):
    """Agent loop output containing multiple trajectories (for StateLM with snapshots)."""
    
    trajectories: list[AgentLoopOutput]
    """List of trajectories, including snapshots and final trajectory."""
```

**Purpose**: Wrapper class to hold multiple `AgentLoopOutput` objects when snapshots exist.

### 2. Modified `ToolAgentLoop.run()` Return Logic

**Location**: `verl/experimental/agent_loop/tool_agent_loop.py:308-370`

**Changes**:
1. After state machine terminates, collect all trajectories
2. For each snapshot in `agent_data.trajectory_snapshots`:
   - Extract snapshot data (prompt_ids, response_ids, response_mask, response_logprobs)
   - Create an `AgentLoopOutput` for the snapshot
   - Mark with `is_snapshot=True` and `snapshot_index`
3. Create final trajectory `AgentLoopOutput` with `is_snapshot=False`
4. If snapshots exist (StateLM), return `MultiTrajectoryAgentLoopOutput`
5. Otherwise, return single `AgentLoopOutput` (backward compatible)

**Key Code**:
```python
trajectories = []

# Add snapshot trajectories
if self.statelm_enabled and agent_data.trajectory_snapshots:
    for idx, snapshot in enumerate(agent_data.trajectory_snapshots):
        snapshot_output = AgentLoopOutput(
            prompt_ids=snapshot_prompt_only_ids,
            response_ids=snapshot_response_ids[:self.response_length],
            response_mask=snapshot_response_mask[:self.response_length],
            # ... other fields ...
            extra_fields={
                "is_snapshot": True,
                "snapshot_index": idx,
            },
        )
        trajectories.append(snapshot_output)

# Add final trajectory
final_output = AgentLoopOutput(
    # ... final trajectory data ...
    extra_fields={
        "is_snapshot": False,
    },
)
trajectories.append(final_output)

# Return appropriate type
if self.statelm_enabled and len(trajectories) > 1:
    return MultiTrajectoryAgentLoopOutput(trajectories=trajectories)
else:
    return final_output
```

### 3. Refactored `_run_agent_loop()` to Handle Multiple Trajectories

**Location**: `verl/experimental/agent_loop/agent_loop.py:492-528`

**Changes**:
1. Changed return type from `_InternalAgentLoopOutput` to `list[_InternalAgentLoopOutput]`
2. Detect if output is `MultiTrajectoryAgentLoopOutput` or single `AgentLoopOutput`
3. Extract list of trajectories
4. Process each trajectory independently using helper method
5. Return list of processed outputs

**Key Code**:
```python
async def _run_agent_loop(...) -> list[_InternalAgentLoopOutput]:
    # ... agent loop instantiation ...
    output = await agent_loop.run(sampling_params, **kwargs)
    
    # Handle both single and multi-trajectory outputs
    if isinstance(output, MultiTrajectoryAgentLoopOutput):
        trajectories = output.trajectories
    else:
        trajectories = [output]
    
    # Process each trajectory independently
    processed_outputs = []
    for traj_output in trajectories:
        processed_output = await self._process_single_trajectory(
            traj_output, kwargs, trajectory
        )
        processed_outputs.append(processed_output)
    
    return processed_outputs
```

### 4. New Helper Method: `_process_single_trajectory()`

**Location**: `verl/experimental/agent_loop/agent_loop.py:530-691`

**Purpose**: Extract common trajectory processing logic into reusable method

**Responsibilities**:
1. Tokenization and padding (prompt_ids, response_ids, response_mask)
2. Multi-modal input processing (Qwen2VL support)
3. Position IDs and attention mask computation
4. Async reward computation (if not pre-computed)
5. Create `_InternalAgentLoopOutput` with padded tensors

**Key Point**: Each trajectory is processed independently, allowing:
- Different prompt/response lengths per trajectory
- Independent reward computation per trajectory
- Separate attention masks and position IDs

### 5. Updated `generate_sequences()` to Flatten Outputs

**Location**: `verl/experimental/agent_loop/agent_loop.py:475-490`

**Changes**:
1. Each task in `outputs` now returns a list of trajectories
2. Flatten the nested list structure before postprocessing
3. Pass flattened list to `_postprocess()`

**Key Code**:
```python
outputs = await asyncio.gather(*tasks)

# Flatten outputs: each sample may have multiple trajectories (for StateLM snapshots)
flattened_outputs = []
for output_list in outputs:
    flattened_outputs.extend(output_list)

output = self._postprocess(flattened_outputs)
```

### 6. `_postprocess()` Remains Unchanged

**Location**: `verl/experimental/agent_loop/agent_loop.py:693-757`

**Why**: The method already handles a flat list of `_InternalAgentLoopOutput`, so no changes needed. It will now process more trajectories (including snapshots) but treats them all uniformly.

## Data Flow Example

### Without StateLM (Backward Compatible)
```
Sample 1 → ToolAgentLoop.run() → AgentLoopOutput
         → _run_agent_loop() → [_InternalAgentLoopOutput]
         
Sample 2 → ToolAgentLoop.run() → AgentLoopOutput
         → _run_agent_loop() → [_InternalAgentLoopOutput]

Flatten → [InternalOutput1, InternalOutput2]
       → _postprocess() → DataProto (batch_size=2)
```

### With StateLM and deleteContext
```
Sample 1 → ToolAgentLoop.run() → MultiTrajectoryAgentLoopOutput([
             snapshot_1, snapshot_2, final
           ])
         → _run_agent_loop() → [
             InternalOutput1_snap1,
             InternalOutput1_snap2,
             InternalOutput1_final
           ]
         
Sample 2 → ToolAgentLoop.run() → AgentLoopOutput (no deleteContext)
         → _run_agent_loop() → [InternalOutput2]

Flatten → [
    InternalOutput1_snap1,
    InternalOutput1_snap2,
    InternalOutput1_final,
    InternalOutput2
]
       → _postprocess() → DataProto (batch_size=4)
```

## Benefits

### 1. Independent Trajectory Learning
- Each snapshot trajectory can be used for training
- Snapshots have their own rewards (can be pre-computed)
- Enables learning from partial trajectories before context deletion

### 2. Proper Reward Attribution
- Snapshots can have context_length_penalty if they would exceed limits
- Final trajectory has its own reward based on final outcome
- Each trajectory's reward is computed independently

### 3. Efficient Context Management Training
- Model learns from both:
  - Trajectories before deletion (what to delete)
  - Trajectories after deletion (how to continue)
- Encourages learning efficient context management strategies

### 4. Backward Compatibility
- Non-StateLM agents still return single `AgentLoopOutput`
- Single trajectory case works exactly as before
- No breaking changes to existing agent loops

## Key Implementation Details

### Snapshot Creation Timing
**Location**: `verl/experimental/agent_loop/tool_agent_loop.py:544-556`

Snapshots are created in `_handle_processing_tools_state()` when `deleteContext` tool is detected, **before** masking occurs:

```python
if self.statelm_enabled and editor_tool_call:
    # Step 1: Create snapshot of current trajectory state
    snapshot = {
        "prompt_ids": copy.deepcopy(agent_data.prompt_ids),
        "response_ids": copy.deepcopy(agent_data.response_ids),
        "response_mask": copy.deepcopy(agent_data.response_mask),
        "response_logprobs": copy.deepcopy(agent_data.response_logprobs),
        # ... other fields ...
    }
    agent_data.trajectory_snapshots.append(snapshot)
    
    # Step 2: Mask ALL previous messages
    # Step 3: Mark messages as deleted
```

### Reward Computation Strategy
**Location**: `verl/experimental/agent_loop/agent_loop.py:646-676`

Each trajectory can have:
1. **Pre-computed reward**: Set in `output.reward_score` (e.g., context_length_penalty)
2. **Async reward**: Computed independently per trajectory if not pre-set
3. **No reward**: Computed later in training pipeline (if `enable_async_reward=False`)

### Extra Fields Tracking
Each trajectory's `extra_fields` includes:
- `is_snapshot`: Boolean indicating if this is a snapshot trajectory
- `snapshot_index`: Integer index of snapshot (if applicable)
- `turn_scores`: Scores accumulated up to this point
- `tool_rewards`: Tool rewards accumulated up to this point

## Testing Considerations

### Test Cases to Add

1. **Single trajectory (no deleteContext)**:
   - Verify backward compatibility
   - Check output is single `AgentLoopOutput`, not multi

2. **Multiple trajectories (with deleteContext)**:
   - Verify snapshots are created correctly
   - Check each snapshot has proper data
   - Verify final trajectory is included

3. **Reward computation**:
   - Test pre-computed rewards (context_length_penalty)
   - Test async reward computation per trajectory
   - Verify rewards are independent

4. **Batch processing**:
   - Mix samples with and without deleteContext
   - Verify flattening works correctly
   - Check final batch size is correct

5. **Edge cases**:
   - Multiple deleteContext in single rollout
   - deleteContext at different positions
   - Empty snapshots (should not happen but handle gracefully)

## Future Enhancements

1. **Trajectory Weighting**: Allow different learning rates for snapshots vs final trajectory
2. **Selective Snapshot Learning**: Option to learn only from certain snapshots based on quality
3. **Snapshot Compression**: Store only essential data in snapshots to reduce memory
4. **Rollback Support**: Use snapshots to implement trajectory rollback during training

## Migration Guide

### For Existing Agent Loops
No changes required - single trajectory output still works.

### For New StateLM Agent Loops
1. Enable StateLM: Set `statelm_enabled=True` in config
2. Use `trajectory_snapshots` to store state before deletions
3. Return `MultiTrajectoryAgentLoopOutput` when snapshots exist
4. Mark trajectories with `is_snapshot` in extra_fields

### For Training Pipeline
No changes required - the flattening and postprocessing handle everything automatically.

## Conclusion

This implementation enables StateLM to learn from partial trajectories created by `deleteContext` operations, while maintaining full backward compatibility with existing agent loops. Each trajectory is processed independently, allowing proper reward attribution and efficient context management training.

