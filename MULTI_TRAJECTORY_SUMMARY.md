# Multi-Trajectory Support for StateLM - Implementation Summary

## Overview

Successfully implemented multi-trajectory support to enable learning from partial trajectories (snapshots) created during StateLM `deleteContext` operations, while maintaining full backward compatibility with existing agent loops.

## Problem Solved

### Issue 1: Single Trajectory Limitation
**Before**: When `deleteContext` was called, trajectory snapshots were created but never returned for learning.
**After**: All snapshots are now returned as separate trajectories, each available for training.

### Issue 2: Independent Trajectory Processing
**Before**: Could not handle different rewards for different parts of the trajectory.
**After**: Each trajectory (snapshot or final) is processed independently with its own reward.

## Files Modified

1. **`verl/experimental/agent_loop/agent_loop.py`**
   - Added `MultiTrajectoryAgentLoopOutput` class (lines 143-147)
   - Modified `_run_agent_loop()` to return `list[_InternalAgentLoopOutput]` (line 492)
   - Added `_process_single_trajectory()` helper method (lines 530-691)
   - Updated `generate_sequences()` to flatten multi-trajectory outputs (lines 475-490)

2. **`verl/experimental/agent_loop/tool_agent_loop.py`**
   - Modified `run()` to collect and return multiple trajectories (lines 308-370)
   - Returns `MultiTrajectoryAgentLoopOutput` when snapshots exist
   - Returns single `AgentLoopOutput` when no snapshots (backward compatible)

## Key Changes

### 1. New Data Structure

```python
class MultiTrajectoryAgentLoopOutput(BaseModel):
    """Agent loop output containing multiple trajectories."""
    trajectories: list[AgentLoopOutput]
```

### 2. Multi-Trajectory Return from ToolAgentLoop

```python
# In ToolAgentLoop.run():
trajectories = []

# Add all snapshot trajectories
for idx, snapshot in enumerate(agent_data.trajectory_snapshots):
    snapshot_output = AgentLoopOutput(...)
    trajectories.append(snapshot_output)

# Add final trajectory
final_output = AgentLoopOutput(...)
trajectories.append(final_output)

# Return appropriate type
if len(trajectories) > 1:
    return MultiTrajectoryAgentLoopOutput(trajectories=trajectories)
else:
    return final_output
```

### 3. Handle Both Single and Multi-Trajectory

```python
# In _run_agent_loop():
output = await agent_loop.run(sampling_params, **kwargs)

# Handle both types
if isinstance(output, MultiTrajectoryAgentLoopOutput):
    trajectories = output.trajectories
else:
    trajectories = [output]

# Process each independently
for traj_output in trajectories:
    processed = await self._process_single_trajectory(traj_output, ...)
```

### 4. Flatten Before Postprocessing

```python
# In generate_sequences():
outputs = await asyncio.gather(*tasks)  # list[list[InternalOutput]]

# Flatten
flattened_outputs = []
for output_list in outputs:
    flattened_outputs.extend(output_list)

output = self._postprocess(flattened_outputs)
```

## Data Flow

### Without deleteContext (Backward Compatible)
```
Sample → ToolAgentLoop.run() → AgentLoopOutput
       → _run_agent_loop() → [_InternalAgentLoopOutput]
       → Flatten → [InternalOutput1]
       → Postprocess → DataProto (batch_size=1)
```

### With deleteContext (StateLM)
```
Sample → ToolAgentLoop.run() → MultiTrajectoryAgentLoopOutput([
           snapshot_1,  # Before first delete
           snapshot_2,  # Before second delete  
           final        # After all deletes
         ])
       → _run_agent_loop() → [
           InternalOutput1_snap1,
           InternalOutput1_snap2,
           InternalOutput1_final
         ]
       → Flatten → [InternalOutput1_snap1, InternalOutput1_snap2, InternalOutput1_final]
       → Postprocess → DataProto (batch_size=3)
```

### Mixed Batch
```
Sample1 (with deleteContext) → [snap1, snap2, final] = 3 trajectories
Sample2 (no deleteContext)   → [final] = 1 trajectory
Sample3 (with deleteContext) → [snap1, final] = 2 trajectories

Flatten → Total 6 trajectories
Postprocess → DataProto (batch_size=6)
```

## Benefits

### 1. **Learning from Partial Trajectories**
- Model learns from states before context deletion
- Understands what led to deletion decision
- Improves context management strategy

### 2. **Independent Reward Attribution**
- Each trajectory has its own reward
- Snapshots can have different rewards than final trajectory
- Supports pre-computed rewards (e.g., context_length_penalty)

### 3. **Efficient Training**
- More training samples from same rollout
- Better signal for context management
- No wasted computation on deleted context

### 4. **Backward Compatibility**
- Existing agent loops work without changes
- Single trajectory case unchanged
- No breaking changes to training pipeline

## Extra Fields

Each trajectory includes in `extra_fields`:

```python
{
    "is_snapshot": bool,           # True for snapshots, False for final
    "snapshot_index": int,         # Index of snapshot (if applicable)
    "turn_scores": list[float],    # Accumulated turn scores
    "tool_rewards": list[float],   # Accumulated tool rewards
}
```

For trajectories with context length penalty:
```python
{
    "context_length_exceeded": True,
    "early_stop_penalty": -1.0,
}
```

## Testing

### Verification Script
Run `verify_multi_trajectory.py` to test:
1. ✓ Single trajectory (backward compatibility)
2. ✓ Multi-trajectory with snapshots
3. ✓ Batch flattening
4. ✓ Context length penalty

```bash
python verify_multi_trajectory.py
# All tests passed! ✓
```

### Integration Testing Needed

1. **With Real Agent Loop**:
   ```bash
   # Test with actual StateLM rollout
   python verify_statelm.py
   ```

2. **With Training Pipeline**:
   - Run one training step with StateLM enabled
   - Verify snapshots are created when deleteContext is called
   - Check that batch size increases appropriately
   - Confirm rewards are computed independently

3. **Mixed Batch**:
   - Include samples with and without deleteContext
   - Verify correct flattening and postprocessing
   - Check final DataProto structure

## Configuration

Enable StateLM multi-trajectory in config:

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      statelm_enabled: true
      context_length_penalty: -1.0
    max_model_length: 8192
```

## Usage Example

### In Tool Definition
No changes needed - `deleteContext` tool already creates snapshots in `tool_agent_loop.py`.

### In Training Code
No changes needed - automatic handling by:
1. `_run_agent_loop()` - detects and processes multi-trajectory
2. `generate_sequences()` - flattens outputs
3. `_postprocess()` - creates batch from all trajectories

### Accessing Trajectory Info
In reward function or training code:

```python
# Check if trajectory is a snapshot
is_snapshot = data.non_tensor_batch["is_snapshot"][i]

# Get snapshot index
if is_snapshot:
    snapshot_idx = data.non_tensor_batch["snapshot_index"][i]

# Get accumulated scores
turn_scores = data.non_tensor_batch["turn_scores"][i]
tool_rewards = data.non_tensor_batch["tool_rewards"][i]
```

## Performance Implications

### Memory
- **Snapshots**: Each snapshot stores full trajectory state
- **Impact**: Minimal - only created on deleteContext
- **Mitigation**: Snapshots are shallow copies when possible

### Computation
- **Processing**: Each trajectory processed independently
- **Reward Computation**: Each trajectory gets own reward call
- **Impact**: Moderate - 2-3x more trajectories with deleteContext
- **Benefit**: Better training signal outweighs cost

### Batch Size
- **Original**: N samples → N trajectories
- **With StateLM**: N samples → M trajectories (M ≥ N)
- **Example**: 8 samples with avg 1.5 snapshots → ~20 trajectories
- **Adjustment**: May need to reduce batch size for memory

## Troubleshooting

### Issue: Batch size larger than expected
**Cause**: Multiple trajectories per sample
**Solution**: Expected behavior - each snapshot becomes a trajectory

### Issue: Reward computation slow
**Cause**: More trajectories to process
**Solution**: Enable async reward computation (already enabled)

### Issue: Memory overflow
**Cause**: Too many snapshots
**Solution**: 
- Reduce number of deleteContext operations
- Lower batch size
- Adjust `max_model_length` to trigger early stopping

### Issue: Backward compatibility broken
**Check**: 
- Single trajectory still returns `AgentLoopOutput`?
- `_run_agent_loop()` handles both types?
- Flattening preserves list structure?

## Future Enhancements

1. **Selective Learning**: Option to learn only from certain snapshots
2. **Trajectory Weighting**: Different learning rates for snapshots vs final
3. **Snapshot Compression**: Store only essential data in snapshots
4. **Rollback Support**: Use snapshots to implement trajectory rollback

## Documentation Files

1. **`STATELM_MULTI_TRAJECTORY_CHANGES.md`**: Detailed implementation guide
2. **`verify_multi_trajectory.py`**: Verification script with tests
3. **`MULTI_TRAJECTORY_SUMMARY.md`**: This summary (quick reference)

## Verification Status

✅ **Code Implementation**: Complete
✅ **Logic Verification**: All tests passed
✅ **Backward Compatibility**: Verified
✅ **Documentation**: Complete

⏳ **Integration Testing**: Needs user validation with real training

## Next Steps

1. **Run integration test**:
   ```bash
   python verify_statelm.py
   ```

2. **Run training step**:
   - Start training with StateLM config
   - Monitor batch sizes
   - Check trajectory creation
   - Verify rewards

3. **Monitor metrics**:
   - Number of snapshots per sample
   - Average trajectory length
   - Reward distribution
   - Training convergence

4. **Adjust if needed**:
   - Batch size tuning
   - Snapshot frequency
   - Reward scaling

## Contact

For issues or questions about this implementation, refer to:
- Implementation details: `STATELM_MULTI_TRAJECTORY_CHANGES.md`
- Original StateLM docs: `STATELM_IMPLEMENTATION.md`
- Usage examples: `STATELM_USAGE_EXAMPLE.md`

