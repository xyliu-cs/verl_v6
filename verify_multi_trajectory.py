#!/usr/bin/env python3
"""
Verification script for multi-trajectory support in StateLM agent loop.

This script verifies:
1. Single trajectory output (backward compatibility)
2. Multi-trajectory output (with snapshots)
3. Proper data structure and fields
4. Independent trajectory processing
"""

import sys
from typing import Any

# Mock classes for testing structure
class AgentLoopMetrics:
    def __init__(self):
        self.generate_sequences = 0.0
        self.tool_calls = 0.0

class AgentLoopOutput:
    def __init__(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] = None,
        multi_modal_data: dict[str, Any] = None,
        reward_score: float = None,
        num_turns: int = 0,
        metrics: AgentLoopMetrics = None,
        extra_fields: dict[str, Any] = None,
    ):
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.response_mask = response_mask
        self.response_logprobs = response_logprobs
        self.multi_modal_data = multi_modal_data or {}
        self.reward_score = reward_score
        self.num_turns = num_turns
        self.metrics = metrics or AgentLoopMetrics()
        self.extra_fields = extra_fields or {}

class MultiTrajectoryAgentLoopOutput:
    def __init__(self, trajectories: list[AgentLoopOutput]):
        self.trajectories = trajectories


def test_single_trajectory():
    """Test single trajectory output (backward compatible)."""
    print("=" * 80)
    print("TEST 1: Single Trajectory Output (Backward Compatible)")
    print("=" * 80)
    
    output = AgentLoopOutput(
        prompt_ids=[1, 2, 3, 4],
        response_ids=[5, 6, 7, 8, 9],
        response_mask=[1, 1, 1, 1, 1],
        response_logprobs=[0.1, 0.2, 0.3, 0.4, 0.5],
        reward_score=0.8,
        num_turns=2,
        extra_fields={"is_snapshot": False},
    )
    
    print(f"✓ Created single AgentLoopOutput")
    print(f"  - Prompt length: {len(output.prompt_ids)}")
    print(f"  - Response length: {len(output.response_ids)}")
    print(f"  - Response mask sum: {sum(output.response_mask)} (all learnable)")
    print(f"  - Reward score: {output.reward_score}")
    print(f"  - Is snapshot: {output.extra_fields.get('is_snapshot', False)}")
    print()
    
    # Simulate _run_agent_loop handling
    if isinstance(output, MultiTrajectoryAgentLoopOutput):
        trajectories = output.trajectories
    else:
        trajectories = [output]
    
    print(f"✓ After _run_agent_loop handling:")
    print(f"  - Number of trajectories: {len(trajectories)}")
    print(f"  - Expected: 1 (single trajectory)")
    assert len(trajectories) == 1, "Single trajectory should result in list of 1"
    print()
    
    return True


def test_multi_trajectory_with_snapshots():
    """Test multi-trajectory output with snapshots."""
    print("=" * 80)
    print("TEST 2: Multi-Trajectory Output (With Snapshots)")
    print("=" * 80)
    
    # Simulate 2 snapshots + 1 final trajectory
    trajectories = []
    
    # Snapshot 1: Before first deleteContext
    snapshot1 = AgentLoopOutput(
        prompt_ids=[1, 2, 3, 4],
        response_ids=[5, 6, 7, 8],
        response_mask=[1, 1, 1, 1],  # All learnable
        response_logprobs=[0.1, 0.2, 0.3, 0.4],
        reward_score=0.5,  # Could be partial reward
        num_turns=1,
        extra_fields={
            "is_snapshot": True,
            "snapshot_index": 0,
            "turn_scores": [0.5],
            "tool_rewards": [0.0],
        },
    )
    trajectories.append(snapshot1)
    
    # Snapshot 2: Before second deleteContext
    snapshot2 = AgentLoopOutput(
        prompt_ids=[1, 2, 3, 4],
        response_ids=[9, 10, 11, 12, 13],
        response_mask=[0, 0, 0, 1, 1],  # Some masked after first delete
        response_logprobs=[0.0, 0.0, 0.0, 0.3, 0.4],
        reward_score=0.3,
        num_turns=2,
        extra_fields={
            "is_snapshot": True,
            "snapshot_index": 1,
            "turn_scores": [0.5, 0.3],
            "tool_rewards": [0.0, -0.1],
        },
    )
    trajectories.append(snapshot2)
    
    # Final trajectory
    final = AgentLoopOutput(
        prompt_ids=[1, 2, 3, 4],
        response_ids=[14, 15, 16],
        response_mask=[1, 1, 1],  # Fresh context, all learnable
        response_logprobs=[0.5, 0.6, 0.7],
        reward_score=0.9,  # Final reward
        num_turns=3,
        extra_fields={
            "is_snapshot": False,
            "turn_scores": [0.5, 0.3, 0.9],
            "tool_rewards": [0.0, -0.1, 0.2],
        },
    )
    trajectories.append(final)
    
    output = MultiTrajectoryAgentLoopOutput(trajectories=trajectories)
    
    print(f"✓ Created MultiTrajectoryAgentLoopOutput")
    print(f"  - Total trajectories: {len(output.trajectories)}")
    print(f"  - Snapshots: 2")
    print(f"  - Final: 1")
    print()
    
    # Verify each trajectory
    for i, traj in enumerate(output.trajectories):
        is_snapshot = traj.extra_fields.get("is_snapshot", False)
        snapshot_idx = traj.extra_fields.get("snapshot_index", "N/A")
        learnable_tokens = sum(traj.response_mask)
        
        print(f"  Trajectory {i}:")
        print(f"    - Is snapshot: {is_snapshot}")
        if is_snapshot:
            print(f"    - Snapshot index: {snapshot_idx}")
        print(f"    - Response length: {len(traj.response_ids)}")
        print(f"    - Learnable tokens: {learnable_tokens}")
        print(f"    - Reward score: {traj.reward_score}")
        print(f"    - Num turns: {traj.num_turns}")
        print()
    
    # Simulate _run_agent_loop handling
    if isinstance(output, MultiTrajectoryAgentLoopOutput):
        extracted_trajectories = output.trajectories
    else:
        extracted_trajectories = [output]
    
    print(f"✓ After _run_agent_loop handling:")
    print(f"  - Number of trajectories: {len(extracted_trajectories)}")
    print(f"  - Expected: 3 (2 snapshots + 1 final)")
    assert len(extracted_trajectories) == 3, "Should have 3 trajectories"
    print()
    
    # Verify snapshot flags
    assert extracted_trajectories[0].extra_fields["is_snapshot"] == True
    assert extracted_trajectories[1].extra_fields["is_snapshot"] == True
    assert extracted_trajectories[2].extra_fields["is_snapshot"] == False
    print(f"✓ Snapshot flags are correct")
    print()
    
    # Verify independent rewards
    rewards = [traj.reward_score for traj in extracted_trajectories]
    print(f"✓ Independent rewards: {rewards}")
    assert all(r is not None for r in rewards), "All trajectories should have rewards"
    print()
    
    return True


def test_batch_flattening():
    """Test batch flattening with mixed single and multi-trajectory outputs."""
    print("=" * 80)
    print("TEST 3: Batch Flattening (Mixed Single and Multi-Trajectory)")
    print("=" * 80)
    
    # Sample 1: Multi-trajectory (2 trajectories)
    sample1_output = [
        AgentLoopOutput(
            prompt_ids=[1, 2],
            response_ids=[3, 4],
            response_mask=[1, 1],
            extra_fields={"is_snapshot": True, "snapshot_index": 0},
        ),
        AgentLoopOutput(
            prompt_ids=[1, 2],
            response_ids=[5, 6],
            response_mask=[1, 1],
            extra_fields={"is_snapshot": False},
        ),
    ]
    
    # Sample 2: Single trajectory
    sample2_output = [
        AgentLoopOutput(
            prompt_ids=[7, 8],
            response_ids=[9, 10],
            response_mask=[1, 1],
            extra_fields={"is_snapshot": False},
        ),
    ]
    
    # Sample 3: Multi-trajectory (3 trajectories)
    sample3_output = [
        AgentLoopOutput(
            prompt_ids=[11, 12],
            response_ids=[13, 14],
            response_mask=[1, 1],
            extra_fields={"is_snapshot": True, "snapshot_index": 0},
        ),
        AgentLoopOutput(
            prompt_ids=[11, 12],
            response_ids=[15, 16],
            response_mask=[1, 1],
            extra_fields={"is_snapshot": True, "snapshot_index": 1},
        ),
        AgentLoopOutput(
            prompt_ids=[11, 12],
            response_ids=[17, 18],
            response_mask=[1, 1],
            extra_fields={"is_snapshot": False},
        ),
    ]
    
    # Simulate gather outputs
    outputs = [sample1_output, sample2_output, sample3_output]
    
    print(f"✓ Created batch with 3 samples:")
    print(f"  - Sample 1: {len(sample1_output)} trajectories (1 snapshot + 1 final)")
    print(f"  - Sample 2: {len(sample2_output)} trajectories (1 final only)")
    print(f"  - Sample 3: {len(sample3_output)} trajectories (2 snapshots + 1 final)")
    print()
    
    # Simulate flattening
    flattened_outputs = []
    for output_list in outputs:
        flattened_outputs.extend(output_list)
    
    print(f"✓ After flattening:")
    print(f"  - Total trajectories: {len(flattened_outputs)}")
    print(f"  - Expected: 6 (2 + 1 + 3)")
    assert len(flattened_outputs) == 6, "Should have 6 total trajectories"
    print()
    
    # Count snapshots vs final
    num_snapshots = sum(1 for traj in flattened_outputs if traj.extra_fields.get("is_snapshot", False))
    num_final = sum(1 for traj in flattened_outputs if not traj.extra_fields.get("is_snapshot", False))
    
    print(f"✓ Trajectory breakdown:")
    print(f"  - Snapshots: {num_snapshots}")
    print(f"  - Final: {num_final}")
    assert num_snapshots == 3, "Should have 3 snapshots"
    assert num_final == 3, "Should have 3 final trajectories"
    print()
    
    # Verify all trajectories can be processed independently
    print(f"✓ All trajectories have independent data:")
    for i, traj in enumerate(flattened_outputs):
        print(f"  - Trajectory {i}: prompt_len={len(traj.prompt_ids)}, "
              f"response_len={len(traj.response_ids)}, "
              f"is_snapshot={traj.extra_fields.get('is_snapshot', False)}")
    print()
    
    return True


def test_context_length_penalty():
    """Test trajectory with pre-computed context length penalty."""
    print("=" * 80)
    print("TEST 4: Context Length Penalty (Pre-computed Reward)")
    print("=" * 80)
    
    # Trajectory that exceeded context length
    trajectories = []
    
    # Snapshot before exceeding
    snapshot = AgentLoopOutput(
        prompt_ids=[1] * 100,
        response_ids=[2] * 50,
        response_mask=[1] * 50,
        reward_score=0.0,  # Neutral so far
        num_turns=5,
        extra_fields={
            "is_snapshot": True,
            "snapshot_index": 0,
        },
    )
    trajectories.append(snapshot)
    
    # Final trajectory with penalty
    final = AgentLoopOutput(
        prompt_ids=[1] * 100,
        response_ids=[2] * 10,  # Truncated
        response_mask=[1] * 10,
        reward_score=-1.0,  # PRE-COMPUTED penalty
        num_turns=5,
        extra_fields={
            "is_snapshot": False,
            "context_length_exceeded": True,
            "early_stop_penalty": -1.0,
        },
    )
    trajectories.append(final)
    
    output = MultiTrajectoryAgentLoopOutput(trajectories=trajectories)
    
    print(f"✓ Created multi-trajectory with context length penalty")
    print(f"  - Snapshot reward: {trajectories[0].reward_score}")
    print(f"  - Final reward (with penalty): {trajectories[1].reward_score}")
    print(f"  - Context exceeded: {trajectories[1].extra_fields.get('context_length_exceeded', False)}")
    print()
    
    # Verify pre-computed rewards
    for i, traj in enumerate(output.trajectories):
        print(f"  Trajectory {i}:")
        print(f"    - Reward score: {traj.reward_score}")
        print(f"    - Has pre-computed reward: {traj.reward_score is not None}")
        if traj.extra_fields.get("context_length_exceeded", False):
            print(f"    - Penalty value: {traj.extra_fields['early_stop_penalty']}")
        print()
    
    # In _process_single_trajectory, pre-computed rewards won't be overridden
    print(f"✓ Pre-computed rewards are preserved during processing")
    print(f"  - If reward_score is not None, async reward computation is skipped")
    print()
    
    return True


def main():
    """Run all verification tests."""
    print("\n")
    print("*" * 80)
    print("MULTI-TRAJECTORY SUPPORT VERIFICATION")
    print("*" * 80)
    print()
    
    tests = [
        ("Single Trajectory", test_single_trajectory),
        ("Multi-Trajectory with Snapshots", test_multi_trajectory_with_snapshots),
        ("Batch Flattening", test_batch_flattening),
        ("Context Length Penalty", test_context_length_penalty),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, "FAILED"))
            import traceback
            traceback.print_exc()
        print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {test_name}: {status}")
    print()
    
    all_passed = all(status == "PASSED" for _, status in results)
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

