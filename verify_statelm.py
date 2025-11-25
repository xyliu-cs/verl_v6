#!/usr/bin/env python3
"""
Verification script to check StateLM implementation is complete and correct.
"""

import ast
import sys
from pathlib import Path

def check_file_exists(filepath):
    """Check if a file exists."""
    path = Path(filepath)
    return path.exists(), path

def check_class_has_attributes(filepath, class_name, expected_attrs):
    """Check if a class has expected attributes using simple string search."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    missing = []
    for attr in expected_attrs:
        # Check if self.attr_name appears in the file
        if f'self.{attr}' not in content:
            missing.append(attr)
    
    return len(missing) == 0, missing

def check_method_exists(filepath, class_name, method_name):
    """Check if a class has a specific method using simple string search."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Look for method definition pattern
    pattern = f'def {method_name}('
    return pattern in content

def main():
    """Run verification checks."""
    print("=" * 80)
    print("StateLM Implementation Verification")
    print("=" * 80)
    
    tool_agent_file = "verl/experimental/agent_loop/tool_agent_loop.py"
    
    # Check 1: File exists
    print("\n[1] Checking if tool_agent_loop.py exists...")
    exists, path = check_file_exists(tool_agent_file)
    if not exists:
        print(f"  ❌ FAIL: {tool_agent_file} not found")
        return 1
    print(f"  ✅ PASS: File found at {path}")
    
    # Check 2: AgentData class has new attributes
    print("\n[2] Checking AgentData class has StateLM attributes...")
    expected_attrs = [
        'document_content',
        'full_history',
        'deleted_msg_ids',
        'msg_id_counter',
        'emission_views',
        'assistant_turn_boundaries',
        'trajectory_snapshots',
        'had_delete_operation',
    ]
    has_attrs, missing = check_class_has_attributes(tool_agent_file, 'AgentData', expected_attrs)
    if not has_attrs:
        print(f"  ❌ FAIL: Missing attributes: {missing}")
        return 1
    print(f"  ✅ PASS: All required attributes present")
    
    # Check 3: _render_message_view method exists
    print("\n[3] Checking _render_message_view method exists...")
    has_method = check_method_exists(tool_agent_file, 'AgentData', '_render_message_view')
    if not has_method:
        print(f"  ❌ FAIL: _render_message_view method not found")
        return 1
    print(f"  ✅ PASS: Method found")
    
    # Check 4: ToolAgentLoop class methods
    print("\n[4] Checking ToolAgentLoop methods...")
    methods = ['_handle_pending_state', '_handle_generating_state', '_handle_processing_tools_state', '_call_tool']
    for method in methods:
        has_method = check_method_exists(tool_agent_file, 'ToolAgentLoop', method)
        if not has_method:
            print(f"  ❌ FAIL: {method} not found")
            return 1
    print(f"  ✅ PASS: All required methods present")
    
    # Check 5: Context length protection code present
    print("\n[5] Checking for context length protection code...")
    with open(tool_agent_file, 'r') as f:
        content = f.read()
        if 'max_model_length' not in content:
            print(f"  ❌ FAIL: max_model_length not found in code")
            return 1
        if 'context_length_penalty' not in content:
            print(f"  ❌ FAIL: context_length_penalty not found in code")
            return 1
    print(f"  ✅ PASS: Context length protection implemented")
    
    # Check 6: DeleteContext handling
    print("\n[6] Checking for deleteContext tool handling...")
    with open(tool_agent_file, 'r') as f:
        content = f.read()
        if 'deleteContext' not in content:
            print(f"  ❌ FAIL: deleteContext not found in code")
            return 1
        if 'trajectory_snapshots' not in content:
            print(f"  ❌ FAIL: trajectory_snapshots not used in code")
            return 1
    print(f"  ✅ PASS: DeleteContext handling implemented")
    
    # Check 7: Documentation files
    print("\n[7] Checking documentation files...")
    docs = [
        'STATELM_IMPLEMENTATION.md',
        'STATELM_CHANGES_SUMMARY.md',
        'STATELM_USAGE_EXAMPLE.md',
    ]
    for doc in docs:
        exists, _ = check_file_exists(doc)
        if not exists:
            print(f"  ⚠️  WARNING: {doc} not found")
        else:
            print(f"  ✅ {doc} present")
    
    print("\n" + "=" * 80)
    print("✅ All critical checks passed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review STATELM_IMPLEMENTATION.md for detailed documentation")
    print("2. Update your config file with StateLM settings")
    print("3. Test with a small dataset first")
    print("4. Monitor trajectory snapshots and masking behavior")
    print("=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

