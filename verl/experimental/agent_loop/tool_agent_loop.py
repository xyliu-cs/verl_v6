# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    MultiTrajectoryAgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
        document_content: Optional[str] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}
        self.document_content = document_content

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # ===========================
        # StateLM Customizations
        # ===========================
        # document_content is already set above from parameter
        
        # NEW: Editable state tracking
        self.full_history: list[dict] = copy.deepcopy(self.messages)
        self.deleted_msg_ids: set[int] = set()  # Track deleted message IDs
        self.msg_id_counter: int = 0  # Assign unique IDs to each message
        
        # NEW: View snapshots for causal log-prob
        self.emission_views: list[list[int]] = []  # prompt_ids when each assistant turn started
        self.assistant_turn_boundaries: list[tuple[int, int]] = []  # (start_idx, end_idx) in response_ids
        
        # Track snapshots for deleteContext operations
        self.trajectory_snapshots: list[dict[str, Any]] = []  # Store trajectory state before each delete
        
        # Track whether we had a deleteContext operation
        self.had_delete_operation: bool = False

        self.notes: dict[str, dict] = {}
        
        # StateLM State Manager for document operations
        self.doc_state_manager = None

    def _render_message_view(self) -> list[dict[str, Any]]:
        """
        Build a rendered view of the message history by taking into account editions invoked by StateLM.
        """
        stub_message = "Content has been deleted to save space."
        rendered_message_view: list[dict[str, Any]] = []

        if self.notes:
            notes_summary = (
                f"\n\n<external_memory>\n## Available Notes\n"
                f"{"\n".join([f"- **{key}**: {data['summary']}" for key, data in self.notes.items()])}\n</external_memory>"
            )
        else:
            notes_summary = (
                f"\n\n<external_memory>\n## Available Notes\n"
                "No notes recorded.\n</external_memory>"
            )

        for idx, msg in enumerate(self.full_history):
            role = msg.get("role")

            if role == "user":
                text = msg["content"] + (notes_summary if idx == 0 else "")
                rendered_message_view.append({"role": "user", "content": text})

            elif role == "assistant":
                msg_id = msg["msg_id"]
                tool_calls = msg.get("tool_calls", [])

                if msg_id in self.deleted_msg_ids:
                    tool_calls = msg.get("tool_calls") or []
                    if tool_calls:
                        stub_tool_calls = []
                        for tc in tool_calls:
                            fn = tc.get("function") or {}
                            name = fn.get("name") or ""

                            stub_tool_calls.append({
                                "id": tc.get("id"),
                                "type": "function",
                                "function": {
                                    "name": name,
                                    # arguments MUST be a JSON string
                                    "arguments": json.dumps(
                                        {"message": stub_message},
                                    ),
                                },
                            })
                        rendered_message_view.append({
                            "role": "assistant",
                            "content": stub_message,
                            "tool_calls": stub_tool_calls,
                        })
                    else:
                        rendered_message_view.append({
                            "role": "assistant",
                            "content": stub_message,
                        })
                # Undeleted messages
                else:
                    assistant_content = msg["content"]
                    assert len(assistant_content) == 1, "Expected single content block in assistant message."
                    raw_text = assistant_content[0]["text"]
                    cleaned_text = raw_text.strip()
                    assistant_msg = {
                        "role": "assistant",
                        "content": (cleaned_text if cleaned_text else ""),
                    }
                    # If this assistant turn contained tool calls, forward them verbatim
                    if tool_calls:
                        assistant_msg["tool_calls"] = msg["tool_calls"]
                    rendered_message_view.append(assistant_msg)

            elif role == "tool":
                msg_id = msg["msg_id"]
                msg_id_ia = msg["msg_id(invoking_assistant)"]
                tool_result_content_cp = copy.deepcopy(msg["content"])
                tool_result_content_cp["msg_id"] = msg_id
                tool_result_content_cp["msg_id(invoking_assistant)"] = msg_id_ia
                if msg_id in self.deleted_msg_ids:
                    tool_name = msg.get("tool_name", "unknown")
                    tool_result_content_cp = {
                        "msg_id": msg_id,
                        "msg_id(invoking_assistant)": msg_id_ia,
                        "status": "success",
                        "message": stub_message,
                        "original_tool": tool_name
                    }
                    # if tool_name == "nextChunk":
                    #     tool_result_content_cp["reading_progress"] = msg["content"]["reading_progress"]
                
                rendered_message_view.append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_result_content_cp, ensure_ascii=False),
                    }
                )
        return rendered_message_view



@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)
        
        cls.statelm_enabled = config.actor_rollout_ref.rollout.multi_turn.get("statelm_enabled", False)
        if cls.statelm_enabled:
            print("StateLM features enabled in ToolAgentLoop.")
        
        # Max model context length for protection
        cls.max_model_length = config.actor_rollout_ref.rollout.get("max_model_length", 8192)
        cls.context_length_penalty = config.actor_rollout_ref.rollout.multi_turn.get("context_length_penalty", -1.0)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        # NEW: Extract document content from kwargs (not in messages)
        document_content = kwargs.get("document_content", "")
        
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            document_content=document_content,  # NEW
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # NEW: Initialize full_history with the first user message
        agent_data.msg_id_counter = 0
        agent_data.full_history.append({
            "role": "user",
            "content": messages[0]["content"],  # Just the question
            "msg_id": agent_data.msg_id_counter
        })
        agent_data.msg_id_counter += 1
        
        # Initialize StateLM state manager if document_content is provided
        if self.statelm_enabled and document_content:
            from verl.tools.statelm_tools import DocStateManager
            agent_data.doc_state_manager = DocStateManager(self.tokenizer, document_content)

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output - handle multiple trajectories for StateLM
        trajectories = []
        
        # First, add all snapshot trajectories (for StateLM deleteContext)
        if self.statelm_enabled and agent_data.trajectory_snapshots:
            for idx, snapshot in enumerate(agent_data.trajectory_snapshots):
                # Extract data from snapshot
                snapshot_prompt_ids = snapshot["prompt_ids"]
                snapshot_response_mask = snapshot["response_mask"]
                snapshot_response_logprobs = snapshot.get("response_logprobs", None)
                
                # Compute response_ids and prompt_ids from snapshot
                snapshot_response_ids = snapshot_prompt_ids[-len(snapshot_response_mask):]
                snapshot_prompt_only_ids = snapshot_prompt_ids[:len(snapshot_prompt_ids) - len(snapshot_response_mask)]
                
                # Create output for this snapshot trajectory
                snapshot_output = AgentLoopOutput(
                    prompt_ids=snapshot_prompt_only_ids,
                    response_ids=snapshot_response_ids[: self.response_length],
                    response_mask=snapshot_response_mask[: self.response_length],
                    multi_modal_data={"image": agent_data.image_data} if agent_data.image_data is not None else {},
                    response_logprobs=snapshot_response_logprobs[: self.response_length]
                    if snapshot_response_logprobs
                    else None,
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,  # Approximate
                    metrics=agent_data.metrics,
                    extra_fields={
                        "turn_scores": agent_data.turn_scores[:idx+1] if agent_data.turn_scores else [],
                        "tool_rewards": agent_data.tool_rewards[:idx+1] if agent_data.tool_rewards else [],
                        "is_snapshot": True,
                        "snapshot_index": idx,
                    },
                )
                trajectories.append(snapshot_output)
        
        # Add the final trajectory
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        final_output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "is_snapshot": False,
            },
        )
        trajectories.append(final_output)
        
        # Cleanup state manager if needed
        if self.statelm_enabled and agent_data.doc_state_manager:
            try:
                agent_data.doc_state_manager.clear_current_document()
            except Exception as e:
                logger.warning(f"Error clearing state manager: {e}")
        
        # Return MultiTrajectoryAgentLoopOutput if we have snapshots, otherwise single output
        if self.statelm_enabled and len(trajectories) > 1:
            return MultiTrajectoryAgentLoopOutput(trajectories=trajectories)
        else:
            return final_output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if self.statelm_enabled:
            messages = agent_data._render_message_view()
        else:
            messages = agent_data.messages

        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Need to update response_id as well
            if self.statelm_enabled:
                agent_data.response_ids = agent_data.prompt_ids[len(self.system_prompt) :]
        
        # Context length protection: check if trajectory exceeds max model length
        if self.statelm_enabled and len(agent_data.prompt_ids) + self.response_length > self.max_model_length:
            logger.warning(
                f"Trajectory length {len(agent_data.prompt_ids)} + response_length {self.response_length} "
                f"exceeds max_model_length {self.max_model_length}. Early stopping with penalty."
            )
            # Set negative reward for exceeding context length
            agent_data.metrics["context_length_exceeded"] = True
            agent_data.metrics["early_stop_penalty"] = self.context_length_penalty
            agent_data.turn_scores.append(self.context_length_penalty)
            return AgentState.TERMINATED
        
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        # Track emission view before generation (for StateLM)
        if self.statelm_enabled:
            # Save the prompt_ids snapshot before this assistant turn
            emission_view = copy.deepcopy(agent_data.prompt_ids)
            agent_data.emission_views.append(emission_view)
            
            # Track the start position in response_mask for this assistant turn
            turn_start_idx = len(agent_data.response_mask)

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        
        # Track assistant turn boundaries (for StateLM)
        if self.statelm_enabled:
            turn_end_idx = len(agent_data.response_mask)
            agent_data.assistant_turn_boundaries.append((turn_start_idx, turn_end_idx))

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        text_response, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        if self.statelm_enabled:
            # Append assistant message to full_history
            tool_calls_copy = copy.deepcopy(agent_data.tool_calls)
            agent_data.full_history.append({
                "role": "assistant",
                "content": [{"text": text_response}],  # Content should be a list with text block
                "tool_calls": [tc.model_dump() for tc in tool_calls_copy],
                "msg_id": agent_data.msg_id_counter,
            })
            agent_data.msg_id_counter += 1

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        finish_tool_call = False
        editor_tool_call = False
        delete_msg_ids = []  # Collect message IDs to delete
        
        for tool_response, tool_reward, tool_result_dict , tool_name in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            # Handle special tool types
            if tool_name == 'finish':
                finish_tool_call = True
            elif tool_name == 'deleteContext':
                editor_tool_call = True
                agent_data.had_delete_operation = True
                # Extract message IDs to delete from tool result
                if tool_result_dict and "deleted_msg_ids" in tool_result_dict:
                    delete_msg_ids.extend(tool_result_dict["deleted_msg_ids"])

            add_messages.append(message)
            agent_data.messages.extend(add_messages)
            
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

            # Handle image data
            if tool_response.image:
                if agent_data.image_data is None:
                    agent_data.image_data = []
                elif not isinstance(agent_data.image_data, list):
                    agent_data.image_data = [agent_data.image_data]

                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            agent_data.image_data.append(img)
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        agent_data.image_data.append(tool_response.image)
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        # Handle deleteContext operation: snapshot, mask, and delete
        if self.statelm_enabled and editor_tool_call:
            # Step 1: Create snapshot of current trajectory state
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
            
            # Step 2: Mask ALL previous messages (set all previous response_mask to 0)
            # Keep only the current tool result visible for learning
            if len(agent_data.response_mask) > 0:
                # Mask everything before the tool response we just added
                current_length = len(agent_data.response_mask)
                # We need to find where the tool response starts
                # The tool response was added after processing, so we mask everything before it
                for i in range(current_length):
                    agent_data.response_mask[i] = 0
                # Note: The tool response tokens will be added below with mask 0 anyway
            
            # Step 3: Mark messages as deleted
            for msg_id in delete_msg_ids:
                agent_data.deleted_msg_ids.add(msg_id)
            
            logger.info(f"DeleteContext operation: masked {len(agent_data.response_mask)} previous tokens, "
                       f"deleted message IDs: {delete_msg_ids}")

        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Use only the new images from this turn for processing tool responses
            current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        
        # For StateLM, we leave the state data handling to the PENDING state
        if self.statelm_enabled:
            if finish_tool_call or agent_data.assistant_turns >= self.max_assistant_turns:
                return AgentState.TERMINATED
            elif editor_tool_call: # For editing tool call, need to re-render the message view
                return AgentState.PENDING

        # For non-editing tool call, we can just append the tool response to prompt directly
        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict, str]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        tool_name = ""
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            
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
            
            # Handle regular tools
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            
            # Prepare execution kwargs for StateLM tools
            exec_kwargs = {}
            if self.statelm_enabled:
                statelm_tool_names = {
                    'analyzeText', 'loadDocument', 'buildIndex', 'readChunk',
                    'searchEngine', 'note', 'readNote', 'updateNote', 'mergeNotes',
                    'checkBudget', 'getContextStats', 'finish'
                }
                if tool_name in statelm_tool_names:
                    exec_kwargs['agent_data'] = agent_data
                    exec_kwargs['state_manager'] = agent_data.doc_state_manager
                    exec_kwargs['tokenizer'] = self.tokenizer
                    exec_kwargs['tool_schemas'] = self.tool_schemas
                    exec_kwargs['max_context_exp'] = getattr(self, 'max_model_length', 8192)
                    exec_kwargs['max_output_tokens'] = self.response_length
                    exec_kwargs['max_turns'] = self.max_assistant_turns
            
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args, **exec_kwargs)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
                tool_name,
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res if res else {}, tool_name

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
