import json
import logging
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.apertus2509_detector import Apertus2509Detector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
)

logger = logging.getLogger(__name__)


class Apertus2608Detector(Apertus2509Detector):
    """
    Detector for the Apertus v1.5 (2608) tool/function call format
    ```
    <|tools_prefix|>[{"tool1": {...}}, {"tool2": {...}}]<|tools_suffix|>
    ```

    The envelope is the same as the 2509 format, but the closing
    ``<|tools_suffix|>`` is treated as OPTIONAL once the JSON call list is
    complete. On Apertus v1.5, ``<|tools_suffix|>`` is an effective stop
    token (the model config exposes ``eos_token_id`` only via
    ``generation_config.json``, which lists it), so the server stops
    generation on it and trims it from the decoded text — a complete call
    list at end-of-text IS a finished tool call. Apertus 2509 pinned
    ``eos_token_id: 68`` (``<|assistant_end|>``) in ``config.json``, so the
    suffix always reached the 2509 detector and its mandatory-suffix check
    was safe there.
    """

    def __init__(self):
        super().__init__()
        self._pending_suffix: bool = False

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Extract all Apertus tools blocks and parse their JSON payloads.
        """
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text, calls=[])

        calls: List[ToolCallItem] = []
        normal_parts: List[str] = []
        cursor = 0

        while True:
            if (start := text.find(self.bot, cursor)) == -1:
                normal_parts.append(text[cursor:])
                break

            normal_parts.append(text[cursor:start])
            tool_part = text[start:]
            parsed_arr, json_end = self._try_parse_json_array(tool_part)
            if parsed_arr is None:
                normal_parts.append(tool_part)
                break

            if (suffix_pos := tool_part.find(self.suffix, json_end)) != -1:
                consumed = suffix_pos + len(self.suffix)
            elif tool_part[json_end:].strip() == "":
                # Implicit suffix: generation stopped on the <|tools_suffix|>
                # EOS token and the server trimmed it from the decoded text.
                consumed = len(tool_part)
            else:
                normal_parts.append(tool_part)
                break

            calls.extend(
                self._parse_apertus_call_list(
                    parsed_arr, tools, tool_index_offset=len(calls)
                )
            )

            cursor = start + consumed

        return StreamingParseResult(
            normal_text="".join(normal_parts).strip(), calls=calls
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for Apertus tool calls.

        - Streams any normal text before `<|tools_prefix|>[` immediately.
        - Buffers tool calls until the JSON call list is complete, then emits:
          - Tool name (empty args), then
          - Full JSON arguments string
        - Does NOT wait for `<|tools_suffix|>`: it is usually trimmed with the
          stop and never reaches the decoded text. If it does arrive (servers
          that keep stop tokens), it is swallowed.
        """
        self._buffer += new_text
        out_normal = ""
        out_calls: List[ToolCallItem] = []

        if self._pending_suffix and self._drain_pending_suffix():
            return StreamingParseResult(normal_text="", calls=[])

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        while True:
            if not self._in_tools_block:
                if (pos := self._buffer.find(self.bot)) > 0:
                    out_normal += self._buffer[:pos]
                    self._buffer = self._buffer[pos:]
                elif pos == -1:
                    if partial_bot := self._ends_with_partial_token(
                        self._buffer, self.bot
                    ):
                        out_normal += self._buffer[:-partial_bot]
                        self._buffer = self._buffer[-partial_bot:]
                    else:
                        out_normal += self._buffer
                        self._buffer = ""
                    return StreamingParseResult(normal_text=out_normal, calls=out_calls)

                self._in_tools_block = True

            if not self._buffer.startswith(self.bot):
                if (marker_pos := self._buffer.find(self.bot)) == -1:
                    out_normal += self._buffer
                    self._buffer = ""
                    self._in_tools_block = False
                    return StreamingParseResult(normal_text=out_normal, calls=out_calls)
                out_normal += self._buffer[:marker_pos]
                self._buffer = self._buffer[marker_pos:]
                continue

            parsed_arr, json_end = self._try_parse_json_array(self._buffer)
            if parsed_arr is None:
                if self.suffix in self._buffer:
                    out_normal += self._buffer
                    self._buffer = ""
                    self._in_tools_block = False
                    return StreamingParseResult(normal_text=out_normal, calls=out_calls)
                return StreamingParseResult(normal_text=out_normal, calls=out_calls)

            if self.current_tool_id == -1:
                self.current_tool_id = 0

            for item in parsed_arr:
                name, args = self._apertus_obj_to_call(item)
                if name is None:
                    continue
                if args is None:
                    args = {}

                if (
                    name not in self._tool_indices
                    and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
                ):
                    logger.warning(
                        f"Model attempted to call undefined function: {name}"
                    )
                    continue

                tool_id = self.current_tool_id
                self.current_tool_id += 1

                args_json = json.dumps(args, ensure_ascii=False)

                while len(self.prev_tool_call_arr) <= tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= tool_id:
                    self.streamed_args_for_tool.append("")

                self.prev_tool_call_arr[tool_id] = {"name": name, "arguments": args}
                self.streamed_args_for_tool[tool_id] = args_json

                # Emit tool name first, then full args (OpenAI streaming semantics)
                out_calls.append(
                    ToolCallItem(tool_index=tool_id, name=name, parameters="")
                )
                out_calls.append(
                    ToolCallItem(tool_index=tool_id, name=None, parameters=args_json)
                )

            # Consume the parsed call list and reset state. The explicit
            # suffix, if the server delivers one, is drained here or on a
            # later increment.
            self._buffer = self._buffer[json_end:]
            self._in_tools_block = False
            self._pending_suffix = True
            if self._drain_pending_suffix():
                return StreamingParseResult(normal_text=out_normal, calls=out_calls)

            if out_calls:
                # Flush normal text after the tools block, but keep a tool marker or its partial prefix in the buffer for the next stream
                if (marker_pos := self._buffer.find(self.bot)) > 0:
                    out_normal += self._buffer[:marker_pos]
                    self._buffer = self._buffer[marker_pos:]
                elif marker_pos == -1:
                    if partial_bot := self._ends_with_partial_token(
                        self._buffer, self.bot
                    ):
                        out_normal += self._buffer[:-partial_bot]
                        self._buffer = self._buffer[-partial_bot:]
                    else:
                        out_normal += self._buffer
                        self._buffer = ""
                return StreamingParseResult(normal_text=out_normal, calls=out_calls)

            continue

    def _drain_pending_suffix(self) -> bool:
        """Swallow an explicit `<|tools_suffix|>` left over after an emitted
        call list. It only reaches the decoded text when the server keeps stop
        tokens in the output (e.g. no_stop_trim), so it may arrive any number
        of increments after the calls were emitted, or never.

        Returns True when the buffer must be held back because more text is
        needed to decide (so far only whitespace or a partial suffix)."""
        stripped = self._buffer.lstrip()
        if stripped.startswith(self.suffix):
            pos = self._buffer.find(self.suffix)
            self._buffer = self._buffer[pos + len(self.suffix) :]
            self._pending_suffix = False
            return False
        if self.suffix.startswith(stripped):
            return True
        self._pending_suffix = False
        return False
