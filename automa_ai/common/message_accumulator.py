"""
AI Message Accumulator for handling streaming chunks from LangChain.

This module provides a class to accumulate AI message chunks, properly routing
content between normal assistant text and artifact outputs based on special markers.
"""

from typing import Any
from langchain_core.messages import AIMessage, AIMessageChunk

ARTIFACT_START = "<<<ARTIFACT_OUTPUT>>>"
ARTIFACT_END = "<<<END_ARTIFACT_OUTPUT>>>"


class AIMessageAccumulator:
    """
    Accumulates AI message chunks and separates assistant text from artifact content.

    Artifacts are enclosed in special markers:
    - Start: <<<ARTIFACT_OUTPUT>>>
    - End: <<<END_ARTIFACT_OUTPUT>>>

    The accumulator handles edge cases like:
    - Markers split across multiple chunks
    - Multiple artifacts in the same message
    - Nested markers (treated as text, not as actual nesting)
    """

    def __init__(self):
        # Text buffers
        self._thread_id: str | None = None
        self._assistant_parts: list[str] = []
        self._artifact_parts: list[str] = []

        # Metadata
        self._additional_kwargs: dict[str, Any] = {}
        self._response_metadata: dict[str, Any] = {}
        self._tool_calls: list[dict] = []

        # State tracking
        self._in_artifact = False
        self._carry = ""  # Handles marker split across chunks

    def add_chunk(self, chunk: AIMessageChunk) -> None:
        """
        Add a chunk to the accumulator.

        Args:
            chunk: An AIMessageChunk from LangChain streaming
        """
        # ---- Accumulate metadata ----
        if chunk.additional_kwargs:
            self._merge_dict(self._additional_kwargs, chunk.additional_kwargs)

        if chunk.response_metadata:
            self._response_metadata.update(chunk.response_metadata)

        if getattr(chunk, "tool_calls", None):
            self._tool_calls.extend(chunk.tool_calls)

        # ---- Route content ----
        if not chunk.content:
            return

        content = chunk.content
        if isinstance(content, list):
            if not content:
                return
            content = content[0]
            if isinstance(content, dict):
                content = content.get("text") or ""
            if not isinstance(content, str):
                return

        if not isinstance(content, str):
            return

        text = self._carry + content
        self._carry = ""

        while text:
            if not self._in_artifact:
                # Looking for artifact start marker
                start_idx = text.find(ARTIFACT_START)
                if start_idx == -1:
                    # No marker found - check if we have a partial marker at the end
                    partial_len = self._get_partial_marker_length(text, ARTIFACT_START)
                    if partial_len > 0:
                        # Save the partial marker for the next chunk
                        self._assistant_parts.append(text[:-partial_len])
                        self._carry = text[-partial_len:]
                    else:
                        # No partial marker, add all text
                        self._assistant_parts.append(text)
                    break
                else:
                    # Found start marker
                    self._assistant_parts.append(text[:start_idx])
                    text = text[start_idx + len(ARTIFACT_START):]
                    self._in_artifact = True
            else:
                # Looking for artifact end marker
                end_idx = text.find(ARTIFACT_END)
                if end_idx == -1:
                    # No marker found - check if we have a partial marker at the end
                    partial_len = self._get_partial_marker_length(text, ARTIFACT_END)
                    if partial_len > 0:
                        # Save the partial marker for the next chunk
                        self._artifact_parts.append(text[:-partial_len])
                        self._carry = text[-partial_len:]
                    else:
                        # No partial marker, add all text
                        self._artifact_parts.append(text)
                    break
                else:
                    # Found end marker
                    self._artifact_parts.append(text[:end_idx])
                    text = text[end_idx + len(ARTIFACT_END):]
                    self._in_artifact = False

    def finalize(self) -> AIMessage:
        """
        Finalize the accumulated chunks into a complete AIMessage, and reset the state

        Returns:
            AIMessage with all accumulated content and metadata
        """
        # Handle any remaining carry (shouldn't happen in normal use)
        if self._carry:
            if self._in_artifact:
                self._artifact_parts.append(self._carry)
            else:
                self._assistant_parts.append(self._carry)
            self._carry = ""

        combine_parts = "".join(self._assistant_parts) + "".join(self._artifact_parts)

        # Build the final message
        message = AIMessage(
            content=combine_parts,
            additional_kwargs=self._additional_kwargs if self._additional_kwargs else {},
            response_metadata=self._response_metadata if self._response_metadata else {},
            tool_calls=self._tool_calls if self._tool_calls else [],
        )

        # Reset state for potential reuse
        self._assistant_parts.clear()
        self._artifact_parts.clear()
        self._additional_kwargs.clear()
        self._response_metadata.clear()
        self._tool_calls.clear()
        self._in_artifact = False
        self._carry = ""
        return message

    def get_assistant_text(self) -> str | None:
        """
        Get the accumulated assistant content.

        Returns:
            The assistant text (stripped of whitespace), or None if no assistant
        """
        if not self._assistant_parts:
            return None
        return "".join(self._assistant_parts).strip()

    def get_artifact_text(self) -> str | None:
        """
        Get the accumulated artifact content.

        Returns:
            The artifact text (stripped of whitespace), or None if no artifact
        """
        if not self._artifact_parts:
            return None
        artifact_text = "".join(self._artifact_parts).strip()
        return artifact_text if artifact_text else None

    def _merge_dict(self, target: dict, source: dict) -> None:
        """
        Merge dictionaries recursively, with last-write-wins for scalars.

        Args:
            target: Dictionary to merge into
            source: Dictionary to merge from
        """
        for key, value in source.items():
            if (
                    key in target
                    and isinstance(target[key], dict)
                    and isinstance(value, dict)
            ):
                self._merge_dict(target[key], value)
            else:
                target[key] = value

    def _get_partial_marker_length(self, text: str, marker: str) -> int:
        """
        Check if text ends with a partial marker.

        This handles the edge case where a marker is split across chunks.
        For example, if the marker is "<<<ARTIFACT>>>" and the text ends with
        "<<<AR", we return 5 so those characters can be carried over.

        Args:
            text: The text to check
            marker: The marker to look for partial matches of

        Returns:
            Length of the partial marker at the end of text, or 0 if none
        """
        for i in range(1, len(marker)):
            if text.endswith(marker[:i]):
                return i
        return 0
