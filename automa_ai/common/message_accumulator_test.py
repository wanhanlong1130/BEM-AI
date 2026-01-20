"""
Unit tests for AIMessageAccumulator class.

These tests cover:
- Basic functionality (text routing, metadata accumulation)
- Edge cases (split markers, multiple artifacts, empty chunks)
- Metadata merging
- Tool calls
"""

import pytest
from langchain_core.messages import AIMessageChunk
from automa_ai.common.message_accumulator import AIMessageAccumulator, ARTIFACT_START, ARTIFACT_END


class TestBasicFunctionality:
    """Test basic accumulation and routing."""

    def test_simple_text_no_artifact(self):
        """Test accumulating simple text without any artifact."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content="Hello "))
        acc.add_chunk(AIMessageChunk(content="world!"))

        msg = acc.finalize()
        assert msg.content == "Hello world!"
        assert acc.get_artifact_text() is None

    def test_simple_artifact(self):
        """Test accumulating a simple artifact."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content=f"Here is your file: {ARTIFACT_START}"))
        acc.add_chunk(AIMessageChunk(content="def hello():\n    print('Hello')"))
        acc.add_chunk(AIMessageChunk(content=f"{ARTIFACT_END}"))

        msg = acc.finalize()
        assert msg.content == "Here is your file: "
        assert acc.get_artifact_text() == "def hello():\n    print('Hello')"

    def test_text_before_and_after_artifact(self):
        """Test text both before and after an artifact."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content=f"Before {ARTIFACT_START}artifact{ARTIFACT_END} after"))

        msg = acc.finalize()
        assert msg.content == "Before  after"
        assert acc.get_artifact_text() == "artifact"

    def test_multiple_artifacts(self):
        """Test multiple artifacts (they get concatenated)."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(
            content=f"{ARTIFACT_START}first{ARTIFACT_END} middle {ARTIFACT_START}second{ARTIFACT_END}"
        ))

        msg = acc.finalize()
        assert msg.content == " middle "
        # Both artifacts are concatenated
        assert acc.get_artifact_text() == "firstsecond"

    def test_empty_artifact(self):
        """Test an empty artifact."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content=f"Text {ARTIFACT_START}{ARTIFACT_END} more"))

        msg = acc.finalize()
        assert msg.content == "Text  more"
        # Empty string strips to None
        assert acc.get_artifact_text() is None

    def test_whitespace_only_artifact(self):
        """Test an artifact with only whitespace."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content=f"{ARTIFACT_START}   \n  {ARTIFACT_END}"))

        msg = acc.finalize()
        assert msg.content == ""
        # Whitespace is stripped
        assert acc.get_artifact_text() is None


class TestSplitMarkers:
    """Test handling of markers split across chunks."""

    def test_start_marker_split_simple(self):
        """Test artifact start marker split across two chunks."""
        acc = AIMessageAccumulator()

        # Split the marker in the middle
        marker_split = len(ARTIFACT_START) // 2
        acc.add_chunk(AIMessageChunk(content=f"Text {ARTIFACT_START[:marker_split]}"))
        acc.add_chunk(AIMessageChunk(content=f"{ARTIFACT_START[marker_split:]}content{ARTIFACT_END}"))

        msg = acc.finalize()
        assert msg.content == "Text "
        assert acc.get_artifact_text() == "content"

    def test_end_marker_split_simple(self):
        """Test artifact end marker split across two chunks."""
        acc = AIMessageAccumulator()

        marker_split = len(ARTIFACT_END) // 2
        acc.add_chunk(AIMessageChunk(content=f"{ARTIFACT_START}content{ARTIFACT_END[:marker_split]}"))
        acc.add_chunk(AIMessageChunk(content=f"{ARTIFACT_END[marker_split:]} after"))

        msg = acc.finalize()
        assert msg.content == " after"
        assert acc.get_artifact_text() == "content"

    def test_start_marker_split_one_char(self):
        """Test start marker split with only one character in first chunk."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content="Text <"))
        acc.add_chunk(AIMessageChunk(content="<<ARTIFACT_OUTPUT>>>content<<<END_ARTIFACT_OUTPUT>>>"))

        msg = acc.finalize()
        assert msg.content == "Text "
        assert acc.get_artifact_text() == "content"

    def test_end_marker_split_one_char(self):
        """Test end marker split with only one character in first chunk."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content="<<<ARTIFACT_OUTPUT>>>content<"))
        acc.add_chunk(AIMessageChunk(content="<<END_ARTIFACT_OUTPUT>>> after"))

        msg = acc.finalize()
        assert msg.content == " after"
        assert acc.get_artifact_text() == "content"

    def test_false_partial_marker(self):
        """Test that similar but different text doesn't trigger partial marker logic."""
        acc = AIMessageAccumulator()

        # "<<<" looks like start of marker but isn't actually the marker
        acc.add_chunk(AIMessageChunk(content="Use <<< for comparison"))
        acc.add_chunk(AIMessageChunk(content=" operators"))

        msg = acc.finalize()
        assert msg.content == "Use <<< for comparison operators"
        assert acc.get_artifact_text() is None

    def test_marker_split_three_ways(self):
        """Test marker split across three chunks."""
        acc = AIMessageAccumulator()

        # Split into three parts
        acc.add_chunk(AIMessageChunk(content="<<<"))
        acc.add_chunk(AIMessageChunk(content="ARTIFACT_"))
        acc.add_chunk(AIMessageChunk(content="OUTPUT>>>content<<<END_ARTIFACT_OUTPUT>>>"))

        msg = acc.finalize()
        assert msg.content == ""
        assert acc.get_artifact_text() == "content"


class TestMetadata:
    """Test metadata accumulation."""

    def test_additional_kwargs_simple(self):
        """Test accumulating additional_kwargs."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content="test", additional_kwargs={"model": "gpt-4"}))
        acc.add_chunk(AIMessageChunk(content="", additional_kwargs={"temperature": 0.7}))

        msg = acc.finalize()
        assert msg.additional_kwargs == {"model": "gpt-4", "temperature": 0.7}

    def test_additional_kwargs_nested_merge(self):
        """Test merging nested dictionaries in additional_kwargs."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(
            content="",
            additional_kwargs={"usage": {"prompt_tokens": 10}}
        ))
        acc.add_chunk(AIMessageChunk(
            content="",
            additional_kwargs={"usage": {"completion_tokens": 20}}
        ))

        msg = acc.finalize()
        assert msg.additional_kwargs == {
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }

    def test_additional_kwargs_overwrite(self):
        """Test that later values overwrite earlier ones."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content="", additional_kwargs={"model": "gpt-3.5"}))
        acc.add_chunk(AIMessageChunk(content="", additional_kwargs={"model": "gpt-4"}))

        msg = acc.finalize()
        assert msg.additional_kwargs["model"] == "gpt-4"

    def test_response_metadata(self):
        """Test accumulating response_metadata."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content="", response_metadata={"stop": "end"}))
        acc.add_chunk(AIMessageChunk(content="", response_metadata={"finish_reason": "stop"}))

        msg = acc.finalize()
        assert msg.response_metadata == {"stop": "end", "finish_reason": "stop"}

    def test_tool_calls(self):
        """Test accumulating tool calls."""
        acc = AIMessageAccumulator()

        tool_call_1 = {"name": "search", "args": {"query": "test"}}
        tool_call_2 = {"name": "calculator", "args": {"expression": "2+2"}}

        acc.add_chunk(AIMessageChunk(content="", tool_calls=[tool_call_1]))
        acc.add_chunk(AIMessageChunk(content="", tool_calls=[tool_call_2]))

        msg = acc.finalize()
        assert msg.tool_calls == [tool_call_1, tool_call_2]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_chunks(self):
        """Test handling empty chunks."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content=""))
        acc.add_chunk(AIMessageChunk(content="Hello"))
        acc.add_chunk(AIMessageChunk(content=""))
        acc.add_chunk(AIMessageChunk(content=" world"))
        acc.add_chunk(AIMessageChunk(content=""))

        msg = acc.finalize()
        assert msg.content == "Hello world"

    def test_none_content(self):
        """Test handling chunks with None content."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content=None))
        acc.add_chunk(AIMessageChunk(content="Hello"))
        acc.add_chunk(AIMessageChunk(content=None))

        msg = acc.finalize()
        assert msg.content == "Hello"

    def test_unclosed_artifact(self):
        """Test handling an artifact that's never closed."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(content=f"Text {ARTIFACT_START}artifact content"))

        msg = acc.finalize()
        assert msg.content == "Text "
        assert acc.get_artifact_text() == "artifact content"

    def test_unopened_artifact_end(self):
        """Test handling an end marker without a start marker."""
        acc = AIMessageAccumulator()

        # End marker without start - should be treated as normal text
        acc.add_chunk(AIMessageChunk(content=f"Text {ARTIFACT_END} more"))

        msg = acc.finalize()
        assert msg.content == f"Text {ARTIFACT_END} more"
        assert acc.get_artifact_text() is None

    def test_nested_start_markers(self):
        """Test handling nested start markers (second start is treated as content)."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(
            content=f"{ARTIFACT_START}content {ARTIFACT_START} more{ARTIFACT_END}"
        ))

        msg = acc.finalize()
        assert msg.content == ""
        # The nested start marker is part of the artifact content
        assert acc.get_artifact_text() == f"content {ARTIFACT_START} more"

    def test_multiple_consecutive_chunks_same_content(self):
        """Test adding many chunks with the same content."""
        acc = AIMessageAccumulator()

        for _ in range(100):
            acc.add_chunk(AIMessageChunk(content="a"))

        msg = acc.finalize()
        assert msg.content == "a" * 100

    def test_large_artifact(self):
        """Test handling a large artifact."""
        acc = AIMessageAccumulator()

        large_content = "x" * 10000
        acc.add_chunk(AIMessageChunk(content=f"{ARTIFACT_START}{large_content}{ARTIFACT_END}"))

        msg = acc.finalize()
        assert msg.content == ""
        assert acc.get_artifact_text() == large_content

    def test_marker_as_part_of_content(self):
        """Test that markers inside artifact are treated as content."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(
            content=f"{ARTIFACT_START}This contains {ARTIFACT_START} in the middle{ARTIFACT_END}"
        ))

        msg = acc.finalize()
        assert acc.get_artifact_text() == f"This contains {ARTIFACT_START} in the middle"


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_streaming_code_with_explanation(self):
        """Test a realistic scenario of streaming code with explanation."""
        acc = AIMessageAccumulator()

        chunks = [
            "Here's a Python function to calculate fibonacci numbers:\n\n",
            ARTIFACT_START,
            "def fibonacci(n):\n",
            "    if n <= 1:\n",
            "        return n\n",
            "    return fibonacci(n-1) + fibonacci(n-2)",
            ARTIFACT_END,
            "\n\nThis uses recursion to calculate the nth fibonacci number."
        ]

        for chunk_content in chunks:
            acc.add_chunk(AIMessageChunk(content=chunk_content))

        msg = acc.finalize()
        assert "Here's a Python function" in msg.content
        assert "This uses recursion" in msg.content
        assert ARTIFACT_START not in msg.content
        assert ARTIFACT_END not in msg.content

        artifact = acc.get_artifact_text()
        assert artifact is not None
        assert "def fibonacci(n):" in artifact
        assert "return fibonacci(n-1) + fibonacci(n-2)" in artifact

    def test_metadata_with_artifact(self):
        """Test that metadata is preserved alongside artifact content."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(
            content=f"{ARTIFACT_START}code",
            additional_kwargs={"model": "claude"}
        ))
        acc.add_chunk(AIMessageChunk(
            content=f"{ARTIFACT_END}",
            response_metadata={"stop_reason": "end_turn"}
        ))

        msg = acc.finalize()
        assert msg.additional_kwargs["model"] == "claude"
        assert msg.response_metadata["stop_reason"] == "end_turn"
        assert acc.get_artifact_text() == "code"

    def test_interleaved_text_and_artifacts(self):
        """Test text and artifacts interleaved."""
        acc = AIMessageAccumulator()

        acc.add_chunk(AIMessageChunk(
            content=f"First text {ARTIFACT_START}artifact1{ARTIFACT_END} middle {ARTIFACT_START}artifact2{ARTIFACT_END} last"
        ))

        msg = acc.finalize()
        assert msg.content == "First text  middle  last"
        assert acc.get_artifact_text() == "artifact1artifact2"

    def test_real_streaming_pattern(self):
        """Test a realistic streaming pattern with small chunks."""
        acc = AIMessageAccumulator()

        # Simulate realistic small chunks
        full_text = f"I'll create that for you: {ARTIFACT_START}const x = 42;{ARTIFACT_END} There you go!"
        chunk_size = 5

        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size]
            acc.add_chunk(AIMessageChunk(content=chunk))

        msg = acc.finalize()
        assert msg.content == "I'll create that for you:  There you go!"
        assert acc.get_artifact_text() == "const x = 42;"

    def test_add_chunk_failed_case(self):
        """Test a realistic streaming pattern with small chunks."""
        acc = AIMessageAccumulator()

        # Simulate realistic small chunks
        full_text = f"Of course, I can help with that. To get started, please provide me with a description of your project and let"
        chunk = AIMessageChunk(content=full_text)
        print(chunk)
        acc.add_chunk(chunk)

        msg = acc.get_last_assistant_text()
        assert msg == "Of course, I can help with that. To get started, please provide me with a description of your project and let"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])