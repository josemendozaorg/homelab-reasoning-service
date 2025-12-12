"""Tests for the reasoning graph components."""
import pytest

from src.reasoning.state import ReasoningState, create_initial_state
from src.reasoning.nodes import parse_reasoning_response, should_continue


def test_create_initial_state():
    """Initial state is created correctly."""
    state = create_initial_state("What is 2+2?")

    assert state["query"] == "What is 2+2?"
    assert state["reasoning_trace"] == []
    assert state["current_answer"] is None
    assert state["critique"] is None
    assert state["iteration"] == 0
    assert state["is_complete"] is False
    assert state["final_answer"] is None


def test_parse_reasoning_response_with_think_tags():
    """Parse response extracts think content and answer."""
    response = """<think>
First, I need to add 2 and 2.
2 + 2 = 4
</think>

The answer is 4."""

    reasoning, answer = parse_reasoning_response(response)

    assert "2 + 2 = 4" in reasoning
    assert answer == "The answer is 4."


def test_parse_reasoning_response_without_think_tags():
    """Parse response handles missing think tags."""
    response = "The answer is simply 4."

    reasoning, answer = parse_reasoning_response(response)

    assert reasoning == ""
    assert answer == "The answer is simply 4."


def test_should_continue_returns_end_when_complete():
    """Router returns 'end' when reasoning is complete."""
    state: ReasoningState = {
        "query": "test",
        "reasoning_trace": [],
        "current_answer": "answer",
        "critique": "APPROVED",
        "iteration": 1,
        "is_complete": True,
        "final_answer": "answer"
    }

    result = should_continue(state)
    assert result == "end"


def test_should_continue_returns_reason_when_incomplete():
    """Router returns 'reason' when more iterations needed."""
    state: ReasoningState = {
        "query": "test",
        "reasoning_trace": [],
        "current_answer": "answer",
        "critique": "needs improvement",
        "iteration": 1,
        "is_complete": False,
        "final_answer": None
    }

    result = should_continue(state)
    assert result == "reason"
