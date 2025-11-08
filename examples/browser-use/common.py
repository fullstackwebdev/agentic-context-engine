#!/usr/bin/env python3
"""
Common utilities for browser-use examples.

Generic utilities shared across all browser-use demos.
Example-specific utilities should live in their respective example folders.
"""

from typing import Dict, Any, Optional
import json
from pathlib import Path


def calculate_timeout_steps(timeout_seconds: float) -> int:
    """
    Calculate additional steps for timeout based on 1 step per 12 seconds.

    Args:
        timeout_seconds: The timeout in seconds

    Returns:
        Number of additional steps to allow
    """
    return int(timeout_seconds // 12)


def format_result_output(
    task_name: str,
    success: bool,
    steps: int,
    error: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format a consistent output message for task results.

    Args:
        task_name: Name of the task completed
        success: Whether the task succeeded
        steps: Number of steps taken
        error: Error message if failed
        additional_info: Any additional information to include

    Returns:
        Formatted output string
    """
    status = "✅ SUCCESS" if success else "❌ FAILED"
    output = f"\n{status}: {task_name}\n"
    output += f"Steps taken: {steps}\n"

    if error:
        output += f"Error: {error}\n"

    if additional_info:
        for key, value in additional_info.items():
            output += f"{key}: {value}\n"

    return output


def save_results_to_file(
    results: Dict[str, Any],
    filename: str,
    directory: str = "results"
) -> Path:
    """
    Save task results to a JSON file.

    Args:
        results: Dictionary of results to save
        filename: Name of the file to save
        directory: Directory to save in (created if doesn't exist)

    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    results_dir = Path(directory)
    results_dir.mkdir(exist_ok=True)

    # Save results
    filepath = results_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return filepath


def get_browser_config(headless: bool = True) -> Dict[str, Any]:
    """
    Get common browser configuration settings.

    Args:
        headless: Whether to run browser in headless mode

    Returns:
        Dictionary of browser configuration options
    """
    return {
        "headless": headless,
        "viewport": {"width": 1920, "height": 1080},
        "timeout": 30000,  # 30 seconds default timeout
        "wait_for_network_idle": True,
    }


def get_llm_config(model: str = "gpt-4o", temperature: float = 0.0) -> Dict[str, Any]:
    """
    Get common LLM configuration settings.

    Args:
        model: The model to use
        temperature: Temperature setting for the model

    Returns:
        Dictionary of LLM configuration options
    """
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }


# Constants for retry logic
MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 180.0
STEPS_PER_SECOND = 1 / 12  # 1 step per 12 seconds