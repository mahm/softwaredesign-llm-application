# -*- coding: utf-8 -*-
"""
Tool Specification Generator for File Exploration Agent

This module provides utilities to dynamically generate tool specifications
for DSPy ReAct agents using dspy.Tool properties.
"""

import dspy


def generate_tool_specifications(tools: list) -> str:
    """
    Generate tool specifications from a list of functions.

    This function wraps each tool function with dspy.Tool and extracts
    its name, description, and argument schema to create formatted
    specifications compatible with DSPy ReAct agents.

    Args:
        tools: List of callable functions to generate specifications for.
               Each function should have proper docstrings and type hints.

    Returns:
        Formatted tool specifications string in DSPy ReAct format:
        "(1) tool_name, whose description is <desc>...</desc>. It takes arguments {...}."

    Examples:
        >>> def my_tool(arg1: str, arg2: int = 5) -> str:
        ...     \"\"\"My tool description.\"\"\"
        ...     return f"Result: {arg1}, {arg2}"
        >>> specs = generate_tool_specifications([my_tool])
        >>> print(specs)
        (1) my_tool, whose description is <desc>My tool description.</desc>.
        It takes arguments {'arg1': {'type': 'string'}, 'arg2': {'type': 'integer', 'default': 5}}.
    """
    tool_objects = [dspy.Tool(func) for func in tools]

    specs = []
    for i, tool in enumerate(tool_objects, 1):
        spec = f"({i}) {tool.name}, whose description is <desc>{tool.desc}</desc>. "
        spec += f"It takes arguments {tool.args}."
        specs.append(spec)

    return "\n".join(specs)
