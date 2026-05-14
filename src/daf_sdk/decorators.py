"""
DAF SDK - Custom Tool Decorator
===============================

Provides @custom_tool decorator for creating custom tools from Python functions.

Usage:
    from daf_sdk import custom_tool

    @custom_tool
    def my_tool(arg1: str, arg2: int) -> str:
        '''Tool description from docstring.'''
        return f"Result: {arg1}, {arg2}"

    # Register with backend
    client.tools.register(my_tool)
"""

import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union


class ToolFunction:
    """
    Wrapper for functions decorated with @custom_tool.

    Extracts metadata from function signature and docstring.
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        requires_approval: bool = False,
    ):
        self._func = func
        self._name = name or func.__name__
        self._description = description or self._extract_description(func)
        self._requires_approval = requires_approval
        self._parameters = self._extract_parameters(func)
        self._code = self._extract_code(func)

        # Preserve function metadata
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        """Allow the decorated function to still be called normally."""
        return self._func(*args, **kwargs)

    @property
    def name(self) -> str:
        """Tool name."""
        return self._name

    @property
    def description(self) -> str:
        """Tool description."""
        return self._description

    @property
    def requires_approval(self) -> bool:
        """Whether tool requires human approval."""
        return self._requires_approval

    @property
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters schema (JSON Schema format)."""
        return self._parameters

    @property
    def code(self) -> str:
        """Source code of the function."""
        return self._code

    def _extract_description(self, func: Callable) -> str:
        """Extract description from docstring."""
        doc = func.__doc__ or ""
        # Get first line or paragraph
        lines = doc.strip().split("\n")
        if lines:
            # Get first non-empty line
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("Args:"):
                    return stripped
        return f"Custom tool: {func.__name__}"

    def _extract_parameters(self, func: Callable) -> Dict[str, Any]:
        """Extract parameters from function signature and docstring."""
        sig = inspect.signature(func)
        type_hints = {}
        try:
            type_hints = func.__annotations__
        except AttributeError:
            pass

        properties = {}
        required = []

        # Parse docstring for parameter descriptions
        param_docs = self._parse_docstring_params(func.__doc__ or "")

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Get type from annotations
            param_type = type_hints.get(param_name, Any)
            json_type = self._python_type_to_json(param_type)

            prop = {"type": json_type}

            # Add description from docstring
            if param_name in param_docs:
                prop["description"] = param_docs[param_name]

            properties[param_name] = prop

            # Check if required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {"type": "object", "properties": properties, "required": required}

    def _parse_docstring_params(self, docstring: str) -> Dict[str, str]:
        """Parse Args section from docstring."""
        params = {}
        in_args = False

        for line in docstring.split("\n"):
            stripped = line.strip()

            if stripped.startswith("Args:"):
                in_args = True
                continue
            elif stripped.startswith(("Returns:", "Raises:", "Example:", "Note:")):
                in_args = False
                continue

            if in_args and ":" in stripped:
                # Parse "param_name: description" or "param_name (type): description"
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    # Remove type annotation if present
                    if "(" in param_name:
                        param_name = param_name.split("(")[0].strip()
                    description = parts[1].strip()
                    params[param_name] = description

        return params

    def _python_type_to_json(self, python_type) -> str:
        """Convert Python type to JSON Schema type."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }

        # Handle Optional, Union, etc.
        origin = getattr(python_type, "__origin__", None)
        if origin is Union:
            args = getattr(python_type, "__args__", ())
            # For Optional[X] (Union[X, None]), return X's type
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return self._python_type_to_json(non_none[0])

        return type_map.get(python_type, "string")

    def _extract_code(self, func: Callable) -> str:
        """Extract source code of the function."""
        try:
            return inspect.getsource(func)
        except (OSError, TypeError):
            # If source not available, create a simple wrapper
            return f"def {func.__name__}(**kwargs):\n    pass  # Source not available"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API registration."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "code": self.code,
            "requires_approval": self.requires_approval,
        }


def custom_tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    requires_approval: bool = False,
) -> Union[ToolFunction, Callable[[Callable], ToolFunction]]:
    """
    Decorator to create a custom tool from a Python function.

    Can be used with or without arguments:

        @custom_tool
        def my_tool(arg: str) -> str:
            '''Description.'''
            return arg

        @custom_tool(requires_approval=True)
        def dangerous_tool(arg: str) -> str:
            '''Requires approval.'''
            return arg

    Args:
        func: The function to decorate (when used without parentheses)
        name: Override tool name (defaults to function name)
        description: Override description (defaults to docstring)
        requires_approval: Whether tool requires human approval

    Returns:
        ToolFunction wrapper that can be registered with the backend
    """

    def decorator(fn: Callable) -> ToolFunction:
        return ToolFunction(
            func=fn, name=name, description=description, requires_approval=requires_approval
        )

    if func is not None:
        # Called without parentheses: @custom_tool
        return decorator(func)
    else:
        # Called with parentheses: @custom_tool(...)
        return decorator
