from __future__ import annotations

import json

from tokentrim._llm import generate_text
from tokentrim.errors.base import TokentrimError
from tokentrim.tools.base import ToolStep
from tokentrim.tools.request import ToolsRequest
from tokentrim.types.tool import Tool


class ToolCreatorStep(ToolStep):
    """
    Ask a model to propose missing tools and return only valid additions.
    """

    def __init__(self, model: str | None) -> None:
        self._model = model

    def run(self, tools: list[Tool], request: ToolsRequest) -> list[Tool]:
        if not tools and not request.task_hint:
            return []
        if not self._model:
            raise TokentrimError(
                "Tool creation is enabled but no tool creation model is configured."
            )

        existing_names = {tool["name"] for tool in tools}
        raw_output = self._generate(tools, task_hint=request.task_hint)

        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise TokentrimError("Tool creator returned invalid JSON.") from exc

        candidates = payload["tools"] if isinstance(payload, dict) else payload
        if not isinstance(candidates, list):
            raise TokentrimError("Tool creator response must contain a tools list.")

        created: list[Tool] = []
        for candidate in candidates:
            tool = self._validate_tool(candidate)
            if tool["name"] in existing_names:
                continue
            existing_names.add(tool["name"])
            created.append(tool)

        return created

    def _generate(self, tools: list[Tool], *, task_hint: str | None) -> str:
        tool_text = json.dumps(tools, sort_keys=True)
        task_text = task_hint or "No task hint provided."
        prompt = [
            {
                "role": "system",
                "content": (
                    "Return JSON only. Produce an object with a 'tools' array. "
                    "Each tool must have string fields 'name' and 'description', "
                    "plus an object field 'input_schema'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current tools: {tool_text}\n"
                    f"Task hint: {task_text}\n"
                    "Suggest only missing tools."
                ),
            },
        ]
        try:
            return generate_text(
                model=self._model,
                messages=prompt,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except TokentrimError:
            raise
        except Exception as exc:
            raise TokentrimError("Tool creation failed.") from exc

    def _validate_tool(self, candidate: object) -> Tool:
        if not isinstance(candidate, dict):
            raise TokentrimError("Tool creator returned an invalid tool entry.")

        name = candidate.get("name")
        description = candidate.get("description")
        input_schema = candidate.get("input_schema")
        if not isinstance(name, str) or not isinstance(description, str):
            raise TokentrimError("Tool creator returned an invalid tool entry.")
        if not isinstance(input_schema, dict):
            raise TokentrimError("Tool creator returned an invalid tool entry.")

        return {
            "name": name,
            "description": description,
            "input_schema": input_schema,
        }
