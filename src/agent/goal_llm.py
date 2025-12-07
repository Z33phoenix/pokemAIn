import json
from typing import Any, Dict, Optional
import urllib.error
import urllib.request


class PokemonGoalLLM:
    """
    Lightweight client for a local goal-setting LLM (e.g., Ollama).

    The model is expected to return a JSON object containing the Goal
    dataclass fields: goal_type, priority, target, metadata, max_steps.
    Note: the prompt/model definition itself is kept in ModelFile.txt for
    local LLMs; this client only calls the HTTP API and does not read that
    file directly.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:11434/api/chat",
        model: str = "pokemon-goal",
        enabled: bool = False,
        timeout: float = 60.0,
    ):
        """Configure the HTTP client for the local goal-setting LLM."""
        self.api_url = api_url
        self.model = model
        self.enabled = enabled
        self.timeout = timeout

    def _parse_raw_response(self, raw: str) -> Optional[Dict[str, Any]]:
        """Handle both standard JSON and potential NDJSON streaming outputs."""
        try:
            return json.loads(raw)
        except Exception:
            pass

        # Fallback: handle newline-delimited JSON (streaming) by combining content.
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        combined_content = ""
        last_obj: Optional[Dict[str, Any]] = None
        for line in lines:
            try:
                obj = json.loads(line)
                last_obj = obj
                msg = obj.get("message", {})
                if msg and isinstance(msg, dict):
                    piece = msg.get("content")
                    if piece:
                        combined_content += piece
                elif "response" in obj:
                    piece = obj.get("response")
                    if piece:
                        combined_content += str(piece)
            except Exception:
                continue
        if combined_content:
            return {"content": combined_content}
        return last_obj

    def _parse_goal_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse a JSON object from the model response."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Some models wrap JSON in text; try to extract the first object.
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start : end + 1])
                except Exception:
                    return None
        except Exception:
            return None
        return None

    def _normalize_goal(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in required fields with defaults when missing."""
        target = goal.get("target") or {}
        metadata = goal.get("metadata") or {}
        if not isinstance(target, dict):
            target = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "goal_type": goal.get("goal_type", "explore"),
            "priority": int(goal.get("priority", 0) or 0),
            "target": target,
            "metadata": metadata,
            "max_steps": int(goal.get("max_steps", 0) or 0),
            "name": goal.get("name"),
        }

    def generate_goal(self, state_summary: Dict[str, Any], obs_image_b64: str) -> Optional[Dict[str, Any]]:
        """
        Send the compact state summary plus a base64-encoded screen grab to the local LLM and return a goal dict.
        Returns None when disabled or on failure.
        """
        if not self.enabled:
            return None

        party_size = state_summary.get("state_party_size") or 0

        prompt_parts = [
            "You are the goal planner. Base your decision on the current screen (base64 PNG) and the state summary.",
            "Rules: if state_party_size is 0, first obtain the starter Pokemon (avoid menu/parcel goals until then).",
            "Respond with a JSON object containing: goal_type, priority, target, metadata, max_steps (and optional name).",
            "State summary:",
            json.dumps(state_summary),
            "Screen (base64 PNG, 84x84 grayscale):",
            obs_image_b64,
        ]
        user_content = "\n".join(prompt_parts)

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "user", "content": user_content}
            ],
        }

        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            print(f"[DEBUG][LLM] Sending goal request to {self.api_url} with payload: {payload}")
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
            print(f"[DEBUG][LLM] Received raw response: {raw}")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return None
        except Exception:
            return None

        parsed = self._parse_raw_response(raw) or raw

        content: Optional[str] = None
        if isinstance(parsed, dict):
            # Ollama chat responses place content under message.content
            if "message" in parsed and isinstance(parsed["message"], dict):
                content = parsed["message"].get("content")
            if content is None and "content" in parsed:
                content = parsed.get("content")
            if content is None and "response" in parsed:
                content = parsed.get("response")
            if (content is None or content == "") and "thinking" in parsed:
                # Some ollama backends return JSON content in "thinking" when using generate+format=json.
                content = parsed.get("thinking")
        elif isinstance(parsed, str):
            content = parsed

        if not content:
            # Fallback to a safe explore goal when the model returns no content.
            return {
                "goal_type": "explore",
                "priority": 10,
                "target": {"novel_states": 1, "prefer_new_map": True},
                "metadata": {},
                "max_steps": 256,
                "name": "llm-fallback-empty",
            }

        goal = self._parse_goal_json(content)
        if goal is None:
            return {
                "goal_type": "explore",
                "priority": 10,
                "target": {"novel_states": 1, "prefer_new_map": True},
                "metadata": {},
                "max_steps": 256,
                "name": "llm-fallback-parse",
            }
        normalized = self._normalize_goal(goal)

        # Guardrail: early game should focus on getting the starter, not menu actions.
        if party_size <= 0 and normalized.get("goal_type") == "menu":
            normalized["goal_type"] = "explore"
            normalized["target"] = {"novel_states": 1, "prefer_new_map": True}
            normalized["metadata"]["reason"] = "starter_required_before_menu"
        return normalized
