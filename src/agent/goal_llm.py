import json
from typing import Any, Dict, Optional
import urllib.error
import urllib.request

from src.utils.game_data import map_id_to_name


class PokemonGoalLLM:
    """
    Lightweight client for a local text-only goal-setting LLM.

    The model consumes a structured JSON summary of the current game state
    (no images) and returns a JSON object describing the next high-level goal.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:11434/api/chat",
        model: str = "pokemon-goal",
        enabled: bool = True,
        timeout: float = 60.0,
        debug: bool = True,
    ):
        """Configure the HTTP client for the local goal-setting LLM."""
        self.api_url = api_url
        self.model = model
        self.enabled = enabled
        self.timeout = timeout
        self.debug = debug

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

    def generate_goal(self, state_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send the compact state summary (JSON-serializable dict) to the local LLM
        and return its JSON response as a dict. Returns None when disabled or on
        failure. No images are transmitted.
        """
        if self.debug:
            print(f"[LLM][DEBUG] generate_goal called | enabled={self.enabled}")
        if not self.enabled:
            if self.debug:
                print("[LLM][DEBUG] LLM disabled; skipping request")
            return None

        # Ensure we never send bare integers without context for the map name.
        state_summary = state_summary.copy()
        location = state_summary.get("location", {}) or {}
        if location.get("map_name") is None and "map_id" in location:
            location["map_name"] = map_id_to_name(location["map_id"])
            state_summary["location"] = location

        payload = {
            "model": self.model,
            "stream": False,
            # Do not disable model thinking; omit the flag so servers that support it can return it.
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(state_summary),
                }
            ],
        }

        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            if self.debug:
                print(f"[LLM][REQ] -> {self.api_url} | payload={payload}")
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
            if self.debug:
                print(f"[LLM][RESP RAW] {raw}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if self.debug:
                print(f"[LLM][ERR] request failed: {exc}")
            return None
        except Exception as exc:
            if self.debug:
                print(f"[LLM][ERR] unexpected exception: {exc}")
            return None

        parsed = self._parse_raw_response(raw) or raw

        content: Optional[str] = None
        if isinstance(parsed, dict):
            if "message" in parsed and isinstance(parsed["message"], dict):
                content = parsed["message"].get("content")
            if content is None and "content" in parsed:
                content = parsed.get("content")
            if content is None and "response" in parsed:
                content = parsed.get("response")
            if content is None and "thinking" in parsed:
                # Only use thinking if primary content is absent; helps when the model only returns thoughts.
                content = parsed.get("thinking")
        elif isinstance(parsed, str):
            content = parsed

        if self.debug:
            print(f"[LLM][PARSED] content={content!r}")

        if not content:
            return None

        goal = self._parse_goal_json(content)
        if self.debug:
            print(f"[LLM][GOAL] parsed={goal}")
        return goal
