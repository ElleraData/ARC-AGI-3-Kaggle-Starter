"""ARC-AGI-3 agent: state-graph exploration with learned action effects.

Idea (the "why", for the paper):
  A new game comes with no instructions, so the agent first has to find out
  WHAT ITS ACTIONS DO. It does that with three cheap ingredients:

  1. Novelty-driven exploration. In every distinct screen it prefers actions
     it has not yet tried *in that screen*, so it does not repeat itself.
  2. Learned action effects. It keeps a running count of which actions
     actually changed the screen. Actions that never change anything are
     tried less and less; actions that do are tried more.
  3. Object-centred clicking. For the coordinate action (ACTION6) it only
     clicks on objects (connected blobs of colour) instead of random pixels,
     which shrinks the 64x64 click space to a handful of candidates.

  No game-specific rules, no model, no internet: it generalises to any game
  with the same interface, and spends few actions (RHAE rewards efficiency).

STATUS: interface checked by reading the official ARC-AGI-3-Agents source
(agent.py, templates/random_agent.py); NOT yet run on the real game engine
(no package access in the authoring sandbox). Run `make verify-local` first.
If the smart policy ever raises, choose_action falls back to random play.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`; class must be named `MyAgent`.
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
"""
from __future__ import annotations

import random
import zlib
from collections import Counter, deque
from typing import Any

from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent


def _grid(frame: FrameData) -> list[list[int]] | None:
    """Return the latest 2-D colour grid of a frame, or None if unavailable."""
    data = getattr(frame, "frame", None)
    if not data:
        return None
    last = data[-1]  # a frame may hold several animation sub-frames
    try:
        return [list(row) for row in last]
    except TypeError:
        return None


def _state_key(grid: list[list[int]] | None) -> int:
    if grid is None:
        return 0
    return hash(tuple(tuple(row) for row in grid))


def _object_centres(grid: list[list[int]] | None, limit: int = 12) -> list[tuple[int, int]]:
    """Centres of connected same-colour blobs, ignoring the background colour."""
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    background = Counter(c for row in grid for c in row).most_common(1)[0][0]
    seen = [[False] * w for _ in range(h)]
    blobs: list[tuple[int, int, int]] = []  # (size, x, y)
    for y in range(h):
        for x in range(w):
            if seen[y][x] or grid[y][x] == background:
                continue
            colour = grid[y][x]
            queue = deque([(x, y)])
            seen[y][x] = True
            cells = []
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and grid[ny][nx] == colour:
                        seen[ny][nx] = True
                        queue.append((nx, ny))
            mx = sum(c[0] for c in cells) // len(cells)
            my = sum(c[1] for c in cells) // len(cells)
            blobs.append((len(cells), mx, my))
    # Smaller blobs first: they are more often the interactive pieces.
    blobs.sort(key=lambda b: b[0])
    return [(x, y) for _, x, y in blobs[:limit]]


class MyAgent(Agent):
    """Explores novel screens first, then leans on actions that change things."""

    MAX_ACTIONS = 80  # per-game cap; the framework also enforces global limits

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # crc32 (not hash()) so a game gets the same seed on every run: reproducible.
        self._rng = random.Random(zlib.crc32(str(self.game_id).encode()))
        self._tried: dict[int, set[tuple]] = {}   # screen -> action keys tried there
        self._changed: Counter = Counter()         # action key -> times it changed screen
        self._total: Counter = Counter()           # action key -> times it was tried
        self._last: tuple[int, tuple] | None = None  # (screen before, action key)
        self._errors = 0                             # times the smart policy crashed

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Don't stop on GAME_OVER: we RESET and retry.
        return latest_frame.state is GameState.WIN

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _available_ids(latest_frame: FrameData) -> set[int] | None:
        """Ids of the actions the game currently allows, or None if unknown.

        The framework may hand us ints, GameAction members or action names
        (see agents/templates/openclaw_agent), so normalise all three.
        """
        available = getattr(latest_frame, "available_actions", None)
        if not available:
            return None
        ids: set[int] = set()
        for a in available:
            if isinstance(a, GameAction):
                ids.add(int(a.value))
                continue
            try:
                ids.add(int(a))
                continue
            except (TypeError, ValueError):
                pass
            member = next((m for m in GameAction if m.name == str(a).upper()), None)
            if member is not None:
                ids.add(int(member.value))
        return ids or None

    def _legal_simple_actions(self, latest_frame: FrameData) -> list[GameAction]:
        simple = [
            a for a in GameAction
            if a is not GameAction.RESET and not a.is_complex() and a.name != "ACTION7"
        ]
        ids = self._available_ids(latest_frame)
        if ids is None:
            return simple
        return [a for a in simple if int(a.value) in ids]

    def _click_allowed(self, latest_frame: FrameData) -> bool:
        ids = self._available_ids(latest_frame)
        if ids is None:
            return True
        return any(int(a.value) in ids for a in GameAction if a.is_complex())

    @staticmethod
    def _key(action: GameAction, xy: tuple[int, int] | None = None) -> tuple:
        return (action.name,) if xy is None else (action.name, xy[0], xy[1])

    def _score(self, key: tuple) -> float:
        """Smoothed rate at which this action changed the screen (optimistic start)."""
        return (self._changed[key] + 1.0) / (self._total[key] + 1.5)

    # ── main decision ───────────────────────────────────────────────────────
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Crash-safe wrapper: a bug in the smart policy must never end the run."""
        try:
            return self._choose_smart(frames, latest_frame)
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all
            self._errors += 1
            if self._errors == 1:
                print(f"[MyAgent] smart policy failed, using random fallback: {exc!r}")
            return self._fallback(latest_frame)

    def _fallback(self, latest_frame: FrameData) -> GameAction:
        """Plain random play (same idea as the official starter agent)."""
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            action.reasoning = {"why": "fallback reset"}
            return action
        action = self._rng.choice([a for a in GameAction if a is not GameAction.RESET])
        if action.is_complex():
            action.set_data(
                {"x": self._rng.randint(0, 63), "y": self._rng.randint(0, 63)}
            )
            action.reasoning = {"why": "fallback random click"}
        else:
            action.reasoning = f"fallback random action: {action.value}"
        return action

    def _choose_smart(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # First call or after a death → reset the level.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._last = None
            action = GameAction.RESET
            action.reasoning = {"why": "start or retry after game over"}
            return action

        grid = _grid(latest_frame)
        here = _state_key(grid)

        # Learn from the previous action: did it change the screen?
        if self._last is not None:
            before, key = self._last
            self._total[key] += 1
            if before != here:
                self._changed[key] += 1

        # Build the menu of candidate (action, optional click point) pairs.
        menu: list[tuple[GameAction, tuple[int, int] | None]] = [
            (a, None) for a in self._legal_simple_actions(latest_frame)
        ]
        click = next((a for a in GameAction if a.is_complex()), None)
        if click is not None and self._click_allowed(latest_frame):
            centres = _object_centres(grid) or [
                (self._rng.randint(0, 63), self._rng.randint(0, 63)) for _ in range(4)
            ]
            menu += [(click, xy) for xy in centres]
        if not menu:  # nothing recognised as legal: just try every simple action
            menu = [
                (a, None) for a in GameAction
                if a is not GameAction.RESET and not a.is_complex()
            ]

        tried_here = self._tried.setdefault(here, set())
        fresh = [(a, xy) for a, xy in menu if self._key(a, xy) not in tried_here]
        pool = fresh if fresh else menu  # novelty first; otherwise exploit

        weights = [self._score(self._key(a, xy)) for a, xy in pool]
        action, xy = self._rng.choices(pool, weights=weights, k=1)[0]

        key = self._key(action, xy)
        tried_here.add(key)
        self._last = (here, key)

        if xy is not None:
            action.set_data({"x": xy[0], "y": xy[1]})
            action.reasoning = {"why": f"click object at {xy}", "fresh": bool(fresh)}
        else:
            action.reasoning = f"{'new' if fresh else 'repeat'} action here: {action.name}"
        return action
