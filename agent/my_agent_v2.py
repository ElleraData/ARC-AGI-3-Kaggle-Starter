"""ARC-AGI-3 agent v2: state-graph exploration with backtracking.

Idea (the "why", for the paper):
  A new game comes with no instructions, so the agent first has to find out
  WHAT ITS ACTIONS DO, and it has to do that with very few actions because the
  score of a level is (human_actions / agent_actions)^2.  Games are
  deterministic, so every screen the agent has seen can be remembered as a node
  in a graph and every action it tried as an edge.  Four cheap ingredients:

  1. Screen graph.  Nodes are screens (the bottom HUD rows are ignored, so a
     shrinking energy bar does not make every screen look new).  Edges are the
     actions tried there.  The agent never repeats an action in a screen where
     it already knows the outcome.
  2. Backtracking.  When every action in the current screen is explored, a
     breadth-first search over the known edges walks to the NEAREST screen that
     still has an untried action, instead of wandering.  Edges that led to
     GAME_OVER are never taken again.
  3. Learned action effects.  A running count of which action TYPES changed the
     screen: types that never do are tried less, types that do are tried more.
  4. Object-centred clicking.  For the coordinate action (ACTION6) only the
     centres of coloured objects are candidates, which shrinks the 64x64 click
     space to a handful of points.

  No game-specific rules, no model, no internet: it generalises to any game
  with the same interface.

STATUS: interface checked by reading the official ARC-AGI-3-Agents source and
exercised on toy worlds with stub modules; the real game engine was NOT
available in the authoring sandbox.  Run `make play-local` first and compare
with v1 (local score 0.208 at 80 actions).
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

# Bottom rows that usually hold energy / step bars.  They are ignored when
# deciding "have I seen this screen before?".  Set to 0 to look at every row.
HUD_ROWS = 4

# An action type that changed the screen 0 times in this many tries is treated
# as "inert" and explored last.
INERT_AFTER = 5


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


def _playfield(grid: list[list[int]] | None) -> list[list[int]] | None:
    """The grid without its bottom HUD rows."""
    if not grid:
        return grid
    return grid[: max(1, len(grid) - HUD_ROWS)]


def _state_key(grid: list[list[int]] | None) -> int:
    body = _playfield(grid)
    if body is None:
        return 0
    return hash(tuple(tuple(row) for row in body))


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
    """Explores a screen graph, backtracks to the nearest unexplored screen."""

    # Per-game cap on actions.  v1 used 80 (the starter default), which is
    # often too few to finish even the first level.  The first Kaggle run with
    # 80 actions took well under 20 minutes, so 300 is safe on runtime.
    MAX_ACTIONS = 300

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # crc32 (not hash()) so a game gets the same seed on every run: reproducible.
        self._rng = random.Random(zlib.crc32(str(self.game_id).encode()))
        self._by_name = {a.name: a for a in GameAction}
        self._reset_level_memory()
        self._changed: Counter = Counter()   # action TYPE -> times it changed the screen
        self._total: Counter = Counter()     # action TYPE -> times it was tried
        self._levels = 0                     # levels completed so far
        self._steps_this_level = 0
        self._errors = 0                     # times the smart policy crashed

    def _reset_level_memory(self) -> None:
        """Forget the screen graph (it is only valid inside one level)."""
        self._graph: dict[int, dict[tuple, int]] = {}   # screen -> {action key -> next screen}
        self._menus: dict[int, list[tuple]] = {}        # screen -> candidate action keys
        self._dead: set[tuple[int, tuple]] = set()      # (screen, action key) that ended the game
        self._last: tuple[int, tuple] | None = None     # (screen before, action key)

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

    def _score(self, action_name: str) -> float:
        """Smoothed rate at which this action TYPE changed the screen (optimistic start)."""
        return (self._changed[action_name] + 1.0) / (self._total[action_name] + 1.5)

    def _menu(self, here: int, grid: list[list[int]] | None, latest_frame: FrameData) -> list[tuple]:
        """Candidate action keys for this screen (click points cached per screen)."""
        if here not in self._menus:
            keys = [self._key(a) for a in self._legal_simple_actions(latest_frame)]
            click = next((a for a in GameAction if a.is_complex()), None)
            if click is not None and self._click_allowed(latest_frame):
                centres = _object_centres(_playfield(grid)) or [
                    (self._rng.randint(0, 63), self._rng.randint(0, 63)) for _ in range(4)
                ]
                keys += [self._key(click, xy) for xy in centres]
            if not keys:  # nothing recognised as legal: just try every simple action
                keys = [
                    self._key(a) for a in GameAction
                    if a is not GameAction.RESET and not a.is_complex()
                ]
            self._menus[here] = keys
        return self._menus[here]

    def _inert(self, action_name: str) -> bool:
        """True once an action TYPE has been tried a few times and never changed anything."""
        return self._total[action_name] >= INERT_AFTER and self._changed[action_name] == 0

    def _has_untried(self, state: int, live_only: bool) -> bool:
        tried = self._graph.get(state, {})
        return any(
            k not in tried
            and (state, k) not in self._dead
            and not (live_only and self._inert(k[0]))
            for k in self._menus.get(state, [])
        )

    def _pick(self, keys: list[tuple]) -> tuple:
        """Pick an action TYPE by its score, then a point within it uniformly."""
        by_type: dict[str, list[tuple]] = {}
        for k in keys:
            by_type.setdefault(k[0], []).append(k)
        names = list(by_type)
        name = self._rng.choices(names, weights=[self._score(n) for n in names], k=1)[0]
        return self._rng.choice(by_type[name])

    def _step_toward_frontier(self, here: int, live_only: bool) -> tuple | None:
        """First action of the shortest known path to a screen with untried actions."""
        parent: dict[int, tuple[int, tuple] | None] = {here: None}
        queue = deque([here])
        while queue:
            s = queue.popleft()
            if s != here and self._has_untried(s, live_only):
                while parent[s][0] != here:  # type: ignore[index]
                    s = parent[s][0]  # type: ignore[index]
                return parent[s][1]  # type: ignore[index]
            for key, nxt in self._graph.get(s, {}).items():
                if nxt == s or nxt in parent or (s, key) in self._dead:
                    continue
                parent[nxt] = (s, key)
                queue.append(nxt)
        return None

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
        # First call: nothing to learn from yet.
        if latest_frame.state is GameState.NOT_PLAYED:
            self._last = None
            action = GameAction.RESET
            action.reasoning = {"why": "start"}
            return action

        # Died: remember which step killed us (never take it again), then retry.
        if latest_frame.state is GameState.GAME_OVER:
            if self._last is not None:
                self._dead.add(self._last)
            self._last = None
            action = GameAction.RESET
            action.reasoning = {"why": "retry after game over"}
            return action

        grid = _grid(latest_frame)
        here = _state_key(grid)
        levels = int(getattr(latest_frame, "levels_completed", 0) or 0)

        # A new level has new screens: forget the graph, keep what we learnt
        # about what each action TYPE does (the mechanics usually carry over).
        if levels != self._levels:
            if levels > self._levels and self._last is not None:
                kind = self._last[1][0]
                self._total[kind] += 1
                self._changed[kind] += 1
                print(
                    f"[MyAgent] {self.game_id}: level {levels} reached after "
                    f"{self._steps_this_level} actions in the previous level"
                )
            self._levels = levels
            self._steps_this_level = 0
            self._reset_level_memory()
        self._steps_this_level += 1

        # Learn from the previous action: where did it lead?
        if self._last is not None:
            before, key = self._last
            self._total[key[0]] += 1
            if before != here:
                self._changed[key[0]] += 1
            self._graph.setdefault(before, {})[key] = here

        menu = self._menu(here, grid, latest_frame)
        node = self._graph.setdefault(here, {})
        untried = [k for k in menu if k not in node and (here, k) not in self._dead]

        # Action types that never changed anything (after a few tries) are
        # "inert": explore everything else first, probe the inert ones last.
        live = [k for k in untried if not self._inert(k[0])]
        if live:
            key, why = self._pick(live), "new"
        else:
            step = self._step_toward_frontier(here, live_only=True)
            if step is not None:
                key, why = step, "backtrack"
            elif untried:
                key, why = self._pick(untried), "probe"
            else:
                step = self._step_toward_frontier(here, live_only=False)
                if step is not None:
                    key, why = step, "backtrack-probe"
                else:
                    allowed = [k for k in menu if (here, k) not in self._dead] or menu
                    key, why = self._pick(allowed), "repeat"

        self._last = (here, key)
        action = self._by_name[key[0]]
        if len(key) == 3:
            action.set_data({"x": key[1], "y": key[2]})
            action.reasoning = {"why": f"{why}: click object at {key[1:]}"}
        else:
            action.reasoning = f"{why} action here: {action.name}"
        return action
