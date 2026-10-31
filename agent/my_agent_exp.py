"""EXPERIMENT FILE (v3 with switches).  Not for submission.

Switches (environment variables, default = v3 behaviour):
  ARC_MASK=0   turn off the volatile-cell (counter) filter
  ARC_SCAN=0   turn off the lattice click scan
  ARC_CLICKS=N number of blob-centre click candidates (v3: 20, v2: 12)

ARC-AGI-3 agent v3: screen-graph exploration that ignores counters and scans for clicks.

Idea (the "why", for the paper):
  A new game comes with no instructions, so the agent first has to find out
  WHAT ITS ACTIONS DO, and it has to do that with very few actions because the
  score of a level is (human_actions / agent_actions)^2.  Games are
  deterministic, so every screen the agent has seen can be remembered as a node
  in a graph and every action it tried as an edge.  Ingredients:

  1. Screen graph.  Nodes are screens, edges are the actions tried there.  The
     agent never repeats an action in a screen where it already knows the outcome.
  2. Backtracking.  When every action in the current screen is explored, a
     breadth-first search over the known edges walks to the NEAREST screen that
     still has an untried action.  Edges that led to GAME_OVER are never taken again.
  3. Learned action effects.  A running count of which action TYPES changed the
     screen; types that never do are explored last ("inert").
  4. Volatile-cell filter (new in v3).  Many games draw a step counter or an
     energy bar that changes after EVERY action.  Such cells (bottom HUD rows
     plus any cell that changed in >= 90% of all observed steps) are ignored
     when deciding whether a screen is new or an action did anything.  Without
     this, a one-pixel counter makes every action look "useful" and every
     screen look "new".
  5. Click candidates and scanning (new in v3).  For the coordinate action
     (ACTION6) the first candidates are the centres of coloured blobs.  If
     everything else has been tried and nothing happened, the agent scans the
     playfield on a coarse lattice (stride 4), in a fixed shuffled order, so a
     clickable area that is not a coloured blob is still found.

  No game-specific rules, no model, no internet: it generalises to any game
  with the same interface.

STATUS: interface checked against the official ARC-AGI-3-Agents source, behaviour
checked on toy worlds with stub modules and on the 25 public games by the
project owner (v2: local 0.396, Kaggle 0.20).  v3 is compared against v2 locally
before it is submitted.  If the smart policy ever raises, choose_action falls
back to random play.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`; class must be named `MyAgent`.
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
"""
from __future__ import annotations

import os
import random
import zlib
from collections import Counter, deque
from typing import Any

from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent

# Bottom rows that usually hold energy / step bars; always ignored when judging
# "is this screen new?".  Set to 0 to look at every row.
HUD_ROWS = 4

# An action type that changed the screen 0 times in this many tries is "inert"
# and explored last.
INERT_AFTER = 5

# A cell that changed in at least VOLATILE_RATE of all observed steps (after
# VOLATILE_MIN_TRANSITIONS steps) is a counter/timer and is ignored.  If more
# than VOLATILE_MAX_CELLS cells qualify, the whole screen is animating and the
# filter is left alone.  The set may be re-learned at most VOLATILE_MAX_UPDATES times.
VOLATILE_RATE = 0.9
VOLATILE_MIN_TRANSITIONS = 8
VOLATILE_MAX_CELLS = 150
VOLATILE_MAX_UPDATES = 4

# Coarse lattice used to look for clickable areas that are not coloured blobs.
SCAN_STRIDE = 4
SCAN_BATCH = 16

# Experiment switches (see the module docstring).
ENABLE_VOLATILE = os.environ.get("ARC_MASK", "1") != "0"
ENABLE_SCAN = os.environ.get("ARC_SCAN", "1") != "0"
CLICK_LIMIT = int(os.environ.get("ARC_CLICKS", "20"))


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


def _state_key(grid: list[list[int]] | None, volatile: frozenset = frozenset()) -> int:
    """Hash of the playfield with the volatile (counter) cells blanked out."""
    body = _playfield(grid)
    if body is None:
        return 0
    if not volatile:
        return hash(tuple(tuple(row) for row in body))
    rows = [list(row) for row in body]
    for x, y in volatile:
        if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
            rows[y][x] = -1
    return hash(tuple(tuple(row) for row in rows))


def _changed_cells(old: list[list[int]] | None, new: list[list[int]] | None) -> frozenset:
    """Cells (x, y) of the playfield whose colour differs between two grids."""
    if old is None or new is None:
        return frozenset()
    if len(old) != len(new):
        return frozenset({(-1, -1)})  # shape changed: treat as "everything changed"
    cut = max(1, len(new) - HUD_ROWS)
    cells = []
    for y in range(cut):
        ro, rn = old[y], new[y]
        if ro != rn:
            cells.extend((x, y) for x, (a, b) in enumerate(zip(ro, rn)) if a != b)
    return frozenset(cells)


def _object_centres(grid: list[list[int]] | None, limit: int = 20) -> list[tuple[int, int]]:
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
    """Explores a screen graph, backtracks, ignores counters, scans for clicks."""

    # Per-game cap on actions.  v1 used 80 (the starter default).  The Kaggle run
    # of v2 with 300 finished in about 20 minutes.
    MAX_ACTIONS = 300

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # crc32 (not hash()) so a game gets the same seed on every run: reproducible.
        self._rng = random.Random(zlib.crc32(str(self.game_id).encode()))
        self._by_name = {a.name: a for a in GameAction}
        self._click = next((a for a in GameAction if a.is_complex()), None)
        self._scan = [
            (x, y)
            for y in range(SCAN_STRIDE // 2, 64 - HUD_ROWS, SCAN_STRIDE)
            for x in range(SCAN_STRIDE // 2, 64, SCAN_STRIDE)
        ]
        self._rng.shuffle(self._scan)
        # Kept for the whole game (mechanics and counters carry over between levels).
        self._changed: Counter = Counter()   # action TYPE -> times it changed the screen
        self._total: Counter = Counter()     # action TYPE -> times it was tried
        self._flips: Counter = Counter()     # cell -> times it changed
        self._ntrans = 0                     # steps observed
        self._records: list[tuple[str, frozenset]] = []  # (action type, cells it changed)
        self._volatile: frozenset = frozenset()
        self._volatile_updates = 0
        self._levels = 0                     # levels completed so far
        self._steps_this_level = 0
        self._errors = 0                     # times the smart policy crashed
        self._reset_level_memory()

    def _reset_level_memory(self) -> None:
        """Forget the screen graph (it is only valid inside one level)."""
        self._graph: dict[int, dict[tuple, int]] = {}   # screen -> {action key -> next screen}
        self._menus: dict[int, list[tuple]] = {}        # screen -> candidate action keys
        self._dead: set[tuple[int, tuple]] = set()      # (screen, action key) that ended the game
        self._scan_ptr: dict[int, int] = {}             # screen -> next lattice index to try
        self._last: tuple[int, tuple] | None = None     # (screen before, action key)
        self._prev_grid: list[list[int]] | None = None  # the screen before the last action

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
        if self._click is None:
            return False
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
            if self._click_allowed(latest_frame):
                centres = _object_centres(_playfield(grid), CLICK_LIMIT) or [
                    (self._rng.randint(0, 63), self._rng.randint(0, 63)) for _ in range(4)
                ]
                keys += [self._key(self._click, xy) for xy in centres]
            if not keys:  # nothing recognised as legal: just try every simple action
                keys = [
                    self._key(a) for a in GameAction
                    if a is not GameAction.RESET and not a.is_complex()
                ]
            self._menus[here] = keys
        return self._menus[here]

    def _extend_clicks(self, here: int) -> bool:
        """Add the next batch of lattice click points to this screen's menu."""
        menu = self._menus[here]
        have = set(menu)
        ptr = self._scan_ptr.get(here, 0)
        added = False
        while ptr < len(self._scan) and not added:
            for xy in self._scan[ptr: ptr + SCAN_BATCH]:
                key = self._key(self._click, xy)
                if key not in have:
                    menu.append(key)
                    have.add(key)
                    added = True
            ptr += SCAN_BATCH
        self._scan_ptr[here] = ptr
        return added

    # ── learning: what did the last action change? ──────────────────────────
    def _rebuild_stats(self) -> None:
        self._changed, self._total = Counter(), Counter()
        for name, cells in self._records:
            self._total[name] += 1
            if cells - self._volatile:
                self._changed[name] += 1

    def _record(self, name: str, cells: frozenset) -> bool:
        """Log one observed step. True if the set of volatile cells changed."""
        self._ntrans += 1
        self._flips.update(cells)
        self._records.append((name, cells))
        self._total[name] += 1
        if cells - self._volatile:
            self._changed[name] += 1
        if not ENABLE_VOLATILE:
            return False
        if self._volatile_updates >= VOLATILE_MAX_UPDATES or self._ntrans < VOLATILE_MIN_TRANSITIONS:
            return False
        limit = VOLATILE_RATE * self._ntrans
        volatile = frozenset(c for c, n in self._flips.items() if n >= limit and c[0] >= 0)
        if volatile == self._volatile or len(volatile) > VOLATILE_MAX_CELLS:
            return False
        self._volatile = volatile
        self._volatile_updates += 1
        self._rebuild_stats()
        return True

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
        levels = int(getattr(latest_frame, "levels_completed", 0) or 0)

        # A new level has new screens: forget the graph, keep what we learnt
        # about what each action TYPE does (the mechanics usually carry over).
        if levels != self._levels:
            if levels > self._levels and self._last is not None:
                kind = self._last[1][0]
                self._records.append((kind, frozenset({(-1, -1)})))
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

        # Learn from the previous action: which cells did it change?
        if self._last is not None:
            cells = _changed_cells(self._prev_grid, grid)
            if self._record(self._last[1][0], cells):
                # The meaning of a "screen" just changed (a counter was found):
                # old screen ids are stale, so start the graph afresh.
                self._reset_level_memory()

        here = _state_key(grid, self._volatile)
        if self._last is not None:
            before, key = self._last
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
            elif ENABLE_SCAN and self._click_allowed(latest_frame) and self._extend_clicks(here):
                untried = [k for k in menu if k not in node and (here, k) not in self._dead]
                key, why = self._pick(untried), "scan"
            else:
                step = self._step_toward_frontier(here, live_only=False)
                if step is not None:
                    key, why = step, "backtrack-probe"
                else:
                    allowed = [k for k in menu if (here, k) not in self._dead] or menu
                    key, why = self._pick(allowed), "repeat"

        self._last = (here, key)
        self._prev_grid = grid
        action = self._by_name[key[0]]
        if len(key) == 3:
            action.set_data({"x": key[1], "y": key[2]})
            action.reasoning = {"why": f"{why}: click at {key[1:]}"}
        else:
            action.reasoning = f"{why} action here: {action.name}"
        return action
