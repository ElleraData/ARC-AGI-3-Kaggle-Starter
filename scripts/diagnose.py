"""Diagnose every local ARC-AGI-3 game: what do the actions do, and where does
MyAgent get stuck?  Read-only: it only plays games locally.

Usage (from the project folder):
    .venv/bin/python scripts/diagnose.py            # all games, 300 agent steps
    .venv/bin/python scripts/diagnose.py --game ls20,vc33 --steps 100
The report is written to diagnose.txt (and a short copy is printed).

For each game it prints
  PROBE : for every available action, how many cells change on the first,
          second and third press from the start screen (movement shows up as
          the same small number each press), and for clicks how many of the
          clickable objects change the screen.
  AGENT : a run of MyAgent: levels cleared, resets, screens seen in the
          current level, how many actions were no-ops, per-action-type
          change rates.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402


def load_agent_module():
    spec = importlib.util.spec_from_file_location("user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_agent(cls, env, game_id):
    return cls(
        card_id="diagnose", game_id=game_id, agent_name=f"diag.{game_id}",
        ROOT_URL="http://localhost", record=False, arc_env=env, tags=["diagnose"],
    )


def take(agent, action, xy=None):
    """Send one action through the framework and return the new FrameData (or None)."""
    if xy is not None:
        action.set_data({"x": xy[0], "y": xy[1]})
    action.reasoning = {"why": "diagnose"}
    return agent.take_action(action)


def diff(g_old, g_new, hud_rows):
    """(cells changed in the playfield, bounding box text, cells changed in the HUD rows)."""
    if g_old is None or g_new is None or len(g_old) != len(g_new):
        return -1, "?", -1
    cut = max(1, len(g_old) - hud_rows)
    xs, ys, play, hud = [], [], 0, 0
    for y, (ro, rn) in enumerate(zip(g_old, g_new)):
        for x, (a, b) in enumerate(zip(ro, rn)):
            if a != b:
                if y < cut:
                    play += 1
                    xs.append(x)
                    ys.append(y)
                else:
                    hud += 1
    box = f"x{min(xs)}-{max(xs)} y{min(ys)}-{max(ys)}" if xs else "-"
    return play, box, hud


def probe_game(m, arc, game_id, out):
    env = arc.make(game_id, render_mode=None)
    agent = make_agent(m.MyAgent, env, game_id)
    f0 = take(agent, GameAction.RESET)
    if f0 is None:
        out("  PROBE: could not reset the game")
        return
    g0 = m._grid(f0)
    ids = m.MyAgent._available_ids(f0)
    shape = f"{len(g0)}x{len(g0[0])}" if g0 else "?"
    colours = sorted({c for row in g0 for c in row}) if g0 else []
    centres = m._object_centres(m._playfield(g0))
    out(
        f"  PROBE: grid={shape} sub-frames={len(f0.frame)} colours={len(colours)} "
        f"win_levels={getattr(f0, 'win_levels', '?')} available={sorted(ids) if ids else 'unknown'} "
        f"objects={len(centres)}"
    )

    simple = [a for a in GameAction if a is not GameAction.RESET and not a.is_complex()]
    for a in simple:
        if ids is not None and int(a.value) not in ids:
            continue
        f = take(agent, GameAction.RESET)
        prev = m._grid(f)
        seq, box, end = [], "-", ""
        for _ in range(3):
            f = take(agent, a)
            if f is None:
                seq.append("invalid")
                break
            cur = m._grid(f)
            n, b, hud = diff(prev, cur, m.HUD_ROWS)
            seq.append(f"{n}" + (f"+hud{hud}" if hud > 0 else ""))
            if n > 0 and box == "-":
                box = b
            prev = cur
            if f.state is GameState.GAME_OVER:
                end = " GAME_OVER"
                break
            if f.state is GameState.WIN:
                end = " WIN"
                break
        out(f"    {a.name}: changed cells per press {seq} first-change-box {box}{end}")

    click = next((a for a in GameAction if a.is_complex()), None)
    if click is not None and (ids is None or int(click.value) in ids):
        hits, sizes = 0, []
        for xy in centres:
            f = take(agent, GameAction.RESET)
            before = m._grid(f)
            f = take(agent, click, xy)
            if f is None:
                continue
            n, _, _ = diff(before, m._grid(f), m.HUD_ROWS)
            sizes.append(n)
            hits += n > 0
        out(f"    {click.name} clicks on {len(centres)} objects: {hits} changed the screen; cells changed {sizes}")


def agent_run(m, arc, game_id, steps, out):
    class Counting(m.MyAgent):
        MAX_ACTIONS = steps

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._diag = Counter()

        def choose_action(self, frames, latest_frame):
            action = super().choose_action(frames, latest_frame)
            self._diag[action.name] += 1
            return action

    env = arc.make(game_id, render_mode=None)
    agent = make_agent(Counting, env, game_id)
    agent.main()
    final = agent.frames[-1]
    tot = sum(agent._total.values())
    chg = sum(agent._changed.values())
    types = ", ".join(
        f"{k}:{agent._changed[k]}/{agent._total[k]}" for k in sorted(agent._total)
    )
    edges = sum(len(v) for v in agent._graph.values())
    out(
        f"  AGENT: levels={final.levels_completed}/{getattr(final, 'win_levels', '?')} "
        f"state={final.state.name} actions={agent.action_counter} resets={agent._diag['RESET']} "
        f"errors={agent._errors}"
    )
    out(
        f"         screens(this level)={len(agent._graph)} edges={edges} dead={len(agent._dead)} "
        f"no-op rate={(tot - chg) / tot:.0%}" if tot else "         (no learning steps)"
    )
    out(f"         changed/tried per type: {types}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default=None, help="Comma-separated game ids (default: all).")
    p.add_argument("--steps", type=int, default=300, help="Actions per game for the agent run.")
    p.add_argument("--out", default="diagnose.txt", help="Report file.")
    args = p.parse_args()

    logging.basicConfig(level=logging.ERROR)
    m = load_agent_module()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    envs = arc.get_environments()
    ids = [e.game_id.split("-")[0] for e in envs]
    if args.game:
        wanted = {g.strip().split("-")[0] for g in args.game.split(",")}
        ids = [g for g in ids if g in wanted]

    lines: list[str] = []

    def out(text: str) -> None:
        lines.append(text)
        print(text, flush=True)

    out(f"# diagnose: {len(ids)} games, agent steps={args.steps}, HUD_ROWS={m.HUD_ROWS}")
    for i, gid in enumerate(ids, 1):
        out(f"[{i}/{len(ids)}] {gid}")
        for name, fn in (("PROBE", lambda: probe_game(m, arc, gid, out)),
                         ("AGENT", lambda: agent_run(m, arc, gid, args.steps, out))):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - keep going, report the failure
                tb = traceback.format_exc().strip().splitlines()[-3:]
                out(f"  {name} FAILED: {exc!r} | " + " / ".join(t.strip() for t in tb))
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nReport saved to {args.out}")


if __name__ == "__main__":
    main()
