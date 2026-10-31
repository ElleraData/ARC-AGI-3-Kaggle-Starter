"""Compare agent versions fairly: play all local games with several random seeds.

A single local run is mostly luck: only a handful of levels are ever cleared, and
ANY code change also changes the random sequence.  This script plays every game
with N different seeds and reports the mean score.

Usage (from the project folder):
    .venv/bin/python scripts/eval_seeds.py --agent agent/my_agent.py --seeds 4 --steps 300
    .venv/bin/python scripts/eval_seeds.py --agent agent/my_agent_v2.py --seeds 4 --steps 300
Works with any agent file that defines `MyAgent` and draws randomness from `self._rng`.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import random
import statistics
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402


def load_agent_class(path: Path):
    spec = importlib.util.spec_from_file_location("agent_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


def seeded(cls, steps: int, salt: int):
    class Seeded(cls):
        MAX_ACTIONS = steps

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._rng = random.Random(zlib.crc32(f"{self.game_id}#{salt}".encode()))
            if hasattr(self, "_scan"):  # lattice order of v3 and later
                self._rng.shuffle(self._scan)

    return Seeded


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="agent/my_agent.py", help="Agent file (defines MyAgent).")
    p.add_argument("--seeds", type=int, default=4, help="Number of different seeds.")
    p.add_argument("--steps", type=int, default=300, help="Actions per game.")
    args = p.parse_args()

    logging.basicConfig(level=logging.ERROR)
    cls = load_agent_class(Path(args.agent))
    print(f"# eval: agent={args.agent} seeds={args.seeds} steps={args.steps}", flush=True)

    scores = []
    for salt in range(1, args.seeds + 1):
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)  # fresh scorecard per seed
        games = [e.game_id.split("-")[0] for e in arc.get_environments()]
        cleared = {}
        for gid in games:
            env = arc.make(gid, render_mode=None)
            if env is None:
                continue
            agent = seeded(cls, args.steps, salt)(
                card_id="eval", game_id=gid, agent_name=f"eval.{gid}",
                ROOT_URL="http://localhost", record=False, arc_env=env, tags=["eval"],
            )
            agent.main()
            cleared[gid] = agent.frames[-1].levels_completed
        sc = arc.get_scorecard()
        score = float(sc.score if hasattr(sc, "score") else sc)
        scores.append(score)
        wins = sorted(g for g, v in cleared.items() if v > 0)
        print(
            f"seed {salt}: score={score:.3f} games_with_a_level={len(wins)} "
            f"levels={sum(cleared.values())} -> {','.join(wins) or '-'}",
            flush=True,
        )
    print(
        f"MEAN over {len(scores)} seeds = {statistics.mean(scores):.3f} "
        f"(min {min(scores):.3f}, max {max(scores):.3f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
