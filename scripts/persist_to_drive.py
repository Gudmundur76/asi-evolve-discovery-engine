"""Persist daily run results to manus-persistent-drive GitHub repository.

This script is called after each scheduler run. It:
1. Reads the latest run summary from data/logs/
2. Appends a structured entry to the compounding log file
3. Commits and pushes to Gudmundur76/manus-persistent-drive

The compounding log (asi-evolve/compounding_log.md) grows with every run,
creating a persistent record of the engine's progress across all sessions.

Usage:
    python3 scripts/persist_to_drive.py --day 4 --cycles 5 --summary "Daily run complete"

Environment:
    GH_TOKEN: GitHub Personal Access Token with repo scope
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DRIVE_REPO = "Gudmundur76/manus-persistent-drive"
DRIVE_DIR = Path("/home/ubuntu/manus-persistent-drive")
LOG_DIR = Path("data/logs")
COMPOUNDING_LOG = "asi-evolve/compounding_log.md"
DAILY_LOG_TEMPLATE = "asi-evolve/day{day:02d}_run.md"


def run_git(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    env = os.environ.copy()
    gh_token = os.environ.get("GH_TOKEN", "")
    if gh_token:
        env["GH_TOKEN"] = gh_token
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, env=env
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def ensure_drive_cloned() -> bool:
    """Ensure manus-persistent-drive is cloned locally."""
    if DRIVE_DIR.exists() and (DRIVE_DIR / ".git").exists():
        logger.info("manus-persistent-drive already cloned at %s", DRIVE_DIR)
        return True

    gh_token = os.environ.get("GH_TOKEN", "")
    if not gh_token:
        logger.error("GH_TOKEN not set — cannot clone manus-persistent-drive")
        return False

    clone_url = f"https://{gh_token}@github.com/{DRIVE_REPO}.git"
    rc, out, err = run_git(
        ["git", "clone", clone_url, str(DRIVE_DIR)],
        cwd=Path("/home/ubuntu"),
    )
    if rc != 0:
        logger.error("Failed to clone manus-persistent-drive: %s", err)
        return False
    logger.info("Cloned manus-persistent-drive to %s", DRIVE_DIR)
    return True


def pull_latest() -> None:
    """Pull latest changes from remote."""
    rc, out, err = run_git(["git", "pull", "--rebase", "origin", "main"], cwd=DRIVE_DIR)
    if rc != 0:
        logger.warning("git pull failed (may be first push): %s", err)


def get_latest_run_data(day: int) -> dict:
    """Read the latest run JSON from data/logs/."""
    engine_dir = Path(__file__).parent.parent
    log_dir = engine_dir / LOG_DIR
    if not log_dir.exists():
        return {}

    # Find the most recent run file for this day
    pattern = f"run_day{day:02d}_*.json"
    matches = sorted(log_dir.glob(pattern))
    if not matches:
        # Try any recent run file
        all_runs = sorted(log_dir.glob("run_*.json"))
        if not all_runs:
            return {}
        matches = [all_runs[-1]]

    with open(matches[-1]) as f:
        return json.load(f)


def build_daily_entry(day: int, cycles: int, summary: str, run_data: dict) -> str:
    """Build a structured Markdown entry for the daily log."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    corpus_size = run_data.get("corpus_size", "N/A")
    candidates_generated = run_data.get("candidates_generated", "N/A")
    candidates_verified = run_data.get("candidates_verified", 0)
    best_pic50 = run_data.get("best_pic50", "N/A")
    convergence_candidates = run_data.get("convergence_candidates", 0)
    citation_pass_rate = run_data.get("citation_pass_rate", "N/A")

    entry = f"""## Day {day:02d} — {ts}

**Summary:** {summary}

| Metric | Value |
|---|---|
| Cycles run | {cycles} |
| Candidates generated | {candidates_generated} |
| Passed citation gate | {citation_pass_rate} |
| Added to corpus | {candidates_verified} |
| Corpus total | {corpus_size} |
| Best pIC50 | {best_pic50} |
| Convergence candidates | {convergence_candidates} |

"""
    return entry


def write_daily_log(day: int, entry: str) -> Path:
    """Write the individual day log file."""
    log_path = DRIVE_DIR / DAILY_LOG_TEMPLATE.format(day=day)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"# asi-evolve Day {day:02d} Run Log\n\n")
        f.write(entry)
    return log_path


def append_compounding_log(entry: str) -> Path:
    """Append entry to the compounding log."""
    log_path = DRIVE_DIR / COMPOUNDING_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("# asi-evolve-discovery-engine — Compounding Run Log\n\n")
            f.write("HIV-1 Protease Drug Discovery Engine  \n")
            f.write("Repository: https://github.com/Gudmundur76/asi-evolve-discovery-engine\n\n")
            f.write("---\n\n")

    with open(log_path, "a") as f:
        f.write(entry)
        f.write("---\n\n")

    return log_path


def commit_and_push(day: int, summary: str) -> bool:
    """Commit all changes and push to GitHub."""
    gh_token = os.environ.get("GH_TOKEN", "")
    if not gh_token:
        logger.error("GH_TOKEN not set — cannot push to manus-persistent-drive")
        return False

    # Set remote URL with token
    remote_url = f"https://{gh_token}@github.com/{DRIVE_REPO}.git"
    run_git(["git", "remote", "set-url", "origin", remote_url], cwd=DRIVE_DIR)

    # Configure git identity
    run_git(["git", "config", "user.email", "novus@notus.is"], cwd=DRIVE_DIR)
    run_git(["git", "config", "user.name", "novus-engine"], cwd=DRIVE_DIR)

    # Stage all changes
    rc, out, err = run_git(["git", "add", "-A"], cwd=DRIVE_DIR)
    if rc != 0:
        logger.error("git add failed: %s", err)
        return False

    # Check if there is anything to commit
    rc, out, err = run_git(["git", "status", "--porcelain"], cwd=DRIVE_DIR)
    if not out.strip():
        logger.info("Nothing to commit — manus-persistent-drive is up to date")
        return True

    commit_msg = f"asi-evolve day{day:02d}: {summary}"
    rc, out, err = run_git(
        ["git", "commit", "-m", commit_msg], cwd=DRIVE_DIR
    )
    if rc != 0:
        logger.error("git commit failed: %s", err)
        return False

    rc, out, err = run_git(
        ["git", "push", "origin", "main"], cwd=DRIVE_DIR
    )
    if rc != 0:
        logger.error("git push failed: %s", err)
        return False

    logger.info("Pushed to manus-persistent-drive: %s", commit_msg)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persist daily asi-evolve run results to manus-persistent-drive"
    )
    parser.add_argument("--day", type=int, required=True, help="Day number (1–30)")
    parser.add_argument("--cycles", type=int, default=5, help="Number of cycles run")
    parser.add_argument("--summary", type=str, default="Daily run complete")
    args = parser.parse_args()

    logger.info("Persisting day %d results to manus-persistent-drive", args.day)

    if not ensure_drive_cloned():
        sys.exit(1)

    pull_latest()

    run_data = get_latest_run_data(args.day)
    if not run_data:
        logger.warning("No run data found for day %d — writing summary only", args.day)

    entry = build_daily_entry(args.day, args.cycles, args.summary, run_data)
    daily_path = write_daily_log(args.day, entry)
    compounding_path = append_compounding_log(entry)

    logger.info("Written: %s", daily_path)
    logger.info("Appended: %s", compounding_path)

    success = commit_and_push(args.day, args.summary)
    if not success:
        logger.error("Failed to push to manus-persistent-drive")
        sys.exit(1)

    logger.info("Day %d persistence complete", args.day)


if __name__ == "__main__":
    main()
