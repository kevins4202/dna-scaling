"""Compatibility entry point for scripts/score_cnn.py."""

from cnn.release import run_cli

if __name__ == "__main__":
    run_cli("score.py")
