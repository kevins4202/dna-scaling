"""Score continuations with the fixed project CNN; this command never trains it."""

from cnn.release import run_cli

if __name__ == "__main__":
    run_cli("score.py")
