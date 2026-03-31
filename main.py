"""Compatibility entry point for launching the Fepfm pipeline from the repo root."""


def main():
    from utils.main import main as run_main

    return run_main()


if __name__ == "__main__":
    main()
