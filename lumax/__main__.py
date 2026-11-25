"""Allow `python -m lumax <stage>` invocation."""

from lumax.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
