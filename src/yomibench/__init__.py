"""Bots and a tournament arena for Your Only Move Is HUSTLE."""

# Engines beyond the public ones register themselves from an optional module (yomibench.engine: plugins). It is
# imported here, on the importing (main) thread, before any engine thread or pool starts: every entry point
# (yomibench.cli, tools/*) imports this package at top level.
try:
    from yomibench import engines_private  # noqa: F401
except ImportError:  # the public build: the real game, the toy engine, tcp and cmd engines
    pass
