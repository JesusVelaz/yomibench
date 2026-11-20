#!/bin/sh
# Godot 3's X11 build needs a display even with --no-window; give it a virtual one.
# A restarted container keeps /tmp, and Xvfb refuses to start over its old lock file.
rm -f /tmp/.X99-lock
Xvfb :99 -screen 0 320x240x24 -nolisten tcp >/dev/null 2>&1 &
export DISPLAY=:99
# Wait for the display, or the first servers start before it exists.
for _ in $(seq 50); do [ -e /tmp/.X11-unix/X99 ] && break; sleep 0.1; done
exec python3 -m yomibench.cli serve --host '*' "$@"
