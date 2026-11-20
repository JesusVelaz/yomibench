#!/bin/sh
# Render replays to MP4 in the container: `docker run --rm -v <your steamapps>:/steamapps:ro -v <checkout>:/yomibench:ro
#   -v <replays and output>:/work --entrypoint /bin/sh yomibench-engine /yomibench/docker/render.sh
#   /work/a.replay ... --out /work/videos`
rm -f /tmp/.X99-lock
Xvfb :99 -screen 0 640x360x24 -nolisten tcp >/dev/null 2>&1 &
export DISPLAY=:99
for _ in $(seq 50); do [ -e /tmp/.X11-unix/X99 ] && break; sleep 0.1; done
exec python3 -m yomibench.cli render "$@"
