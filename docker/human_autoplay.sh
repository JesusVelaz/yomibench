#!/bin/sh
# Headless human-play smoke test: the real game, rendering on in a virtual
# display, with the YomiHuman mod in autoplay (P1 random legal moves through its own action panel)
# against a bot server on the Docker host. Args: PORT [SPEED].
#   docker run --rm -v <checkout>:/yomibench:ro -v <your steamapps>:/steamapps:ro --entrypoint /bin/sh \
#     yomibench-engine /yomibench/docker/human_autoplay.sh 7790 4
rm -f /tmp/.X99-lock
Xvfb :99 -screen 0 1280x720x24 -nolisten tcp >/dev/null 2>&1 &
export DISPLAY=:99
for _ in $(seq 50); do [ -e /tmp/.X11-unix/X99 ] && break; sleep 0.1; done
RUN=/var/yomibench/human
python3 -c "from pathlib import Path; from yomibench.game import build_run_dir; \
build_run_dir(Path('/steamapps/common/YourOnlyMoveIsHUSTLE'), Path('$RUN'), 'YomiBenchHuman', ('YomiHuman', 'YomiHuman.gd'))"
cd "$RUN"
exec ./YourOnlyMoveIsHUSTLE.x86_64 --audio-driver Dummy --resolution 1280x720 \
  "--yomi-human=$1" --yomi-human-host=host.docker.internal --yomi-human-autoplay "--yomi-human-speed=${2:-1}"
