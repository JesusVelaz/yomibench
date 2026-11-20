# Runs the game's Linux build as yomibench engine servers. On Apple Silicon this is x86_64
# under emulation (Docker Desktop's Rosetta), so expect it to be slower per process than native.
# The image holds no game files: mount your own Steam library's steamapps folder at /steamapps (the Linux build,
# installed with your Steam account) and this checkout at /yomibench.
FROM ubuntu:24.04
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 xvfb libx11-6 libxcursor1 libxinerama1 libxrandr2 libxi6 libxrender1 libxext6 \
      libgl1 libglx-mesa0 libgl1-mesa-dri libasound2t64 libpulse0 libudev1 ca-certificates ffmpeg \
 && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/yomibench/src YOMIBENCH_GAME_DIR=/steamapps/common/YourOnlyMoveIsHUSTLE YOMIBENCH_RUN_DIR=/var/yomibench/run
WORKDIR /yomibench
ENTRYPOINT ["/bin/sh", "/yomibench/docker/entrypoint.sh"]
CMD ["--count", "4", "--port", "7777"]
