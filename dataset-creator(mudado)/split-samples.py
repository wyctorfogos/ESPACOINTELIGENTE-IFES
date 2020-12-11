import os
import json
from subprocess import Popen, PIPE, STDOUT
from is_wire.core import Logger

log = Logger(name="SplitSamples")

if not os.path.exists("samples"):
    log.critical("'samples' folder not found")
if not os.path.exists("samples/spots.json"):
    log.critical("'samples/spots.json' file not found")

with open("samples/spots.json", 'r') as f:
    spots = json.load(f)

ffmpeg_command = "ffmpeg -y -i gestures.MOV -ss {ss:.2f} -t {t:.2f} -an samples/{gesture:02d}.MOV"
for spot in spots:
    g_id = int(spot['gesture'])
    command = ffmpeg_command.format(ss=spot['ss'], t=spot['t'], gesture=g_id)
    process = Popen(command.split(), stdout=PIPE, stderr=STDOUT)
    log.info("{}", command)
    if process.wait() == 0:
        log.info("{} | {:.2f} + {:.2f}", g_id, spot['ss'], spot['t'])
    else:
        log.error("Failed to split video gesture: {}", g_id)
