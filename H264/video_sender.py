import subprocess
import io
import sys
import time

# ==== CONFIG ====
VIDEO_PATH = r"C:\Users\hp\OneDrive\Desktop\video_1_1080p_30fps.mp4"      # path to your video
UDP_IP = "127.0.0.1"          # receiver IP (or LAN IP)
UDP_PORT = 5000               # UDP port
GOP = 30  # match your receiver display rate
# =================

cmd = [
    "ffmpeg",
    "-re",
    "-stream_loop", "-1",                   # loop for continuous testing; remove if not needed
    "-hide_banner",
    "-loglevel", "info",
    "-i", VIDEO_PATH,
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-profile:v", "baseline",
    "-g", str(GOP),
    "-x264-params", "scenecut=0:open_gop=0:repeat-headers=1",
    "-vf", "format=yuv420p",
    "-f", "mpegts",
    "-muxdelay", "0",
    "-muxpreload", "0",
    "-flush_packets", "1",
    f"udp://{UDP_IP}:{UDP_PORT}?pkt_size=1316"
]

print(f"Streaming {VIDEO_PATH} to udp://{UDP_IP}:{UDP_PORT} ...")
proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)

try:
    # Show ffmpeg progress lines so you can see it's running
    for line in proc.stderr:
        if "frame=" in line or "Opening" in line or "Input" in line:
            print(line.rstrip())
except KeyboardInterrupt:
    pass
finally:
    if proc.poll() is None:
        proc.terminate()
    proc.wait()
    print("Streaming stopped.")
