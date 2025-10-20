# video_receiver_auto.py
import subprocess, time, numpy as np, cv2, sys

UDP_PORT = 5000
WIDTH, HEIGHT = 1920, 1080  # set to your sender's resolution
FRAME_SIZE = WIDTH * HEIGHT * 3

FFMPEG_CMD = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel", "warning",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-analyzeduration", "0",
    "-probesize", "32",
    "-i", f"udp://0.0.0.0:{UDP_PORT}?fifo_size=2000000&overrun_nonfatal=1",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "pipe:1"
]

print(f"Listening for UDP H.264 stream on port {UDP_PORT} (expecting {WIDTH}x{HEIGHT})...")
proc = subprocess.Popen(FFMPEG_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

last_bytes_time = time.time()
try:
    while True:
        if proc.poll() is not None:
            # FFmpeg exited; dump last errors
            err = proc.stderr.read().decode(errors="replace")
            print("FFmpeg exited:\n" + err)
            break

        raw = proc.stdout.read(FRAME_SIZE)
        if not raw or len(raw) < FRAME_SIZE:
            # no data yet
            if time.time() - last_bytes_time > 5:
                print("Waiting for packets on UDP 5000...")
                last_bytes_time = time.time()
            time.sleep(0.01)
            continue

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        cv2.imshow("Live H.264 Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    try: proc.terminate()
    except: pass
    cv2.destroyAllWindows()
    print("Receiver stopped.")
