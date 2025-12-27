# ================== SENDER: H.265 (libx265) SAFE CRF SWITCH ==================

import av
import socket
import struct
import time
import threading
import tkinter as tk
from fractions import Fraction

VIDEO_PATH = r"C:\Users\hp\Desktop\robo_arm_1920x1080_25fps.mp4"
SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000
PRESET = "ultrafast"
GOP = "30"
INITIAL_CRF = 28

# ---------------- GUI ----------------
current_crf = {"value": INITIAL_CRF}

def gui():
    root = tk.Tk()
    root.title("Live CRF Control")
    s = tk.Scale(root, from_=0, to=51, orient="horizontal", length=300)
    s.set(INITIAL_CRF)
    s.pack(padx=10, pady=10)

    def poll():
        current_crf["value"] = s.get()
        root.after(200, poll)

    poll()
    root.mainloop()

threading.Thread(target=gui, daemon=True).start()

# ---------------- Socket ----------------
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((SERVER_IP, SERVER_PORT))
sock.listen(1)
print(" Waiting for receiver...")
conn, _ = sock.accept()
print(" Connected")

# ---------------- Video ----------------
container = av.open(VIDEO_PATH)
vstream = container.streams.video[0]
fps = float(vstream.average_rate)
time_base = Fraction(1, int(round(fps)))

def create_encoder(crf):
    enc = av.codec.CodecContext.create("libx265", "w")
    enc.width = vstream.width
    enc.height = vstream.height
    enc.pix_fmt = "yuv420p"
    enc.time_base = time_base
    enc.options = {
        "preset": PRESET,
        "crf": str(crf),
        "x265-params": f"repeat-headers=1:keyint={GOP}"
    }
    enc.open()
    print(f" Encoder started @ CRF={crf}")
    return enc

encoder = create_encoder(INITIAL_CRF)
last_crf = INITIAL_CRF

# ---------------- Stream ----------------
for frame in container.decode(video=0):

    # ---- CRF CHANGE HANDLING (SAFE) ----
    if current_crf["value"] != last_crf:
        print(" CRF changed → restarting encoder safely")

        # 1. Drain encoder fully
        for pkt in encoder.encode(None):
            data = bytes(pkt)
            conn.sendall(struct.pack("!I", len(data)))
            conn.sendall(data)

        # 2. Destroy encoder (NO close())
        del encoder
        # time.sleep(0.05)   # critical for x265 stability

        # 3. Create new encoder
        encoder = create_encoder(current_crf["value"])
        last_crf = current_crf["value"]


    # ---- Encode ----
    frame = frame.reformat(vstream.width, vstream.height, "yuv420p")
    packets = encoder.encode(frame)

    for pkt in packets:
        data = bytes(pkt)
        conn.sendall(struct.pack("!I", len(data)))
        conn.sendall(data)

# ---- Final flush ----
for pkt in encoder.encode(None):
    data = bytes(pkt)
    conn.sendall(struct.pack("!I", len(data)))
    conn.sendall(data)

conn.sendall(struct.pack("!I", 0))
conn.close()
sock.close()
container.close()
