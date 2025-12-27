# ========== SENDER: H.265 TCP + CONGESTION CONTROL (OVERRIDE MODEL) ==========

import av
import socket
import struct
import time
import threading
import tkinter as tk
from fractions import Fraction

# ---------------- CONFIG ----------------
VIDEO_PATH = r"C:\Users\hp\Desktop\robo_arm_1920x1080_25fps.mp4"
SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000

PRESET = "ultrafast"
GOP = "30"

CRF_STEP = 5
MIN_CRF = 18
MAX_CRF = 45

SEND_DELAY_THRESHOLD = 0.03   # seconds
CONGESTION_FRAMES = 10
STABLE_WINDOW = 60            # frames

# Simulated bandwidth (bytes/sec)
TARGET_BW = {"value": 2_000_000}

# User desired CRF (GUI)
USER_CRF = {"value": 28}
# --------------------------------------


# ================= GUI =================
def crf_gui():
    root = tk.Tk()
    root.title("Desired CRF (User Preference)")

    slider = tk.Scale(
        root,
        from_=MIN_CRF,
        to=MAX_CRF,
        orient="horizontal",
        length=350,
        label="Desired CRF (Baseline Quality)"
    )
    slider.set(USER_CRF["value"])
    slider.pack(padx=10, pady=10)

    def poll():
        USER_CRF["value"] = slider.get()
        root.after(200, poll)

    poll()
    root.mainloop()

threading.Thread(target=crf_gui, daemon=True).start()


# ========== Bandwidth Fluctuation (Congestion Simulation) ==========
def bandwidth_fluctuator():
    while True:
        TARGET_BW["value"] = 400_000
        print(" [NET] Bandwidth DROP")
        time.sleep(8)

        TARGET_BW["value"] = 2_000_000
        print(" [NET] Bandwidth RECOVER")
        time.sleep(12)

threading.Thread(target=bandwidth_fluctuator, daemon=True).start()


# ================= Socket =================
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((SERVER_IP, SERVER_PORT))
sock.listen(1)
print(" Waiting for receiver...")
conn, _ = sock.accept()
print(" Connected")


# ================= Video =================
container = av.open(VIDEO_PATH)
vs = container.streams.video[0]
fps = float(vs.average_rate)
time_base = Fraction(1, int(round(fps)))


# ================= Encoder Factory =================
def create_encoder(crf):
    enc = av.codec.CodecContext.create("libx265", "w")
    enc.width = vs.width
    enc.height = vs.height
    enc.pix_fmt = "yuv420p"
    enc.time_base = time_base
    enc.options = {
        "preset": PRESET,
        "crf": str(crf),
        "x265-params": f"repeat-headers=1:keyint={GOP}:scenecut=0"
    }
    enc.open()
    print(f" [ENC] Encoder started @ CRF={crf}")
    return enc


# ================= Throttled Send =================
def throttled_send(data):
    start = time.time()
    conn.sendall(struct.pack("!I", len(data)))
    conn.sendall(data)

    elapsed = time.time() - start
    expected = len(data) / TARGET_BW["value"]

    if elapsed < expected:
        time.sleep(expected - elapsed)

    return elapsed


# ================= Streaming =================
ACTIVE_CRF = USER_CRF["value"]
encoder = create_encoder(ACTIVE_CRF)

congested = 0
stable = 0

for frame in container.decode(video=0):

    frame = frame.reformat(vs.width, vs.height, "yuv420p")
    packets = encoder.encode(frame)

    for pkt in packets:
        delay = throttled_send(bytes(pkt))

        if delay > SEND_DELAY_THRESHOLD:
            congested += 1
            stable = 0
        else:
            stable += 1
            congested = 0

    user_target = USER_CRF["value"]
    new_crf = ACTIVE_CRF

    # ---------- Congestion: OVERRIDE user CRF ----------
    if congested >= CONGESTION_FRAMES:
        new_crf = min(ACTIVE_CRF + CRF_STEP, MAX_CRF)
        congested = 0

    # ---------- Stability: return TOWARD user CRF ----------
    elif stable >= STABLE_WINDOW and ACTIVE_CRF > user_target:
        new_crf = max(ACTIVE_CRF - CRF_STEP, user_target)
        stable = 0

    # ---------- Apply change ----------
    if new_crf != ACTIVE_CRF:
        print(f" [CC] ACTIVE_CRF {ACTIVE_CRF} → {new_crf} | USER_CRF={user_target}")

        # Drain encoder
        for pkt in encoder.encode(None):
            throttled_send(bytes(pkt))

        del encoder
        time.sleep(0.05)

        encoder = create_encoder(new_crf)
        ACTIVE_CRF = new_crf


# ================= Cleanup =================
for pkt in encoder.encode(None):
    throttled_send(bytes(pkt))

conn.sendall(struct.pack("!I", 0))
conn.close()
sock.close()
container.close()

print(" Sender finished")
