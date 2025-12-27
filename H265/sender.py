# ================== SENDER: H.265 (libx265) ==================
import av
import socket
import struct
import time
import os
from fractions import Fraction

# ==== CONFIG ====
VIDEO_PATH = r"C:\Users\hp\Desktop\robo_arm_1920x1080_25fps.mp4"
SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000
ENCODED_DIR = r"C:\Users\hp\OneDrive\Desktop\Video Streaming Pipeline\Encoded"
# HEVC Encoding Parameters (YOU CAN TUNE THESE)
CRF = "51"              # lower = better quality, 18-28 good range, range: 0-51
BITRATE = "2000k"       # override CRF if needed Examples: "500k", "2000k", "5M"
PRESET = "ultrafast"    # ultrafast(real-time but larger bitrate), superfast, veryfast, medium, slow(best compression but very slow)
GOP = "30"              # keyframe interval
# =================

os.makedirs(ENCODED_DIR, exist_ok=True)

# Setup socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((SERVER_IP, SERVER_PORT))
sock.listen(1)
print(f" Waiting for receiver on {SERVER_IP}:{SERVER_PORT} ...")
conn, addr = sock.accept()
print(f" Connected to receiver: {addr}")

# Open input video
container = av.open(VIDEO_PATH)
in_stream = container.streams.video[0]
width = in_stream.width
height = in_stream.height
fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0
frame_interval = 1.0 / fps
time_base = Fraction(1, int(round(fps)))

# HEVC encoder
encoder = av.codec.CodecContext.create('libx265', 'w')
encoder.width = width
encoder.height = height
encoder.pix_fmt = 'yuv420p'
encoder.time_base = time_base
encoder.options = {
    'preset': PRESET,  
    'crf': CRF,
    'x265-params': 'repeat-headers=1',
    # If you want CBR → include bitrate
    # 'b:v': BITRATE,
}
encoder.open()

print(f" Streaming in HEVC (H.265) @ {fps:.2f} FPS ({width}x{height})")

frame_counter = 0
start_time = time.time()

try:
    for frame in container.decode(video=0):
        frame_yuv = frame.reformat(width, height, format='yuv420p')

        packets = encoder.encode(frame_yuv)
        for pkt_idx, packet in enumerate(packets):
            data = bytes(packet)

            # Save packet (optional)
            # packet_path = os.path.join(ENCODED_DIR, f"packet_{frame_counter:06d}.bin")  #####
            # with open(packet_path, "wb") as f:                                          #####
            #     f.write(data)                                                           #####     

            # Send packet
            conn.sendall(struct.pack("!I", len(data)))
            conn.sendall(data)

        frame_counter += 1

        # FPS pacing
        expected_time = start_time + frame_counter * frame_interval
        now = time.time()
        delay = expected_time - now
        if delay > 0:
            time.sleep(delay)

    # Flush encoder
    for packet in encoder.encode(None):
        data = bytes(packet)                              # <-- convert to bytes
        conn.sendall(struct.pack("!I", len(data)))        # <-- use len(data)
        conn.sendall(data)

    # End of stream marker
    conn.sendall(struct.pack("!I", 0))

finally:
    conn.close()
    sock.close()
    container.close()
    print(f"\n Sender finished ({frame_counter} frames sent in {time.time()-start_time:.2f}s)")