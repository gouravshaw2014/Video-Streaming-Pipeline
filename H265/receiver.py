# ================== RECEIVER: H.265 Decoding ==================
import av
import cv2
import socket
import struct
import os
import time

SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000
DECODED_DIR = r"C:\Users\hp\OneDrive\Desktop\Video Streaming Pipeline\Decoded"

os.makedirs(DECODED_DIR, exist_ok=True)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, SERVER_PORT))
print(" Connected to sender, receiving HEVC stream...")

decoder = av.codec.CodecContext.create('hevc', 'r')  # <-- HEVC
decoder.open()

frame_counter = 0
start_time = time.time()

def recvall(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

try:
    while True:
        raw_size = recvall(sock, 4)
        if not raw_size:
            break

        size = struct.unpack("!I", raw_size)[0]
        if size == 0:
            break

        data = recvall(sock, size)
        if not data:
            break

        packet = av.packet.Packet(data)
        decoded_frames = decoder.decode(packet)

        for df in decoded_frames:
            frame = df.to_ndarray(format="bgr24")

            # cv2.imwrite(os.path.join(DECODED_DIR, f"frame_{frame_counter:06d}.png"), frame) #####

            cv2.imshow("HEVC Decoded Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

            frame_counter += 1

except KeyboardInterrupt:
    pass

finally:
    cv2.destroyAllWindows()
    sock.close()
    elapsed = time.time() - start_time
    print(f" Receiver stopped after {frame_counter} frames in {elapsed:.2f}s")