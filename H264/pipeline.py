import av
import cv2
import os
import time
import numpy as np
from fractions import Fraction

# Input video path
video_path = r"C:\Users\hp\OneDrive\Desktop\video_1_1080p_30fps.mp4"

# Output folders
encoded_dir = r"C:\Users\hp\OneDrive\Desktop\Video Streaming Pipeline\Encoded"
decoded_dir = r'C:\Users\hp\OneDrive\Desktop\Video Streaming Pipeline\Decoded'

os.makedirs(encoded_dir, exist_ok=True)
os.makedirs(decoded_dir, exist_ok=True)

# Open input video
container = av.open(video_path)
in_stream = container.streams.video[0]
width = in_stream.width
height = in_stream.height
fps = float(in_stream.average_rate) if in_stream.average_rate else 30
time_base = Fraction(1, int(round(fps)))

# Create encoder (H.264/libx264)
encoder = av.codec.CodecContext.create('libx264', 'w')
encoder.width = width
encoder.height = height
encoder.pix_fmt = 'yuv420p'
encoder.time_base = time_base
encoder.options = {
    'preset': 'ultrafast',
    'tune': 'zerolatency',
    'profile': 'baseline',
    'x264-params': 'repeat-headers=1:scenecut=0:open_gop=0'
}
encoder.open()

# Decoder
decoder = av.codec.CodecContext.create('h264', 'r')
# Help the decoder with encoder headers (SPS/PPS)
if encoder.extradata:
    decoder.extradata = encoder.extradata
decoder.open()

print("Starting live-like streaming simulation...\n")

frame_counter = 0
try:
    for frame in container.decode(video=0):
        # Reformat to the encoder's expected pixel format
        frame_yuv = frame.reformat(width, height, format='yuv420p')

        # Encode may return 0..N packets
        packets = encoder.encode(frame_yuv)
        for pkt_idx, packet in enumerate(packets):
            # Save encoded packet
            packet_path = os.path.join(encoded_dir, f"packet_{frame_counter:06d}_{pkt_idx}.bin")
            with open(packet_path, "wb") as f:
                # PyAV < 12: Packet.to_bytes() may not exist
                try:
                    f.write(packet.to_bytes())
                except AttributeError:
                    f.write(bytes(packet))

            # "Receive" and decode the packet
            decoded_frames = decoder.decode(packet)
            for df in decoded_frames:
                decoded_img = df.to_ndarray(format="bgr24")

                # Save decoded frame (optional)
                cv2.imwrite(os.path.join(decoded_dir, f"frame_{frame_counter:06d}.png"), decoded_img)

                # Display decoded frame
                cv2.imshow("Decoded Stream", decoded_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

        frame_counter += 1
        time.sleep(1 / fps)  # simulate live pacing

    # Flush encoder
    for packet in encoder.encode(None):
        decoded_frames = decoder.decode(packet)
        for df in decoded_frames:
            decoded_img = df.to_ndarray(format="bgr24")
            cv2.imshow("Decoded Stream", decoded_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

    # Flush decoder
    for df in decoder.decode(None):
        decoded_img = df.to_ndarray(format="bgr24")
        cv2.imshow("Decoded Stream", decoded_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

finally:
    cv2.destroyAllWindows()
    container.close()

print("\nStreaming finished.")
