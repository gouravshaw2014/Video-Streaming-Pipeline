# dcvc_receiver.py
import socket
import io
import os
import torch
import numpy as np
import cv2
from DCVC.src.models.video_model import DMC
from DCVC.src.models.image_model import DMCI
from DCVC.src.utils.stream_helper import read_header, read_sps_remaining, read_ip_remaining, SPSHelper, NalType
from DCVC.src.utils.common import get_state_dict, set_torch_env, create_folder
from DCVC.src.utils.transforms import ycbcr2rgb

# ---- Configuration ----
SAVE_DECODED = "received_dcvc"
MODEL_PATH_I = "/home/gourav/Desktop/Video-Streaming-Pipeline/DCVC-RT/DCVC/checkpoints/cvpr2025_image.pth.tar"
MODEL_PATH_P = "/home/gourav/Desktop/Video-Streaming-Pipeline/DCVC-RT/DCVC/checkpoints/cvpr2025_video.pth.tar"
HOST = "127.0.0.1"
PORT = 5555
WIDTH, HEIGHT = 1920, 1080

# ---- Setup ----
create_folder(SAVE_DECODED, True)
set_torch_env()
device = "cuda" if torch.cuda.is_available() else "cpu"

i_frame_net = DMCI().to(device).eval().half()
i_frame_net.load_state_dict(get_state_dict(MODEL_PATH_I))
p_frame_net = DMC().to(device).eval().half()
p_frame_net.load_state_dict(get_state_dict(MODEL_PATH_P))

# Initialize internal components (entropy coder, encoders) so they are ready
# before any decode-time calls that use the entropy coder.
i_frame_net.update()
p_frame_net.update()

p_frame_net.clear_dpb()
p_frame_net.set_curr_poc(0)
sps_helper = SPSHelper()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print(f"Receiver listening on {HOST}:{PORT}")
conn, addr = sock.accept()
print(f"Connected to sender: {addr}")
# Determine whether a display is available for showing frames. In headless
# environments (no DISPLAY) we skip GUI calls to avoid Qt plugin errors.
GUI_AVAILABLE = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

frame_idx = 0
with torch.no_grad():
    while True:
        size_data = conn.recv(4)
        if not size_data:
            break
        size = int.from_bytes(size_data, 'big')
        packet = b''
        while len(packet) < size:
            packet += conn.recv(size - len(packet))

        input_buff = io.BytesIO(packet)
        header = read_header(input_buff)

        while header['nal_type'] == NalType.NAL_SPS:
            sps = read_sps_remaining(input_buff, header['sps_id'])
            sps_helper.add_sps_by_id(sps)
            header = read_header(input_buff)

        sps = sps_helper.get_sps_by_id(header['sps_id'])
        qp, bit_stream = read_ip_remaining(input_buff)

        if header['nal_type'] == NalType.NAL_I:
            decoded = i_frame_net.decompress(bit_stream, sps, qp)
            p_frame_net.add_ref_frame(None, decoded['x_hat'])
        else:
            decoded = p_frame_net.decompress(bit_stream, sps, qp)

        x_hat = decoded['x_hat'][:, :, :HEIGHT, :WIDTH]
        rgb = ycbcr2rgb(x_hat)
        # detach before converting to numpy to avoid gradients being tracked
        rgb_np = torch.clamp(rgb * 255, 0, 255).squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

        # Always save the decoded frame. Only show it if a display is present.
        cv2.imwrite(f"{SAVE_DECODED}/frame_{frame_idx:04d}.png", rgb_np)
        if GUI_AVAILABLE:
            try:
                cv2.imshow("DCVC Stream", rgb_np)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                # If imshow fails (missing Qt platform plugin), disable GUI and continue.
                GUI_AVAILABLE = False

        print(f"Received + Decoded frame {frame_idx}")
        frame_idx += 1

conn.close()
sock.close()
cv2.destroyAllWindows()
print("All frames received and saved.")
