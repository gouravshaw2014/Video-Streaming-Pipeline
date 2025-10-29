# dcvc_sender.py
import socket
import os
import io
import numpy as np
import torch
import time
import argparse
from DCVC.src.models.video_model import DMC
from DCVC.src.models.image_model import DMCI
from DCVC.src.utils.video_reader import PNGReader
from PIL import Image
from DCVC.src.utils.stream_helper import SPSHelper, write_sps, write_ip
from DCVC.src.utils.common import get_state_dict, set_torch_env, create_folder
from DCVC.src.layers.cuda_inference import replicate_pad

# ---- Configuration ----
FRAME_FOLDER = "/home/gourav/Desktop/Video-Streaming-Pipeline/DCVC-RT/Video"   # folder with PNG frames (input)
MODEL_PATH_I = "/home/gourav/Desktop/Video-Streaming-Pipeline/DCVC-RT/DCVC/checkpoints/cvpr2025_image.pth.tar"
MODEL_PATH_P = "/home/gourav/Desktop/Video-Streaming-Pipeline/DCVC-RT/DCVC/checkpoints/cvpr2025_video.pth.tar"
SAVE_FOLDER = "sent_dcvc"
HOST = "127.0.0.1"
PORT = 5555
FPS = 30  # to control sending rate
DEFAULT_QP = 21  # quantization parameter used for testing (can be changed)


# ---- Setup ----
create_folder(SAVE_FOLDER, True)
set_torch_env()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load DCVC models
i_frame_net = DMCI().to(device).eval().half()
i_frame_net.load_state_dict(get_state_dict(MODEL_PATH_I))
p_frame_net = DMC().to(device).eval().half()
p_frame_net.load_state_dict(get_state_dict(MODEL_PATH_P))

# Initialize internal structures (entropy coder, encoders) before using them
# update() creates the EntropyCoder and lets other components register with it.
i_frame_net.update()
p_frame_net.update()

# Setup reader and socket
def _detect_png_size(folder):
    pngs = [p for p in os.listdir(folder) if p.lower().endswith('.png')]
    if not pngs:
        raise ValueError(f"No PNG files found in {folder}")
    pngs.sort()
    first = os.path.join(folder, pngs[0])
    with Image.open(first) as im:
        w, h = im.size
    return w, h

parser = argparse.ArgumentParser(description='DCVC sender')
parser.add_argument('--qp', type=int, default=DEFAULT_QP, help='QP index to use (0..63)')
parser.add_argument('--fps', type=float, default=FPS, help='frames per second to send')
args = parser.parse_args()

DEFAULT_QP = args.qp
FPS = args.fps

detected_w, detected_h = _detect_png_size(FRAME_FOLDER)
reader = PNGReader(FRAME_FOLDER, detected_w, detected_h)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
print(f"Connected to receiver at {HOST}:{PORT}")

s_padding_r, s_padding_b = DMCI.get_padding_size(detected_h, detected_w, 16)
padding_r, padding_b = s_padding_r, s_padding_b
sps_helper = SPSHelper()
use_two_entropy_coders = True

p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
p_frame_net.set_curr_poc(0)
p_frame_net.clear_dpb()

frame_idx = 0
while True:
    rgb = reader.read_one_frame()
    if rgb is None:
        break

    # PNGReader returns arrays in shape (C, H, W). We rely on the model's padding
    # (replicate_pad) computed from detected dimensions rather than manually numpy-padding
    # here to avoid double-padding mismatches between reference and current frames.
    x = torch.from_numpy(rgb).unsqueeze(0).to(device).half() / 255.0
    x_padded = replicate_pad(x, padding_b, padding_r)

    is_i_frame = (frame_idx == 0)
    qp = DEFAULT_QP
    output_buff = io.BytesIO()

    if is_i_frame:
        encoded = i_frame_net.compress(x_padded, qp)
        p_frame_net.add_ref_frame(None, encoded['x_hat'])
        sps = {'sps_id': -1, 'height': detected_h, 'width': detected_w, 'ec_part': 1, 'use_ada_i': 0}
        sps_id, sps_new = sps_helper.get_sps_id(sps)
        sps['sps_id'] = sps_id
        if sps_new:
            write_sps(output_buff, sps)
        write_ip(output_buff, True, sps_id, qp, encoded['bit_stream'])
    else:
        encoded = p_frame_net.compress(x_padded, qp)
        write_ip(output_buff, False, 0, qp, encoded['bit_stream'])

    data = output_buff.getvalue()
    size = len(data)
    sock.sendall(size.to_bytes(4, 'big') + data)

    with open(f"{SAVE_FOLDER}/frame_{frame_idx:04d}.bin", "wb") as f:
        f.write(data)

    print(f"Sent frame {frame_idx} ({size} bytes)")
    frame_idx += 1
    time.sleep(1 / FPS)

sock.close()
reader.close()
print("All frames sent successfully.")
