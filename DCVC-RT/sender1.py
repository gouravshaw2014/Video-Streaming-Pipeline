import socket
import struct
import time
import os
import io
import torch
from glob import glob
from PIL import Image
import numpy as np

from DCVC.src.models.video_model import DMC
from DCVC.src.models.image_model import DMCI
from DCVC.src.layers.cuda_inference import replicate_pad
from DCVC.src.utils.stream_helper import SPSHelper, write_sps, write_ip
from DCVC.src.utils.transforms import rgb2ycbcr

# ==== CONFIG ====
FRAMES_PATH = "/home/gourav/Desktop/video3_1080p"  # PNG frames path
SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000
MODEL_PATH_I = "/home/gourav/Desktop/Video-Streaming-Pipeline/DCVC-RT/DCVC/checkpoints/cvpr2025_image.pth.tar"
MODEL_PATH_P = "/home/gourav/Desktop/Video-Streaming-Pipeline/DCVC-RT/DCVC/checkpoints/cvpr2025_video.pth.tar"
QP_I = 37  # Quality parameter for I-frames
QP_P = 37  # Quality parameter for P-frames
INTRA_PERIOD = 32  # I-frame interval
RESET_INTERVAL = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ENCODED_DIR = "sent_dcvc"
# ================

os.makedirs(ENCODED_DIR, exist_ok=True)

def load_models():
    """Load DCVC I-frame and P-frame models"""
    def strip_module_prefix(state_dict):
        """Remove 'module.' prefix from state dict keys if present"""
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    i_frame_net = DMCI()
    i_checkpoint = torch.load(MODEL_PATH_I, map_location=DEVICE)
    i_state_dict = i_checkpoint['state_dict'] if 'state_dict' in i_checkpoint else i_checkpoint
    i_state_dict = strip_module_prefix(i_state_dict)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(DEVICE)
    i_frame_net.eval()
    i_frame_net.update(None)
    i_frame_net.half()
    
    p_frame_net = DMC()
    p_checkpoint = torch.load(MODEL_PATH_P, map_location=DEVICE)
    p_state_dict = p_checkpoint['state_dict'] if 'state_dict' in p_checkpoint else p_checkpoint
    p_state_dict = strip_module_prefix(p_state_dict)
    p_frame_net.load_state_dict(p_state_dict)
    p_frame_net = p_frame_net.to(DEVICE)
    p_frame_net.eval()
    p_frame_net.update(None)
    p_frame_net.half()
    
    return i_frame_net, p_frame_net

def load_png_frame(frame_path, device):
    """Load PNG frame and convert to tensor"""
    rgb = np.array(Image.open(frame_path))
    image = torch.from_numpy(rgb).to(device=device).to(dtype=torch.float32) / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
    x = rgb2ycbcr(image)
    x = x.to(torch.float16)
    return x, rgb.shape[0], rgb.shape[1]

# Setup socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((SERVER_IP, SERVER_PORT))
sock.listen(1)
print(f"[SENDER] Waiting for receiver on {SERVER_IP}:{SERVER_PORT} ...")
conn, addr = sock.accept()
print(f"[SENDER] Connected to receiver: {addr}")

# Load models
print("[SENDER] Loading DCVC models...")
i_frame_net, p_frame_net = load_models()
print("[SENDER] Models loaded")

# Get frame list
frame_paths = sorted(glob(os.path.join(FRAMES_PATH, "*.png")))
if not frame_paths:
    raise ValueError(f"No PNG frames found in {FRAMES_PATH}")

print(f"[SENDER] Found {len(frame_paths)} frames")

# Get dimensions from first frame
x_first, pic_height, pic_width = load_png_frame(frame_paths[0], DEVICE)
padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)

# Setup entropy coders
use_two_entropy_coders = pic_height * pic_width > 1280 * 720
i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

print(f"[SENDER] Streaming {len(frame_paths)} frames at {pic_width}x{pic_height}")

frame_counter = 0
start_time = time.time()
sps_helper = SPSHelper()
index_map = [0, 1, 0, 2, 0, 2, 0, 2]
total_bits = 0

p_frame_net.set_curr_poc(0)

try:
    with torch.no_grad():
        last_qp = 0
        for frame_idx, frame_path in enumerate(frame_paths):
            # Load frame
            x, _, _ = load_png_frame(frame_path, DEVICE)
            x_padded = replicate_pad(x, padding_b, padding_r)
            
            output_buff = io.BytesIO()
            is_i_frame = False
            
            # Determine frame type
            if frame_idx == 0 or (INTRA_PERIOD > 0 and frame_idx % INTRA_PERIOD == 0):
                is_i_frame = True
                curr_qp = QP_I
                
                # Create SPS for I-frame
                sps = {
                    'sps_id': -1,
                    'height': pic_height,
                    'width': pic_width,
                    'ec_part': 1 if use_two_entropy_coders else 0,
                    'use_ada_i': 0,
                }
                
                # Encode I-frame
                encoded = i_frame_net.compress(x_padded, QP_I)
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(None, encoded['x_hat'])
            else:
                # P-frame
                fa_idx = index_map[frame_idx % 8]
                if RESET_INTERVAL > 0 and frame_idx % RESET_INTERVAL == 1:
                    use_ada_i = 1
                    p_frame_net.prepare_feature_adaptor_i(last_qp)
                else:
                    use_ada_i = 0
                
                curr_qp = p_frame_net.shift_qp(QP_P, fa_idx)
                
                sps = {
                    'sps_id': -1,
                    'height': pic_height,
                    'width': pic_width,
                    'ec_part': 1 if use_two_entropy_coders else 0,
                    'use_ada_i': use_ada_i,
                }
                
                # Encode P-frame
                encoded = p_frame_net.compress(x_padded, curr_qp)
                last_qp = curr_qp
            
            # Write SPS if new
            sps_id, sps_new = sps_helper.get_sps_id(sps)
            sps['sps_id'] = sps_id
            if sps_new:
                write_sps(output_buff, sps)
            
            # Write frame data
            write_ip(output_buff, is_i_frame, sps_id, curr_qp, encoded['bit_stream'])
            
            # Get bitstream
            stream_data = output_buff.getvalue()
            output_buff.close()
            
            # Save bitstream to file
            frame_type = 'I' if is_i_frame else 'P'
            bitstream_path = os.path.join(ENCODED_DIR, f"frame_{frame_counter:06d}_{frame_type}.bin")
            with open(bitstream_path, 'wb') as f:
                f.write(stream_data)
            
            total_bits += len(stream_data) * 8
            
            # Send frame size + data
            conn.sendall(struct.pack("!I", len(stream_data)))
            conn.sendall(stream_data)
            
            frame_counter += 1
            
            if frame_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_counter / elapsed if elapsed > 0 else 0
                avg_bpp = total_bits / (frame_counter * pic_height * pic_width)
                print(f"[SENDER] Frame {frame_counter}/{len(frame_paths)} | {fps:.1f} FPS | {frame_type} | Avg BPP: {avg_bpp:.4f}")
    
    # End of stream signal
    conn.sendall(struct.pack("!I", 0))
    
finally:
    conn.close()
    sock.close()
    elapsed = time.time() - start_time
    fps = frame_counter / elapsed if elapsed > 0 else 0
    avg_bpp = total_bits / (frame_counter * pic_height * pic_width) if frame_counter > 0 else 0
    print(f"\n[SENDER] Finished - {frame_counter} frames sent in {elapsed:.2f}s")
    print(f"[SENDER] Average FPS: {fps:.1f}, Average BPP: {avg_bpp:.4f}")
    print(f"[SENDER] Bitstreams saved to: {ENCODED_DIR}")
