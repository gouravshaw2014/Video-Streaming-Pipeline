import socket
import struct
import os
import time
import io
import torch
import cv2
import numpy as np

from DCVC.src.models.video_model import DMC
from DCVC.src.models.image_model import DMCI
from DCVC.src.utils.stream_helper import SPSHelper, NalType, read_header, read_sps_remaining, read_ip_remaining
from DCVC.src.utils.transforms import ycbcr2rgb

# ==== CONFIG === =
SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000
DECODED_DIR = "received_dcvc"
MODEL_PATH_I = "/home/gourav/Desktop/Video-Streaming-Pipeline/DCVC-RT/DCVC/checkpoints/cvpr2025_image.pth.tar"
MODEL_PATH_P = "/home/gourav/Desktop/Video-Streaming-Pipeline/DCVC-RT/DCVC/checkpoints/cvpr2025_video.pth.tar"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DISPLAY_FRAMES = True  # Set to False to disable OpenCV display
# ================

os.makedirs(DECODED_DIR, exist_ok=True)

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

def recvall(sock, n):
    """Receive exactly n bytes"""
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# Connect to sender
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, SERVER_PORT))
print("[RECEIVER] Connected to sender, waiting for stream...")

# Load models
print("[RECEIVER] Loading DCVC models...")
i_frame_net, p_frame_net = load_models()
print("[RECEIVER] Models loaded")


# Determine whether a display is available for showing frames. In headless
# environments (no DISPLAY) we skip GUI calls to avoid Qt plugin errors.
GUI_AVAILABLE = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


sps_helper = SPSHelper()
frame_counter = 0
start_time = time.time()

p_frame_net.set_curr_poc(0)

try:
    with torch.no_grad():
        while True:
            # Read frame size
            raw_size = recvall(sock, 4)
            if not raw_size:
                break
            size = struct.unpack("!I", raw_size)[0]
            if size == 0:
                print("[RECEIVER] End of stream signal received")
                break
            
            # Read frame data
            stream_data = recvall(sock, size)
            if not stream_data:
                break
            
            # Parse bitstream
            input_buff = io.BytesIO(stream_data)
            
            # Read header
            header = read_header(input_buff)
            
            # Handle SPS
            while header['nal_type'] == NalType.NAL_SPS:
                sps = read_sps_remaining(input_buff, header['sps_id'])
                sps_helper.add_sps_by_id(sps)
                header = read_header(input_buff)
            
            # Get SPS
            sps_id = header['sps_id']
            sps = sps_helper.get_sps_by_id(sps_id)
            pic_height = sps['height']
            pic_width = sps['width']
            
            # Read frame data
            qp, bit_stream = read_ip_remaining(input_buff)
            input_buff.close()
            
            # Decode frame
            frame_type = 'I'
            if header['nal_type'] == NalType.NAL_I:
                decoded = i_frame_net.decompress(bit_stream, sps, qp)
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(None, decoded['x_hat'])
            elif header['nal_type'] == NalType.NAL_P:
                frame_type = 'P'
                if sps['use_ada_i']:
                    p_frame_net.reset_ref_feature()
                decoded = p_frame_net.decompress(bit_stream, sps, qp)
            
            # Get reconstructed frame
            recon_frame = decoded['x_hat']
            x_hat = recon_frame[:, :, :pic_height, :pic_width]
            
            # Convert YCbCr to RGB
            rgb_rec = ycbcr2rgb(x_hat)
            rgb_rec = torch.clamp(rgb_rec * 255, 0, 255).round().to(dtype=torch.uint8)
            rgb_rec = rgb_rec.squeeze(0).permute(1, 2, 0).cpu().numpy()  # NCHW -> HWC
            
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(rgb_rec, cv2.COLOR_RGB2BGR)
            
            # Save frame
            output_path = os.path.join(DECODED_DIR, f"frame_{frame_counter:06d}_{frame_type}.png")
            cv2.imwrite(output_path, bgr_frame)
            
            # Display
            if DISPLAY_FRAMES:
                if GUI_AVAILABLE:
                    cv2.imshow("Decoded DCVC Stream", bgr_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
            
            frame_counter += 1
            
            if frame_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_counter / elapsed if elapsed > 0 else 0
                print(f"[RECEIVER] Frame {frame_counter} | {fps:.1f} FPS | {frame_type} | QP: {qp}")

except KeyboardInterrupt:
    print("\n[RECEIVER] Interrupted by user")
except Exception as e:
    print(f"\n[RECEIVER] Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    if DISPLAY_FRAMES:
        cv2.destroyAllWindows()
    sock.close()
    elapsed = time.time() - start_time
    fps = frame_counter / elapsed if elapsed > 0 else 0
    print(f"[RECEIVER] Stopped after {frame_counter} frames ({elapsed:.2f}s, {fps:.1f} FPS)")
    print(f"[RECEIVER] Decoded frames saved to: {DECODED_DIR}")
