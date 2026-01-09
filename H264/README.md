# H.264 Video Streaming Pipeline

This directory contains a **TCP-based real-time video streaming pipeline** using **FFmpeg (H.264)** via the PyAV library. The pipeline follows a sender–receiver architecture where video frames are encoded on the sender side, transmitted over TCP, decoded on the receiver side, and played back in real time.

---

##  Overview

The H.264 pipeline implements the following flow:


- **Encoding/Decoding:** FFmpeg (H.264)
- **Transport Protocol:** TCP
- **Language:** Python

---

##  Architecture 

Video File → Frame Reader → H.264 Encoder (FFmpeg) → TCP Socket → H.264 Decoder (FFmpeg) → Video Display


---


##  Requirements

### System Requirements

- Python 3.8+
- FFmpeg (must be installed and available in system PATH)

#### Install FFmpeg

**Ubuntu / Debian**
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Install Python Dependencies
```bash
pip install av opencv-python
```

