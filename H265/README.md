# H.265 Video Streaming Pipeline

This directory contains a **TCP-based real-time video streaming pipeline** using **FFmpeg (H.265 / HEVC)** via the PyAV library. The pipeline follows a sender–receiver architecture where video frames are encoded on the sender side, transmitted over TCP, decoded on the receiver side, and played back in real time.

---

##  Overview

The H.265 pipeline implements the following flow:


- **Encoding / Decoding:** FFmpeg (H.265 / HEVC)
- **Transport Protocol:** TCP
- **Language:** Python

This implementation enables efficient compression compared to H.264, making it suitable for bandwidth-constrained streaming experiments.

---

##  Architecture


- Sender reads and encodes frames using H.265
- Encoded packets are sent reliably over TCP
- Receiver decodes packets and displays video frames

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

---

##  Encoding Parameters

The H.265 encoder supports multiple parameters to control **video quality**, **bitrate**, and **latency**. These parameters are commonly configured in the sender script.

###  CRF (Constant Rate Factor)

- **Range:** `0 – 51`
- **Lower value → higher quality → higher bitrate**
- **Higher value → lower quality → lower bitrate**

**Recommended values:**
- `18–23` : High quality
- `24–27` : Balanced quality (default range)
- `28–35` : Low bitrate / lower quality

---

###  Bitrate
Explicitly sets the target bitrate, it overrides CRF when needed

**Examples:**
```text
BITRATE = "2000K"
```

###  Preset (Speed–Quality Trade-off)

The **preset** controls the trade-off between **encoding speed**, **compression efficiency**, and **latency**.

| Preset      | Description |
|------------|-------------|
| ultrafast  | Lowest latency, fastest encoding, highest bitrate |
| superfast  | Very fast encoding with slightly better compression |
| veryfast   | Suitable for real-time streaming |
| medium     | Balanced speed and compression (default) |
| slow       | Best compression efficiency, very slow encoding |

**Notes:**
- Faster presets reduce CPU usage and latency but increase bitrate.
- Slower presets improve compression efficiency at the cost of higher latency.
- For real-time streaming, `ultrafast` or `veryfast` is recommended.

---
