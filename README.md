# Video Streaming Pipeline

A modular video streaming pipeline for real-time video encoding, transmission, and decoding using both traditional and neural video codecs. The current implementation uses **TCP-based streaming** to ensure reliable packet delivery.

---

##  Overview

This repository implements an **end-to-end video streaming pipeline** consisting of a sender and a receiver. Video frames are encoded on the sender side, transmitted over a TCP connection, decoded on the receiver side, and finally played back.

The project is designed for experimentation and research involving:
- Classical video codecs using **FFmpeg**
- Neural video codecs such as **DCVC-RT**
- Network-aware and congestion-controlled streaming pipelines

---

##  Pipeline Architecture

### [Video Source] → [Encoder: H.264 / H.265 / DCVC-RT] → [TCP] → [Decoder] → [Playback]


###  Sender Side

- **Video Frames**  
  Frames are read from a video file or captured from a camera.
- **Encoder**  
  Encodes frames using one of the following:
  - FFmpeg-based **H.264**
  - FFmpeg-based **H.265**
  - Neural codec **DCVC-RT**
- **Send Encoded Packets**  
  Encoded data is packetized and transmitted over a **TCP socket**.

---

### Receiver Side


- **Receive Encoded Packets**  
  Encoded packets are received reliably using TCP.
- **Decoder**  
  Decodes the received bitstream using the corresponding codec.
- **Play Video Frames**  
  Decoded frames are rendered for real-time playback or further processing.

---

##  Repository Structure


Each codec directory contains its own sender and receiver scripts.

---

##  Supported Codecs

| Codec Type | Description |
|-----------|------------|
| **H.264** | FFmpeg-based AVC encoder/decoder |
| **H.265** | FFmpeg-based HEVC encoder/decoder |
| **DCVC-RT** | Neural real-time video codec |

---

##  Network Transport

- **Protocol:** TCP  
- **Reason:** Reliable delivery and ordered packet transmission  
- **Note:** TCP simplifies implementation but may introduce latency under congestion. Future extensions may include UDP or WebRTC-based streaming.

---

##  Requirements

General dependencies (may vary by codec):

- Python 3.x   
- Required Python packages (listed per folder)  
- CUDA-enabled GPU (recommended for DCVC-RT)

---




