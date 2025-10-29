from PIL import Image
import os
folder = "/home/acmu/Desktop/Video-Streaming-Pipeline/DCVC-RT/Video"
files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
for f in files[:5]:
    w,h = Image.open(os.path.join(folder,f)).size
    print(f, w, h)



from DCVC.src.utils.video_reader import PNGReader
from PIL import Image
import os

folder = "/home/gourav/Desktop/Video-Streaming-Pipeline/DCVC-RT/Video"
# print first PNG size (sanity)
files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
print('first files:', files[:5])
with Image.open(os.path.join(folder, files[0])) as im:
    print('PIL size:', im.size)

reader = PNGReader(folder, 1920, 1080)
frame = reader.read_one_frame()
if frame is None:
    print('reader returned None (no frames or naming mismatch)')
else:
    print('reader frame shape (C,H,W):', frame.shape)