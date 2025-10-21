# make_icon.py
from PIL import Image
src = Image.open(r"D:\Projects\Crypto\assets\app_icon.png").convert("RGBA")
sizes = [(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)]
src.save(r"D:\Projects\Crypto\assets\app_icon.ico", sizes=sizes)
print("ICO written ✅")
