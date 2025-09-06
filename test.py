from PIL import Image

img = Image.open("/workspace/isaaclab/IsaacSimData/warehouse/results/frame000008.jpg")
width, height = img.size
print(f"Resolution: {width}x{height}")
