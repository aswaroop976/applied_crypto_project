from PIL import Image, ImageDraw

# Input/output paths
input_path = "./jpg/image_00001.jpg"
output_path = "output_with_rectangle.jpg"

# Open the image
img = Image.open(input_path).convert("RGB")

# Create a drawing context
draw = ImageDraw.Draw(img)

# Define rectangle coordinates (x1, y1, x2, y2)
# This draws a 50x30 blue rectangle at position (20, 20)
x1, y1 = 250, 350
x2, y2 = x1 + 120, y1 + 60

# Draw the rectangle (fill only)
draw.rectangle([x1, y1, x2, y2], fill=(255, 65, 0))  # blue in RGB

# Save result
img.save(output_path)

print("Saved:", output_path)
