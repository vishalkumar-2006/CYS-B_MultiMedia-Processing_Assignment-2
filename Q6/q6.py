import cv2
import numpy as np

# --- Load images ---
low_path = "low_light.jpg"
bright_path = "bright_light.jpg"

low = cv2.imread(low_path, cv2.IMREAD_GRAYSCALE)
bright = cv2.imread(bright_path, cv2.IMREAD_GRAYSCALE)

if low is None or bright is None:
    print("Error: Could not load one of the images.")
    print("LOW =", low_path, "->", low is not None)
    print("BRIGHT =", bright_path, "->", bright is not None)
    exit()

# --- Compute bit-planes ---
def compute_bit_planes(img):
    bit_planes = []
    for i in range(8):
        plane = ((img >> i) & 1) * 255  # keep same logic
        bit_planes.append(plane.astype(np.uint8))
    return bit_planes

low_bits = compute_bit_planes(low)
bright_bits = compute_bit_planes(bright)

# --- Reconstruct using lowest 3 bit-planes (scaled to 0-255) ---
def reconstruct(bit_planes):
    rec = np.zeros_like(bit_planes[0], dtype=np.uint8)
    for i in range(3):  # lowest 3 bits
        bit = (bit_planes[i] // 255)  # same logic
        rec += bit * (2 ** i)

    rec_scaled = (rec * 255 // 7).astype(np.uint8)  # same logic
    return rec_scaled

low_reconstructed = reconstruct(low_bits)
bright_reconstructed = reconstruct(bright_bits)

# --- Resize to original size ---
low_reconstructed = cv2.resize(low_reconstructed, (low.shape[1], low.shape[0]))
bright_reconstructed = cv2.resize(bright_reconstructed, (bright.shape[1], bright.shape[0]))

# --- Compute differences ---
low_diff = cv2.absdiff(low, low_reconstructed)
bright_diff = cv2.absdiff(bright, bright_reconstructed)

# --- Save all images ---
cv2.imwrite("low_reconstructed.jpg", low_reconstructed)
cv2.imwrite("low_difference.jpg", low_diff)
cv2.imwrite("bright_reconstructed.jpg", bright_reconstructed)
cv2.imwrite("bright_difference.jpg", bright_diff)

print("All images saved: reconstructed & difference images.")
