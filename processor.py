import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = Image.open(image_path).convert("RGBA")
        self.image_array = np.array(self.original_image)
        self.modified_image = self.image_array.copy()

    def _show_before_after(self, title):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(self.image_array)
        axs[0].set_title("Original")
        axs[0].axis("off")
        axs[1].imshow(self.modified_image)
        axs[1].set_title(title)
        axs[1].axis("off")
        plt.show()
    def reset_image(self):
        self.modified_image = self.image_array.copy()
    def invert_colors(self):
        self.reset_image()
        self.modified_image[:,:,0:3] = 255 - self.modified_image[:,:,0:3]
        self._show_before_after("Inverted Colors")

    def to_grayscale(self):
        gray = np.mean(self.modified_image[:, :, :3], axis=2, keepdims=True)
        self.modified_image[:, :, :3] = np.repeat(gray, 3, axis=2)
        self._show_before_after("Grayscale")

    def rotate_90_clockwise(self):
        self.reset_image()
        h, w, c = self.modified_image.shape
        rotated = np.zeros((w, h, c), dtype=self.modified_image.dtype)

        for i in range(h):
            for j in range(w):
                rotated[j, h - 1 - i] = self.modified_image[i, j]

        self.modified_image = rotated
        self._show_before_after("Rotated 90 Clockwise")

    def mirror_horizontal(self):
        self.reset_image()
        self.modified_image = self.modified_image[:, ::-1, :]
        self._show_before_after("Mirrored Horizontally")
    
    def change_transparency(self, alpha_fraction):
        self.reset_image()
        self.modified_image[:, :, :3] = (self.modified_image[:, :, :3] * alpha_fraction).astype(np.uint8)
        self._show_before_after("Semi-Transparent")
    def brighten(self, factor):
        self.reset_image()
        rgb = self.modified_image[:, :, :3]
        rgb = np.clip(rgb * factor, 0, 255)
        self.modified_image[:, :, :3] = rgb.astype(np.uint8)
    def add_black_border(self, border_size=20):
        self.reset_image()
        h, w, r = self.modified_image.shape
        new_h, new_w = h + 2 * border_size, w + 2 * border_size
        new_image = np.zeros((new_h, new_w, r), dtype=np.uint8)
        new_image[:, :, 3] = 255  # full opacity
        new_image[border_size:border_size + h, border_size:border_size + w] = self.modified_image
        self.modified_image = new_image
        self._show_before_after("Black Border Added")
    def change_background_to_green(self):
        self.reset_image()
        transparent_mask = self.modified_image[:, :, 3] == 0
        new_img = self.modified_image.copy()
        new_img[transparent_mask] = [0, 255, 0, 255] #sarqum em lriv kanach u pictury opacityn minchev verj
        self.modified_image = new_img
        self._show_before_after("Green Background")
    def apply_sepia(self):
        self.reset_image()
        img = self.modified_image.astype(np.float32)
        R = img[: ,: , 0]
        G = img[: ,: , 1]
        B = img[: ,: , 2]
        img[: ,: , 0] = np.clip(0.393 * R + 0.769 * G + 0.189 * B, 0, 255)
        img[: ,: , 1] = np.clip(0.349 * R + 0.686 * G + 0.168 * B, 0, 255)
        img[: ,: , 2] = np.clip(0.272 * R + 0.534 * G + 0.131 * B, 0, 255)
        self.modified_image = img.astype(np.uint8)
        self._show_before_after("Sepia Effect")

processor = ImageProcessor("logo.png")
processor.invert_colors()
processor.to_grayscale()
processor.rotate_90_clockwise()
processor.mirror_horizontal()
processor.change_transparency(0.5)
processor.brighten(1.5)
processor.add_black_border()
processor.change_background_to_green()
processor.apply_sepia()
    




    

    
