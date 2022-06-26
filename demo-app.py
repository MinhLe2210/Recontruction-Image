import cv2
import streamlit as st
import numpy as np
from PIL import Image


def inpainting(img_path):
    pil_image = Image.open(img_path)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256),
                     interpolation=cv2.INTER_NEAREST)

    height, width, _ = img.shape
    mask = img.copy()
    black_pixels = np.where(
        (mask[:, :, 0] == 0) &
        (mask[:, :, 1] == 0) &
        (mask[:, :, 2] == 0)
    )

    others = np.where(
        (mask[:, :, 0] != 0) |
        (mask[:, :, 1] != 0) |
        (mask[:, :, 2] != 0)
    )
    mask[others] = [0, 0, 0]
    # set those pixels to white
    mask[black_pixels] = [255, 255, 255]
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    output1 = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

    output2 = cv2.inpaint(img, mask, 5, cv2.INPAINT_NS)

    ns = output2
    return ns


def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters!")
    st.text("We use OpenCV and Streamlit for this demo")

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg', 'tiff'])
    if not image_file:
        return None

    pil_image = Image.open(image_file)
    cvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_image = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (256, 256),
                                interpolation=cv2.INTER_NEAREST)
    processed_image = inpainting(image_file)

    st.text("Original Image vs Processed Image")
    st.image([original_image, processed_image])


if __name__ == '__main__':
    main_loop()
