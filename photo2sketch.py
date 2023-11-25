import streamlit as st
import cv2
from PIL import Image
import numpy as np
import io

def pencil_sketch(input_image, blur_value, scale_value, darkness_value):
    # Convert the file to an opencv image.
    img = np.array(input_image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_gray_image = 255 - gray_image

    # Blur the inverted grayscale image
    blurred_img = cv2.GaussianBlur(inverted_gray_image, (blur_value, blur_value), 10)

    # Invert the blurred image
    inverted_blurred_img = 255 - blurred_img

    # Create the pencil sketch image
    pencil_sketch_IMG = cv2.divide(gray_image, inverted_blurred_img, scale=scale_value)

    # Adjust overall darkness
    alpha = 1 - darkness_value / 100  # alpha range is 0-1
    beta = 0  # beta is added to every pixel
    pencil_sketch_IMG = cv2.convertScaleAbs(pencil_sketch_IMG, alpha=alpha, beta=beta)

    return pencil_sketch_IMG

def main():
    st.title("Pencil Sketch App")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Sliders for adjusting sketch parameters
    blur_value = st.slider("Blur Level", min_value=3, max_value=49, value=21, step=2)
    scale_value = st.slider("Sketch Scale", min_value=200, max_value=250, value=210)
    darkness_value = st.slider("Overall Picture Darkness", min_value=0, max_value=100, value=0)

    if uploaded_file is not None:
        # To read file as image:
        input_image = Image.open(uploaded_file)

        # Process the image
        result_image = pencil_sketch(input_image, blur_value, scale_value, darkness_value)

        # Convert the result image to PIL format
        result_image_pil = Image.fromarray(result_image)

        # Display the pencil sketch image
        st.image(result_image_pil, caption="Pencil Sketch", use_column_width=True)

        # Save the result image to a buffer
        buf = io.BytesIO()
        result_image_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        # Download button for the pencil sketch
        st.download_button(label="Download Pencil Sketch",
                           data=byte_im,
                           file_name="pencil_sketch.jpg",
                           mime="image/jpeg")

if __name__ == "__main__":
    main()

