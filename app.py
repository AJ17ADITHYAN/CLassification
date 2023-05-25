import streamlit as st
import tensorflow as tf
import keras
from PIL import Image, ImageOps
import numpy as np
import base64



modelpath = 'my_model2.hdf5'
model = keras.models.load_model(modelpath)




page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://cdn.pixabay.com/photo/2021/11/14/20/08/background-6795624_960_720.jpg");
background-size: 100%;
background-position: center;
-webkit-background-size: cover;
-moz-background-size: cover;
-o-background-size: cover;
background-size: cover;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
data_dir = "C:\Data"
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(180, 180),
  batch_size=batch_size)
class_names = train_ds.class_names


st.title("Lung Disease Detector")
    # Rest of your Streamlit app code





file = st.file_uploader("Please upload a lung scan file", type=["jpg", "png", "jpeg"])
st.set_option('deprecation.showfileUploaderEncoding', False)

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    image_np = np.array(image)
    dimension = len(image_np.shape)
    if dimension == 2:
        converted_image = np.expand_dims(image_np, axis=2)
    elif dimension == 4:
        converted_image = image_np[:, :, :3]
    else:
        converted_image = image_np

    resize = tf.image.resize(converted_image, (180, 180))
    
    if resize.shape[-1] == 4:
        resize = resize[:, :, :3]

    # Convert the image to RGB format before resizing
    if resize.shape[-1] == 1:
        resize = tf.image.grayscale_to_rgb(resize)

    resized_image = np.expand_dims(resize, 0)
    value = model.predict(np.expand_dims(resize/255 , 0))
    st.write(value)


    st.write(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(value)], 100 * np.max(value))
)


