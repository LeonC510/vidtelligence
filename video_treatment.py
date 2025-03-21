import os
import time

import matplotlib
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from moviepy import VideoFileClip, ImageClip, concatenate_videoclips

matplotlib.use('Agg')  # Use TkAgg backend

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor

def convert_image(raw_image):
    max_dim = 512
    img = raw_image
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Load conversion model.
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Load original image clip.
original_clip = VideoFileClip("./test/OceanVideo.mp4")

# Load style reference image.
style_reference_path = './test/TheStarryNight.jpg'
raw_image = tf.io.read_file(style_reference_path)
style_reference = convert_image(tf.image.decode_image(raw_image, channels=3))

# Create video clip to store the result
stylized_frames = list()

# Learned from Kaszanas (2019) for how to find the number of frames. Learned from MoviePy – Getting Duration of Video
# File Clip (2020) how to find the duration of the clip.
frame_duration = original_clip.duration / original_clip.n_frames

print("Starting stylization.")
start_time = time.time()
# Referenced Moviepy.Clip.Clip — MoviePy Documentation, Iter_frames() (n.d.) for how to iterate over frames.
for frame in original_clip.iter_frames():
    tensor_frame = tf.convert_to_tensor(frame, dtype=tf.uint8)
    original_image = convert_image(tensor_frame)
    stylized_image = hub_model(tf.constant(original_image), tf.constant(style_reference))[0]
    conversion_result = tensor_to_image(stylized_image)
    # Referenced Moviepy.Clip.Clip — MoviePy Documentation, With_duration() (n.d.) for use of with_duration().
    image_clip = ImageClip(conversion_result).with_duration(frame_duration)
    stylized_frames.append(image_clip)
end_time = time.time()
print(f"Finished stylization. Took {end_time - start_time} seconds for {original_clip.duration} seconds of video.")

print("Writing stylized video.")
# Inspired by Mr K. (2022) and Blanco (2017). Referenced
# Moviepy.Video.Compositing.CompositeVideoClip.Concatenate_videoclips — MoviePy Documentation (n.d.).
target_clip = concatenate_videoclips(stylized_frames, method="compose")
# Adapted from MoviePy –Saving Video File Clip (2022). Referenced Dey (2019).
target_clip.write_videofile("test/output.mp4", fps=original_clip.fps)
