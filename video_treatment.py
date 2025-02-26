from moviepy import VideoFileClip

clip = VideoFileClip("test/OceanVideo.mp4")

# Adapted from MoviePy –Saving Video File Clip (2022).
clip.write_videofile("test/output.mp4")
