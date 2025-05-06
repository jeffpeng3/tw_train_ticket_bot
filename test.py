from captcha_handler import _process_audio_to_text
from time import time

time_start = time()
print(_process_audio_to_text("audio.mp3"))
time_end = time()

print("Time taken:", time_end - time_start)