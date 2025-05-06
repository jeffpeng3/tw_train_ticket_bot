from time import time
t1 = time()
from captcha_handler import CaptchaResolver
t2 = time()
captcha_solver = CaptchaResolver()
t3 = time()
print("Time taken to import and initialize:", t2 - t1, t3 - t2)

total_time = 0
for i in range(10):
    time_start = time()
    captcha_solver._process_audio_to_text("audio.mp3")
    time_end = time()
    total_time += (time_end - time_start)

print("Time taken:", total_time)
