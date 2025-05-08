from time import time
from collections import Counter
from captcha_handler import CaptchaResolver
from os import listdir
t_init_start: float = time()
captcha_solver: CaptchaResolver = CaptchaResolver()
t_init_end: float = time()
print(f"Time taken to initialize CaptchaResolver: {t_init_end - t_init_start:.4f} seconds")

NUM_SINGLE_TASK_RUNS: int = 1
AUDIO_FILE: str = "audio/JC8ARR.wav"

print(f"\nRunning a single captcha processing task ({NUM_SINGLE_TASK_RUNS} time(s)) to measure performance...")
single_task_results: list[str] = []
total_time_single_task_execution: float = 0.0


for AUDIO_FILE in listdir('audio'):
    print(f"\nProcessing {AUDIO_FILE} ...")
    time_single_task_start: float = time()
    with open(f"audio/{AUDIO_FILE}", "rb") as audio_file:
        result: str = captcha_solver._process_audio_to_text(audio_file, False)
    time_single_task_end: float = time()
    
    current_run_time: float = time_single_task_end - time_single_task_start
    total_time_single_task_execution += current_run_time
    single_task_results.append(result)
    
    print(f"Run {AUDIO_FILE} - Time taken: {current_run_time:.4f} seconds")
    print(f"Run {AUDIO_FILE} - Result: {result}")

average_time_single_task: float = total_time_single_task_execution / NUM_SINGLE_TASK_RUNS
print("\n--- Single Task Performance (with internal batching) ---")
print(f"Audio file: {AUDIO_FILE}")
print(f"Number of runs for measurement: {NUM_SINGLE_TASK_RUNS}")
print(f"Average time per single captcha processing: {average_time_single_task:.4f} seconds")
print(f"Results from runs: {Counter(single_task_results)}")


