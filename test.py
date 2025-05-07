from time import time
from collections import Counter
from captcha_handler import CaptchaResolver

# 初始化 CaptchaResolver
t_init_start: float = time()
captcha_solver: CaptchaResolver = CaptchaResolver()
t_init_end: float = time()
print(f"Time taken to initialize CaptchaResolver: {t_init_end - t_init_start:.4f} seconds")

NUM_SINGLE_TASK_RUNS: int = 1 # We are interested in single task time with internal parallelism
AUDIO_FILE: str = "audio/JC8ARR.wav"

# --- 測試單個任務執行時間 (內部已並行化) ---
print(f"\nRunning a single captcha processing task ({NUM_SINGLE_TASK_RUNS} time(s)) to measure performance...")
single_task_results: list[str] = []
total_time_single_task_execution: float = 0.0

# Run it a few times to get a more stable average if needed, but for now, once is fine.
# For more rigorous benchmarking, consider more runs and averaging, or using timeit.
# However, for this specific request, one run demonstrates the internal parallelism.

for i in range(NUM_SINGLE_TASK_RUNS):
    time_single_task_start: float = time()
    with open(AUDIO_FILE, "rb") as audio_file:
        result: str = captcha_solver._process_audio_to_text(audio_file)
    time_single_task_end: float = time()
    
    current_run_time: float = time_single_task_end - time_single_task_start
    total_time_single_task_execution += current_run_time
    single_task_results.append(result)
    
    print(f"Run {i+1} - Time taken: {current_run_time:.4f} seconds")
    print(f"Run {i+1} - Result: {result}")

average_time_single_task: float = total_time_single_task_execution / NUM_SINGLE_TASK_RUNS
print("\n--- Single Task Performance (with internal batching) ---")
print(f"Audio file: {AUDIO_FILE}")
print(f"Number of runs for measurement: {NUM_SINGLE_TASK_RUNS}")
print(f"Average time per single captcha processing: {average_time_single_task:.4f} seconds")
print(f"Results from runs: {Counter(single_task_results)}")


# The external parallel test using ThreadPoolExecutor is removed as we are now
# focusing on the internal parallelism within _process_audio_to_text.
# Comparing the new single task time (with internal batching) against
# a version of the code *without* internal batching (i.e., the original sequential char-by-char processing)
# would be the correct way to measure the speedup of the internal parallelization.
# For this task, we are just reporting the new single task time.
