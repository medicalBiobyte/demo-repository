import time
import functools


def measure_time(label=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"{label or func.__name__} 시작...")
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"{label or func.__name__} 완료: {end - start:.2f}초")
            print("-" * 40)
            return result

        return wrapper

    return decorator
