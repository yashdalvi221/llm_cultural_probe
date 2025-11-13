from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Type


def retry_with_backoff(
    exceptions: Type[BaseException] | tuple[Type[BaseException], ...],
    max_attempts: int = 5,
):
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(exceptions),
    )

