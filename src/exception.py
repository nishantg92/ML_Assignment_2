import sys
from types import ModuleType


def error_message_detail(
    error: BaseException, error_detail: ModuleType | None = sys
) -> str:
    traceback = error.__traceback__
    if error_detail is not None:
        _, _, active_traceback = error_detail.exc_info()
        traceback = active_traceback or traceback

    if traceback is None:
        return f"{type(error).__name__}: {error}"

    file_name = traceback.tb_frame.f_code.co_filename
    line_number = traceback.tb_lineno
    return (
        f"Error occurred in script: [{file_name}] at line number: "
        f"[{line_number}] with error message: [{error}]"
    )


class CustomException(Exception):
    def __init__(
        self, message: str | BaseException, error_detail: ModuleType | None = sys
    ) -> None:
        if isinstance(message, BaseException):
            formatted_message = error_message_detail(message, error_detail)
        else:
            active_error = (
                error_detail.exc_info()[1] if error_detail is not None else None
            )
            if active_error is None:
                formatted_message = message
            else:
                formatted_message = (
                    f"{message}: {error_message_detail(active_error, error_detail)}"
                )

        super().__init__(formatted_message)

    def __repr__(self) -> str:
        return f"CustomException(message={str(self)!r})"