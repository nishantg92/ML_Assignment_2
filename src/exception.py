import sys

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in script: [{file_name}] at line number: [{line_number}] with error message: [{error}]"

class CustomException(Exception):
    def __init__(self, message, error_detail: sys):
        self.message = message
        self.error_detail = error_detail
        super().__init__(self.message)

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return f"CustomException(message={self.message}, error_detail={self.error_detail})"