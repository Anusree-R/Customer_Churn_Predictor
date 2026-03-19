import sys

def error_message_detail(error, error_detail: sys):
    # exc_tb gives us the traceback object — contains file name and line number
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        f"Error occurred in script: [{file_name}] "
        f"at line number: [{line_number}] "
        f"with message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        # Build the detailed message when exception is raised
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message