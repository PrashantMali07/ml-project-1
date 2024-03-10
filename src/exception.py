import sys
import logging
from src.logger import logging

def error_message_details(error, error_detail:sys):
    _, _, execution_tb = error_detail.exc_info()
    file_name = execution_tb.tb_frame.f_code.co_filename
    err_message = "Error occured in File_name [{0}], at line number [{1}] and error message: [{2}]".format(
        file_name, execution_tb.tb_lineno, str(error)
    )

    return err_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)


    def __str__(self) -> str:
        return self.error_message