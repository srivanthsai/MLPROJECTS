import sys
import logging
import os
def custom_error_message(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() #this gives the line no and on which file
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_msg=f"error occured in python script called {file_name}, line number: {exc_tb.tb_lineno}. error message: {str(error)}"
    return error_msg

class customException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=custom_error_message(error_message,error_detail=error_detail)
    def __str__(self):
        return self.error_message
    
