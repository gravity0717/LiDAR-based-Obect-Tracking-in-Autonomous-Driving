import logging

# 사용자 정의 포매터 클래스
class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.exc_info:
            record.exception_type = type(record.exc_info[1]).__name__
            record.exception_message = str(record.exc_info[1])
        else:
            record.exception_type = 'NoException'
            record.exception_message = ''
        return super().format(record)

def get_logger():
    # 로그 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)

    # 파일 핸들러 설정: 어디서 오류가 났는가
    file_handler = logging.FileHandler('app.log')
    file_formatter = CustomFormatter('%(asctime)s - %(levelname)s - %(exception_type)s - %(exception_message)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러 설정: 무슨 오류가 났는가 
    console_handler = logging.StreamHandler()
    console_formatter = CustomFormatter('%(asctime)s - %(levelname)s - %(exception_type)s - %(exception_message)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger 

def divide(a, b):
    return a / b

if __name__ == "__main__":
    logger = get_logger()
    try:
        result = divide(10, 0)
    except Exception as e:
        logger.error("예외 발생", exc_info=True)
