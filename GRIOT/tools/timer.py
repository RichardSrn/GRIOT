import logging
import time

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def convert_time(seconds:float):
    """
    Convert seconds to days, hours, minutes, seconds and milliseconds.
    Return a string with the result.
    Forget the unnecessary units/
    :param seconds: float number of seconds
    :return: string with the result
    """
    if seconds < 1e-3 :
        result = "less than 0.001 s."
    else :
        days = int(seconds // (24 * 3600))
        seconds = seconds % (24 * 3600)
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        seconds = round(seconds, 2)
        milliseconds = int(seconds%1 * 1000)
        seconds = int(seconds)
        result = ""
        if days > 0:
            result += f"{days}d".rjust(4)
        if hours > 0 or result:
            result += f"{hours}h".rjust(4)
        if minutes > 0 or result:
            result += f"{minutes}min".rjust(6)
        if seconds > 0 or result:
            result += f"{seconds}s".rjust(4)
        if milliseconds > 0:
            result += f"{milliseconds}ms".rjust(6)
    return result

def print_time(start_time):
    # get current date and time with a nice printing format
    current = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    elapsed = convert_time(time.time()-start_time)
    logging.info(f"\n\n DONE. {current}. Elapsed : {elapsed} \n\n")

def timer(fct):

    def inner(*args, **kwargs):
        start_time = time.time()
        try:
            results = fct(*args, **kwargs)
            print_time(start_time)
        except Exception as e:
            print_time(start_time)
            # logging.exception(f"\n\n{e}\n\n")  # Log the exception message
            raise  # Re-raise the exception to display the error message

        return results
    return inner

