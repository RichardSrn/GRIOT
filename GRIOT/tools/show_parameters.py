def show_parameters(logging=None, **parameters):
    for parameter_name, parameter_value in parameters.items():
        if logging:
            logging.info(f"{parameter_name.rjust(20)}: {parameter_value}")
        else:
            print(f"{parameter_name.rjust(20)}: {parameter_value}")