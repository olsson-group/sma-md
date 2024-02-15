import csv

def load_parameters(input_csv_path : str) -> dict:
    """
    Loads model's parameters from CSV (in `input_csv_path`).
    """

    with open(input_csv_path, "r") as csv_file:

        params_reader = csv.reader(csv_file, delimiter=":")
        params_dict = {}

        for key, value in params_reader:
            params_dict[key] = value

    return params_dict