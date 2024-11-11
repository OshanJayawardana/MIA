import pandas as pd
import requests
import argparse

def prepare_data(csv_file: str) -> pd.DataFrame:
    """Prepare the data for submission.

    Args:
        csv_file (str): The path to the csv file.

    Returns:
        pd.DataFrame: The data for submission.
    """
    df = pd.read_csv(csv_file)
    
    # only keep ids and score columns
    df = df[["ids", "score"]]

    return df

def submit_result(df: pd.DataFrame, ip: str, port: int, token: str, path: str="results/submission.csv"):
    """Submit the result.

    Args:
        df (pd.DataFrame): The data to submit.
        ip (str): The IP address of the server.
        port (int): The port of the server.
        token (str): The token for authentication.
        path (str): The path to save the result.
    """
    df.to_csv(path, index=None)
    url = f"http://{ip}:{port}/mia"
    response = requests.post(url, files={"file": open(path, "rb")}, headers={"token": token})
    print(response.json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit results to the server.")
    parser.add_argument("csv_file", type=str, help="The path to the csv file.")
    parser.add_argument("ip", type=str, help="The IP address of the server.")
    parser.add_argument("port", type=int, help="The port of the server.")
    parser.add_argument("token", type=str, help="The token for authentication.")
    args = parser.parse_args()

    df = prepare_data(args.csv_file)
    submit_result(df, args.ip, args.port, args.token)