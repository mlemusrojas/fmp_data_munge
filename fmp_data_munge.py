import sys
import argparse
import csv
import pandas as pd


def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        return data
    
def make_df(data):
    df = pd.DataFrame(data[1:], columns=data[0])
    print(df.head())
    return df

def main():
    # Process command line arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Path to the CSV file to be read')
    args = parser.parse_args()

    # Read the CSV file
    data = read_csv(args.file_path)
    df = make_df(data)

if __name__ == '__main__':
    main()