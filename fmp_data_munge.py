import sys
import argparse
import csv
import pandas as pd


def read_csv(file_path) -> list[list[str]]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        return data
    
def make_df(data) -> pd.DataFrame:
    df = pd.DataFrame(data[1:], columns=data[0])
    print(df.info())
    return df

def create_lc_name(name: str, date: str, role: str, uri: str) -> str:
    return f'{name}, {date}, {role} {uri}'

def create_lc_name_from_piped_fields(names: str, dates: str, roles: str, uris: str) -> str:
    names_list: list = names.split('|')
    dates_list: list = dates.split('|')
    roles_list: list = roles.split('|')
    uris_list: list = uris.split('|')
    
    lc_names = []
    for name, date, role, uri in zip(names_list, dates_list, roles_list, uris_list):
        lc_names.append(create_lc_name(name, date, role, uri))
    
    concatenated_lc_names = '|'.join(lc_names)
    print(concatenated_lc_names)
    return concatenated_lc_names

def create_lc_date(start_date: str, end_date: str | None) -> str:
    if end_date:
        return f'{start_date}-{end_date}'
    else:
        return start_date
    
def validate_uri(uri: str, authority: str) -> bool:
    auth_dict = {
        'lc': 'http://id.loc.gov/authorities/',
        'viaf': 'http://viaf.org/viaf/'
    }

    if uri.startswith(auth_dict[authority.lower()]):
        return True
    else:
        return False



# def add_lc_name_column(df: pd.DataFrame, name_col: str, start_date_col: str, end_date_col: str, role_col: str, uri_col: str, authority_col: str, new_column_name: str) -> pd.DataFrame:
    
        


def main():
    # Process command line arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Path to the CSV file to be read')
    args = parser.parse_args()

    # Read the CSV file
    data = read_csv(args.file_path)
    df = make_df(data)


    # Create some test data
    names = 'Smith, John|Jones, Mary|Brown, David'
    dates = '1970|1980|1990'
    roles = 'author|illustrator|editor'
    uris = 'http://id.loc.gov/authorities/names/n79021383|http://id.loc.gov/authorities/names/n79021384|http://id.loc.gov/authorities/names/n79021385'
    create_lc_name_from_piped_fields(names, dates, roles, uris)

if __name__ == '__main__':
    main()