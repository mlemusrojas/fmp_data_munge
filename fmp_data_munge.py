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

def create_lc_name(name: str, date: str | None, role: str, uri: str) -> str:
    if date:
        return f'{name}, {date}, {role} {uri}'
    else:
        return f'{name}, {role} {uri}'

# def create_lc_name_from_piped_fields(names: str, dates: str, roles: str, uris: str) -> str:
#     names_list: list = names.split('|')
#     dates_list: list = dates.split('|')
#     roles_list: list = roles.split('|')
#     uris_list: list = uris.split('|')
    
#     lc_names = []
#     for name, date, role, uri in zip(names_list, dates_list, roles_list, uris_list):
#         lc_names.append(create_lc_name(name, date, role, uri))
    
#     concatenated_lc_names = '|'.join(lc_names)
#     print(concatenated_lc_names)
#     return concatenated_lc_names

# def create_lc_date_from_piped_fields(start_dates: str, end_dates: str) -> str:
#     start_dates_list: list = start_dates.split('|')
#     end_dates_list: list = end_dates.split('|')
    
#     lc_dates = []
#     for start_date, end_date in zip(start_dates_list, end_dates_list):
#         lc_dates.append(create_lc_date(start_date, end_date))
    
#     concatenated_lc_dates = '|'.join(lc_dates)
#     print(concatenated_lc_dates)
#     return concatenated_lc_dates

def create_lc_from_piped_fields(func, *args) -> str:
    split_fields = [arg.split('|') for arg in args]
    
    lc_values = []
    for values in zip(*split_fields):
        lc_values.append(func(*values))
    
    concatenated_lc_values = '|'.join(lc_values)
    print(concatenated_lc_values)
    return concatenated_lc_values

def create_lc_date(start_date: str | None, end_date: str | None) -> str | None:
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
    
def return_valid_uri(uri: str, authority: str) -> str:
    if validate_uri(uri, authority):
        return uri
    else:
        raise ValueError(f'Invalid URI: {uri}')

def process_row(row: pd.Series, name_col: str, start_date_col: str, end_date_col: str, role_col: str, uri_col: str, authority_col: str, new_column_name: str) -> pd.Series:
    lc_date = create_lc_from_piped_fields(create_lc_date, row[start_date_col], row[end_date_col])
    valid_uri = create_lc_from_piped_fields(return_valid_uri, row[uri_col], row[authority_col])
    row[new_column_name] = create_lc_from_piped_fields(create_lc_name, row[name_col], lc_date, row[role_col], valid_uri)
    return row


def add_lc_name_column(df: pd.DataFrame, 
                       name_col: str, 
                       start_date_col: str, 
                       end_date_col: str, 
                       role_col: str, 
                       uri_col: str, 
                       authority_col: str, 
                       new_column_name: str
                       ) -> pd.DataFrame:
    new_df = df.apply(process_row, args=(name_col, start_date_col, end_date_col, role_col, uri_col, authority_col, new_column_name), axis=1)
    return new_df
    
        


def main():
    # Process command line arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Path to the CSV file to be read')
    args = parser.parse_args()

    # Read the CSV file
    data = read_csv(args.file_path)
    df = make_df(data)


    # # Create some test data
    # names = 'Smith, John|Jones, Mary|Brown, David'
    # dates = '1970|1980|1990'
    # roles = 'author|illustrator|editor'
    # uris = 'http://id.loc.gov/authorities/names/n79021383|http://id.loc.gov/authorities/names/n79021384|http://id.loc.gov/authorities/names/n79021385'
    # create_lc_name_from_piped_fields(names, dates, roles, uris)
    new_df = add_lc_name_column(df, 'Authoritized Name', 'Start Date', 'End Date', 'role', 'uri', 'authority', 'lc_name')
    print(new_df.head())

if __name__ == '__main__':
    main()