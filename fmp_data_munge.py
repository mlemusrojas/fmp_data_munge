import os,sys
import argparse
import csv
import pandas as pd
import logging

##Set up logging

# Get log level from environment variable, default to INFO
log_level = os.getenv('LOG_LEVEL', 'INFO')
numeric_level = getattr(logging, log_level.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {log_level}')

# Set up logging
logging.basicConfig(level=numeric_level)

# Log to a file
log_file = '../fmp_data_munge.log'
file_handler = logging.FileHandler(log_file)


def read_csv(file_path: str) -> list[list[str]]:
    """
    Read a CSV file and return the data as a list of lists

    Args:
        file_path (str): The path to the CSV file

    Returns:
        list[list[str]]: The data from the CSV file
    """

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        return data
    
def make_df(data: list[list[str]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from a list of lists
    
    Args:
        data (list[list[str]]): The data to be converted to a DataFrame,
            in the format created by the read_csv function
        
    Returns:
        pd.DataFrame: The pandas DataFrame
    """

    df = pd.DataFrame(data[1:], columns=data[0])
    print(df.info())
    return df

def create_lc_name(name: str | None, date: str | None, role: str | None, uri: str | None) -> str:
    """
    Create a Library of Congress name from the name, date, role, and URI
    
    Args:
        name (str): The name of the person
        date (str): The date of the person
        role (str): The role of the person
        uri (str): The URI of the person
        
    Returns:
        str: The Library of Congress name
    """

    role_uri_merge = ' '.join([ i for i in [role, uri] if i])
    return ', '.join([i for i in [name, date, role_uri_merge] if i])

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
    """
    Create a Library of Congress field from piped fields

    Args:
        func: The function to apply to the piped fields
        *args: The piped fields to process

    Returns:
        str: The Library of Congress field
    """
    split_fields = [arg.split('|') for arg in args]
    
    lc_values = []
    for values in zip(*split_fields):
        lc_values.append(func(*values))
    
    concatenated_lc_values = '|'.join(lc_values)
    print(concatenated_lc_values)
    return concatenated_lc_values

def create_lc_date(start_date: str | None, end_date: str | None) -> str | None:
    """
    Create a Library of Congress date from the start and end dates

    Args:
        start_date (str): The start date
        end_date (str): The end date
    
    Returns:
        str: The Library of Congress date
    """

    return ' - '.join([i for i in [start_date, end_date] if i])
    
def build_uri(authority: str, id: str) -> str:
    """
    Build a URI from an authority and an ID. The authority can be 'lc' or 'viaf'.

    Args:
        authority (str): The authority
        id (str): The ID

    Returns:
        str: The URI
    """

    auth_dict = {
        'lc': 'http://id.loc.gov/authorities/names/',
        'viaf': 'http://viaf.org/viaf/'
    }
    return f'{auth_dict[authority.lower()]}{id}'

# def validate_uri(uri: str, authority: str) -> bool:
#     auth_dict = {
#         'lc': 'http://id.loc.gov/authorities/',
#         'viaf': 'http://viaf.org/viaf/'
#     }

#     if uri.startswith(auth_dict[authority.lower()]):
#         return True
#     else:
#         return False
    
# def return_valid_uri(uri: str, authority: str) -> str:
#     if validate_uri(uri, authority):
#         return uri
#     else:
#         raise ValueError(f'Invalid URI: {uri}')

def process_row(row: pd.Series, 
                name_col: str, 
                role_col: str, 
                authority_col: str, 
                authority_id_col: str, 
                new_column_name: str,
                start_date_col: str | None = None, 
                end_date_col: str | None = None
                ) -> pd.Series:
    """
    Process a row of a DataFrame to create a new column with a Library of Congress name
    
    Args:
        row (pd.Series): The row to process
        name_col (str): The name of the column containing the name
        role_col (str): The name of the column containing the role
        authority_col (str): The name of the column containing the authority
        authority_id_col (str): The name of the column containing the authority ID
        new_column_name (str): The name of the new column to create
        start_date_col (str): The name of the column containing the start date
            start and end date columns are optional, but if one is provided, 
            both must be provided
        end_date_col (str): The name of the column containing the end date
        
    Returns:
        pd.Series: The processed row
    """

    if start_date_col and end_date_col:
        lc_date = create_lc_from_piped_fields(create_lc_date, row[start_date_col], row[end_date_col])
    else:
        lc_date = None
    valid_uri = create_lc_from_piped_fields(build_uri, row[authority_col], row[authority_id_col])
    row[new_column_name] = create_lc_from_piped_fields(create_lc_name, row[name_col], lc_date, row[role_col], valid_uri)
    return row


def add_lc_name_column(df: pd.DataFrame, 
                       name_col: str, 
                       role_col: str, 
                       authority_col: str, 
                       authority_id_col: str, 
                       new_column_name: str
                       ) -> pd.DataFrame:
    """
    Add a new column to a DataFrame with a Library of Congress name

    Args:
        df (pd.DataFrame): The DataFrame to process
        name_col (str): The name of the column containing the name
        role_col (str): The name of the column containing the role
        authority_col (str): The name of the column containing the authority
        authority_id_col (str): The name of the column containing the authority ID
        new_column_name (str): The name of the new column to create

    Returns:
        pd.DataFrame: The DataFrame with the new column added
    """

    new_df = df.apply(process_row, args=(name_col, role_col, authority_col, authority_id_col, new_column_name), axis=1)
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
    new_df = add_lc_name_column(df, name_col='Authoritized Name', authority_id_col='Authority ID', authority_col='Authority Used', role_col='Position', new_column_name='namePersonOtherVIAF')
    print(new_df.head())

if __name__ == '__main__':
    main()