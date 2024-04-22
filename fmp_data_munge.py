#region IMPORTS
import os, sys
import argparse
import csv
import logging
from typing import NamedTuple, Callable, Optional, Dict
from dataclasses import dataclass


import pandas as pd
from dotenv import load_dotenv, find_dotenv
#endregion

load_dotenv(find_dotenv())
LGLVL = os.environ['LOGLEVEL']

#region LOGGING
## set up logging ---------------------------------------------------
lglvldct = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARN': logging.WARNING
}
logging.basicConfig(
    level=lglvldct[LGLVL],  # type: ignore -- assigns the level-object to the level-key loaded from the envar
    format='[%(asctime)s] %(levelname)s [%(module)s-%(funcName)s()::%(lineno)d] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S',
    # encoding='utf-8',
    filename='../fmp_data_munge.log',
    filemode='w'  # Set filemode to 'w' to overwrite the existing log file
)
log = logging.getLogger(__name__)
log.info(f'\n\n`log` logging working, using level, ``{LGLVL}``')

ch = logging.StreamHandler()  # ch stands for `Console Handler`
ch.setLevel(logging.WARN)  # note: this level is _not_ the same as the file-handler level set in the `.env`
ch.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s [%(module)s-%(funcName)s()::%(lineno)d] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S',
))
log.addHandler(ch)
#endregion

#region CLASSES
# # Create a namedtuple Class to store the formatted output chunks
# class FormattedOutput(NamedTuple):
#     """
#     A named tuple 'FormattedOutput' is used to specify how to create a new column in the process_row function.

#     Attributes:
#         text (str): The static text to include in the new column. Default is None.
#         column_name (str): The name of an existing column whose values are to be included in the new column. Default is None.
#         function (function): A function that returns a string to be included in the new column. Default is None.
#         kwargs (dict): The keyword arguments to pass to the function. Default is None.

#     Any given attribute can be None, but if using a function, the kwargs must be provided.

#     Examples:
#         FormattedOutput can be used in the following ways:

#         ```
#         FormattedOutput(text=',', column_name=None, function=None, kwargs=None)
#         FormattedOutput(text=None, column_name='Authoritized Name', function=None, kwargs=None)
#         FormattedOutput(text=None, column_name=None, function=create_formatted_date, kwargs={'start_date': 'Start Date', 'end_date': 'End Date'})
#         ```
#     """
#     text: str | None
#     column_name: str | None
#     function: Callable | None
#     kwargs: dict[str, str] | None

@dataclass
class FormattedOutput:
    """
    A dataclass 'FormattedOutput' is used to specify how to create a new column in the process_row function.

    Attributes:
        text (str): The static text to include in the new column. Default is None.
        column_name (str): The name of an existing column whose values are to be included in the new column. Default is None.
        function (Callable): A function that returns a string to be included in the new column. Default is None.
        kwargs (Dict[str, str]): The keyword arguments to pass to the function. Default is None.

    Any given attribute can be None, but if using a function, the kwargs must be provided.

    Examples:
        FormattedOutput can be used in the following ways:

        ```
        FormattedOutput(text=',', column_name=None, function=None, kwargs=None)
        FormattedOutput(text=None, column_name='Authoritized Name', function=None, kwargs=None)
        FormattedOutput(text=None, column_name=None, function=create_formatted_date, kwargs={'start_date': 'Start Date', 'end_date': 'End Date'})
        ```
    """
    text: Optional[str] = None
    column_name: Optional[str] = None
    function: Optional[Callable] = None
    kwargs: Optional[Dict[str, str]] = None
#endregion

#region FUNCTIONS
# =============================================================================
# FUNCTIONS
# =============================================================================





def read_csv(file_path: str) -> list[list[str]]: # MARK: read_csv
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
        log.info(f'Read {len(data)} rows from {file_path}')
        return data
    
def make_df(data: list[list[str]]) -> pd.DataFrame: # MARK: make_df
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
    log.info(f'Created DataFrame with {len(df)} rows and {len(df.columns)} columns')
    return df

def create_authority_name(**fields) -> str: # MARK: create_authority_name
    """
    Create an 'authority' name from the name, date, role, and URI

    Example:
        input: 'Smith, John', '1970', 'author', 'http://id.loc.gov/authorities/names/n79021383'
        output: 'Smith, John, 1970, author http://id.loc.gov/authorities/names/n79021383'
    
    Args:
        name (str): The name of the person
        date (str): The date of the person
        role (str): The role of the person
        uri (str): The URI of the person
        
    Returns:
        str: The formatted name
    """
    log.debug(f'entering create_authority_name, ``{fields = }``')
    name = fields.get('name', None)
    date = fields.get('date', None)
    role = fields.get('role', None)
    uri = fields.get('uri', None)
    
    role_uri_merge = ' '.join([ i for i in [role, uri] if i])
    return ', '.join([i for i in [name, date, role_uri_merge] if i])


def create_authority_from_piped_fields(func, **kwargs) -> str | None: # MARK: create_authority_from_piped_fields
    """
    Create an 'authority' formatted field from piped fields

    Example: 
        input: 'Smith, John|Doe Jane', '1970|1980', 'author|illustrator', 'uri1|uri2'
        output: 'Smith, John, 1970, author uri1|Doe Jane, 1980, illustrator, uri2'
    
    Args:
        func: The function to apply to the piped fields
        *args: The piped fields to process

    Returns:
        str: The formatted field
    """

    log.debug(f'entering create_authority_from_piped_fields, ``{func = }, {kwargs = }``')
    split_fields = {k: v.split('|') for k, v in kwargs.items() if v is not None}
    
    log.debug(f'{split_fields = }')
    
    authority_values = []
    for values in zip(*split_fields.values()):
        log.debug(f'values, ``{values}``')
        authority_values.append(func(**dict(zip(split_fields.keys(), values))))
    
    log.debug(f'{authority_values = }')
    if not authority_values:
        log.debug(f'No Authority values created, returning None')
        return None
    if not authority_values[0]:
        log.debug(f'Authority value is empty string or ``[None]``, returning None')
        return None
    log.debug(f'Concatenating Authority values {authority_values = }')
    concatenated_authority_values = '|'.join([i if i else '' for i in authority_values])
    print(concatenated_authority_values)
    return concatenated_authority_values

def create_formatted_date(start_date: str | None, end_date: str | None) -> str | None: # MARK: create_formatted_date
    """
    Create a date range in 'YYYY - YYYY' format from a start date and an end date,
    or a single date if only one is provided

    Args:
        start_date (str): The start date
        end_date (str): The end date
    
    Returns:
        str: The formatted date (range)
    """

    return ' - '.join([i for i in [start_date, end_date] if i])
    
def build_uri(authority: str | None, id: str | None) -> str | None: # MARK: build_uri
    """
    Build a URI from an authority and an ID. The authority can be 'lc', 'viaf', or local. If local, returns None.

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

    if not authority:
        log.debug(f'No authority provided: {authority = }, {id = }')
        return None
    if authority.lower() == 'local':
        log.debug(f'Local authority provided: {authority = }, {id = }')
        return None
    uri = f'{auth_dict[authority.lower()]}{id}'
    log.debug(f'Created URI: {uri}')

    return uri

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

def reduce_list(values: str, flags: list[bool]) -> str: # MARK: reduce_list
    """
    Reduce a list of values based on a list of boolean flags

    Example:
        input: 'a|b|c', [True, False, True]
        output: 'a|c'
    
    Args:
        values str: The pipe-separated list of values
        flags (list[bool]): The flags to reduce by
        
    Returns:
        str: The reduced list of values
    """

    return '|'.join([value for value, flag in zip(values.split('|'), flags) if flag])

def process_row_alpha(row: pd.Series, 
                name_col: str, 
                role_col: str, 
                authority_col: str, 
                authority_id_col: str,
                authority: str, 
                new_column_name: str,
                start_date_col: str | None = None, 
                end_date_col: str | None = None
                ) -> pd.Series: # MARK: process_row_alpha
    """
    Process a row of a DataFrame to create a new column with a formatted name.
    
    Appropriate when the desired output matches:
    
      A:  ```{Name}, {Role} {Authority URI}``` or
      B:  ```{Name}, {Start Date} - {End Date}, {Role} {Authority URI}```
    
    Args:
        row (pd.Series): The row to process
        name_col (str): The name of the column containing the name
        role_col (str): The name of the column containing the role
        authority_col (str): The name of the column containing the authority
        authority_id_col (str): The name of the column containing the authority ID
        authority (str): The authority to filter on
        new_column_name (str): The name of the new column to create
        start_date_col (str): The name of the column containing the start date
            start and end date columns are optional, but if one is provided, 
            both must be provided
        end_date_col (str): The name of the column containing the end date
        
    Returns:
        pd.Series: The processed row
    """
    log.debug(f'entering process_row')
    log.debug(f'Processing row: {row}')

    # track the indices to process based on the authority
    log.debug(f'{authority = }')
    log.debug(f'{row[authority_col] = }')
    values_to_process: list[bool] = [True if i.lower() == authority.lower() else False for i in row[authority_col].split('|')]
    log.debug(f'{values_to_process = }')

    if start_date_col and end_date_col:
        kept_start_dates = reduce_list(row[start_date_col], values_to_process)
        log.debug(f'{kept_start_dates = }')
        kept_end_dates = reduce_list(row[end_date_col], values_to_process)
        log.debug(f'{kept_end_dates = }')
        formatted_dates = create_authority_from_piped_fields(create_formatted_date, start_date=kept_start_dates, end_date=kept_end_dates)
        log.debug(f'Created date with create_formatted_date: {formatted_dates}')
    else:
        formatted_dates = None
        log.debug(f'No date created, start_date_col and/or end_date_col not provided: {start_date_col = }, {end_date_col = }')

    kept_authorities = reduce_list(row[authority_col], values_to_process)
    log.debug(f'{kept_authorities = }')
    kept_ids = reduce_list(row[authority_id_col], values_to_process)
    log.debug(f'{kept_ids = }')
    valid_uris = create_authority_from_piped_fields(build_uri, authority=kept_authorities, id=kept_ids)
    log.debug(f'Created URI with build_uri: {valid_uris}')

    kept_names = reduce_list(row[name_col], values_to_process)
    log.debug(f'{kept_names = }')
    kept_roles = reduce_list(row[role_col], values_to_process)
    log.debug(f'{kept_roles = }')

    row[new_column_name] = create_authority_from_piped_fields(create_authority_name, name=kept_names, date=formatted_dates, role=kept_roles, uri=valid_uris)
    log.debug(f'Created name with create_authority_name: {row[new_column_name]}')
    log.debug(f'Processed row: {row}')
    log.debug(f'Exiting process_row')
    return row

def process_row_beta(row: pd.Series,
                new_column_name: str, 
                output_format: list[FormattedOutput],
                mask_column: str | None = None,
                mask_value: str | None = None
                ) -> pd.Series: # MARK: process_row_beta

    """
    Process a row of a DataFrame to create a new column with a format specified by the FormattedOutput namedtuple

    Args:
        row (pd.Series): The row to process
        output_format (list[FormattedOutput]): A list of FormattedOutput namedtuples specifying how to create the new column
        mask_column (str): The name of the column to use as a mask
        mask_value (str): The value to use as a mask filter, only values in mask_column matching this value will be processed (case-insensitive)

    Any given attribute can be None, but if using a function, the kwargs must be provided.
    If multiple attributes are provided, they will be concatenated in the order they are provided.

    Returns:
        pd.Series: The processed row

    Example:
        This is a partial example of output_format for the name and date range:
        ```
        output_format = [
            FormattedOutput(text=None, column_name='Authoritized Name', function=None, kwargs=None),
            FormattedOutput(text=', ', column_name=None, function=None, kwargs=None),
            FormattedOutput(text=None, column_name=None, function=create_formatted_date, kwargs={'start_date': 'Start Date', 'end_date': 'End Date'})
        ]
        ```

        This is an example of using the mask_column and mask_value arguments:
        ```
        new_df = df.apply(process_row, args=(output_format, 'Authority Used', 'viaf'), axis=1)
        ```
    """

    log.debug(f'entering process_row_beta')

    # check that mask_column and mask_value are both provided or both None
    if isinstance(mask_column, str) ^ isinstance(mask_value, str):
        raise ValueError('Both mask_column and mask_value must be provided')
    
    # track the indices to process based on the mask
    if mask_column and mask_value:
        log.debug(f'{mask_column = }, {mask_value = }')
        values_to_process = [True if i.lower() == mask_value.lower() else False for i in row[mask_column].split('|')]
        log.debug(f'{values_to_process = }')
    else:
        values_to_process = [True] * len(row)
        log.debug(f'No mask provided, processing all values: {values_to_process = }')

    formatted_output_values: list[str] = []

    for i, value in enumerate(values_to_process):
        if value:
            formatted_text: str = ''
            for chunk in output_format:
                # check that function and kwargs are both provided or both None
                if callable(chunk.function) ^ isinstance(chunk.kwargs, dict):
                    raise ValueError("FormattedOutput must specify both 'function' and 'kwargs' or neither")
                if chunk.text:
                    formatted_text += chunk.text
                if chunk.column_name:
                    formatted_text += row[chunk.column_name].split('|')[i]
                if chunk.function:
                    built_kwargs: dict = {}
                    for k, v in chunk.kwargs.items(): # type: ignore
                        built_kwargs[k] = row[v].split('|')[i]
                    formatted_text += chunk.function(**built_kwargs)
            formatted_output_values.append(formatted_text)

    row[new_column_name] = '|'.join(formatted_output_values)
    log.debug(f'Processed row: {row}')
    return row





def add_authority_name_column(df: pd.DataFrame, 
                       name_col: str, 
                       role_col: str, 
                       authority_col: str,
                       authority: str, 
                       authority_id_col: str, 
                       new_column_name: str
                       ) -> pd.DataFrame: # MARK: add_authority_name_column
    """
    Add a new column to a DataFrame with a name in the required format: 'Last, First, Dates, Role URI'

    Args:
        df (pd.DataFrame): The DataFrame to process
        name_col (str): The name of the column containing the name
        role_col (str): The name of the column containing the role
        authority_col (str): The name of the column containing the authority
        authority (str): The authority to filter on
        authority_id_col (str): The name of the column containing the authority ID
        new_column_name (str): The name of the new column to create

    Returns:
        pd.DataFrame: The DataFrame with the new column added
    """

    log.debug(f'entering add_authority_name_column')
    new_df = df.apply(process_row_alpha, args=(name_col, role_col, authority_col, authority_id_col, authority, new_column_name), axis=1)
    return new_df
#endregion    
        

#region MAIN FUNCTION
def main():
    # Process command line arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Path to the CSV file to be read')
    args = parser.parse_args()
    log.info(f'successfully parsed args, ``{args}``')

    # Read the CSV file
    data = read_csv(args.file_path)
    df = make_df(data)

    # Add the namePersonOtherVIAF column using process_row_alpha
    # new_df = add_authority_name_column(df, name_col='Authoritized Name', authority_id_col='Authority ID', authority_col='Authority Used', authority='viaf', role_col='Position', new_column_name='namePersonOtherVIAF')
    
    # Add the namePersonOtherVIAF column using process_row_beta
    output_format = [
        FormattedOutput(text=None, column_name='Authoritized Name', function=None, kwargs=None),
        FormattedOutput(text=', ', column_name=None, function=None, kwargs=None),
        FormattedOutput(text=None, column_name='Position', function=None, kwargs=None),
        FormattedOutput(text=' ', column_name=None, function=None, kwargs=None),
        FormattedOutput(text=None, column_name=None, function=build_uri, kwargs={'authority': 'Authority Used', 'id': 'Authority ID'})
    ]
    new_df = df.apply(process_row_beta, args=('namePersonOtherVIAF', output_format, 'Authority Used', 'viaf'), axis=1)

    # # Add the namePersonOtherLocal column
    # new_df = add_authority_name_column(new_df, name_col='Authoritized Name', authority_id_col='Authority ID', authority_col='Authority Used', authority='local', role_col='Position', new_column_name='namePersonOtherLocal')
    # # Add the namePersonOtherLocal column using process_row_beta
    output_format = [
        FormattedOutput(text=None, column_name='Authoritized Name', function=None, kwargs=None),
        FormattedOutput(text=', ', column_name=None, function=None, kwargs=None),
        FormattedOutput(text=None, column_name='Position', function=None, kwargs=None),
    ]
    new_df = new_df.apply(process_row_beta, args=('namePersonOtherLocal', output_format, 'Authority Used', 'local'), axis=1)

    # Add the namePersonCreatorLC column
#     namePersonCreatorLC (FileMakerPro: sources sheet -> Organization Name, Source, URI)
#       Find name, pull data if LCNAF URIs, ignore all others (this will be the same value as in the subjectNamesLC field)
#       Neipp, Paul C. http://id.loc.gov/authorities/names/no2008182896

    


    print(new_df.head())
    log.info(f'Finished processing DataFrame, writing to CSV')
    if not os.path.exists('../output'):
        os.makedirs('../output')
    new_df.to_csv('../output/processed_data.csv', index=False)
#endregion

#region DUNDER MAIN
if __name__ == '__main__':
    main()
#endregion