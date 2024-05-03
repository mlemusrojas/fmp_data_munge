#region IMPORTS
import os, sys
import argparse
import csv
import logging
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
import requests
import time
import json
import atexit

from tqdm import tqdm, tqdm_pandas
import pandas as pd
from dotenv import load_dotenv, find_dotenv
#endregion

# Load environment variables. If no .env exists, use default values
LGLVL = 'INFO'
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path, override=True)
    LGLVL = os.environ['LOGLEVEL']

#region LOGGING
## set up logging ---------------------------------------------------
lglvldct = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARN': logging.WARNING,
    'ERROR': logging.ERROR,
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
ch.setLevel(logging.ERROR)  # note: this level is _not_ the same as the file-handler level set in the `.env`
ch.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s [%(module)s-%(funcName)s()::%(lineno)d] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S',
))
log.addHandler(ch)
#endregion

#region CLASSES

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

class RateLimiter:
    """
    A class to rate limit API calls to different domains

    Attributes:
        rate_limits (Dict[str, float]): A dictionary of rate limits for different domains
        last_api_call_times (Dict[str, float]): A dictionary of the last time an API call was made to each domain

    Example:
        ```
        rate_limiter = RateLimiter({
            'lc': 1,
            'viaf': 1
        })
        rate_limiter.rate_limit_api_call('lc')
        ```
    """

    def __init__(self, rate_limits):
        self.rate_limits = rate_limits
        self.last_api_call_times = {domain: 0.0 for domain in rate_limits}

    def rate_limit_api_call(self, domain):
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_times[domain]
        rate_limit = self.rate_limits[domain]
        if time_since_last_call < rate_limit:
            rest_time = rate_limit - time_since_last_call
            log.debug(f'Rate limiting API call to {domain} for {rest_time} seconds')
            time.sleep(rest_time)
        self.last_api_call_times[domain] = time.time()

class LocalCache:
    """
    A class to handle retrieving and storing responses from API calls to local
    files to avoid making redundant calls

    Attributes:
        cache_file (str): The path to the cache file
        cache (Dict[str, Any]): The cache dictionary

    Example:
        ```
        cache = LocalCache('cache.json')
        cache.set_response('http://example.com', 'response')
        response = cache.get_response('http://example.com')
        ```
    """

    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.counter = 0
        atexit.register(self.save_cache)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f'''Error loading cache file {self.cache_file},
                      make sure it is a valid JSON file''')
                log.error(f'Error loading cache file {self.cache_file}')
                # Ask the user if they want to exit or continue without the cache
                exit = input('Do you want to proceed without the cache? (y/n) ')
                if exit.lower() == 'y':
                    return {}
                else:
                    sys.exit()
        return {}

    def save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=4)
        except Exception as e:
            log.error(f'Error saving cache file {self.cache_file}: {e}')

    def get_response(self, key):
        return self.cache.get(key, None)
    
    def set_response(self, key, response):
        self.cache[key] = response
        self.counter += 1
        # Save the cache every 10 API calls
        if self.counter >= 10:
            log.debug(f'Saving cache after {self.counter} API calls')
            self.counter = 0
            self.save_cache()

    def write_and_return_response(self, key, response):
        self.set_response(key, response)
        return response

    def clear_cache(self):
        self.cache = {}
        self.counter = 0 # Reset the counter to 0
        self.save_cache()

    def __contains__(self, key):
        return key in self.cache
    
    def __getitem__(self, key):
        return self.get_response(key)
    
    def __setitem__(self, key, value):
        self.set_response(key, value)
    
    def __str__(self):
        return str(self.cache)
    
    def __repr__(self):
        return repr(self.cache)

#endregion

#region FUNCTIONS
# =============================================================================
# FUNCTIONS
# =============================================================================


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and return the data as a pandas DataFrame

    Args:
        file_path (str): The path to the CSV file

    Returns:
        pd.DataFrame: The data from the CSV file
    """

    df: pd.DataFrame = pd.read_csv(file_path, dtype='string')
    # df = pd.read_csv(file_path, dtype=str)
    log.info(f'''Read DataFrame with {len(df)} rows and {len(df.columns)} 
             columns from {file_path}''')
    return df
    
def write_csv(data: pd.DataFrame, file_path: str):
    """
    Write a pandas DataFrame to a CSV file

    Args:
        data (pd.DataFrame): The data to write to the CSV file
        file_path (str): The path to the CSV file
    """

    data.to_csv(file_path, index=False)
    log.info(f'Wrote data to {file_path}')

# MARK: STUDENT SPREADSHEET FUNCTIONS

def clean_student_spreadsheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the student spreadsheet by removing unnecessary columns and rows
    
    Args:
        df (pd.DataFrame): The student spreadsheet DataFrame

    Returns:
        pd.DataFrame: The cleaned student spreadsheet DataFrame
    """

    # Remove 2nd row
    df = df.iloc[1:]

    # Remove all columns except:
    '''
    HH ID, # of folders\ngoing to vendor, dateText, PERMANENT BOX NUMBER(S)
    '''
    columns_to_keep = ['HH ID', 
                       '# of folders\ngoing to vendor', 
                       'dateText', 
                       'PERMANENT BOX NUMBER(S)' 
                       ]
    df = df[columns_to_keep]

    # Rename columns
    df.columns = ['ss_HH ID', 'ss_Number of Folders', 
                  'ss_DateText', 'ss_Box Numbers']

    log.info(f'''Cleaned student spreadsheet with {len(df)} rows 
             and {len(df.columns)} columns''')    

    return df

def get_min_max_dates(col_value: str) -> tuple[str, str]:
    raise NotImplementedError



def create_authority_name(**fields) -> str:
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

def create_formatted_date(start_date: str | None, 
                          end_date: str | None) -> str | None:
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
    
def build_uri(authority: str | None, id: str | None) -> str | None:
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

def reduce_list(values: str, flags: list[bool]) -> str:
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

    return '|'.join([value for value, flag in 
                     zip(values.split('|'), flags) if flag])

def process_row(row: pd.Series,
                new_column_name: str, 
                output_format: list[FormattedOutput],
                mask_column: str | None = None,
                mask_value: str | None = None
                ) -> pd.Series:

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

    log.debug(f'entering process_row')

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

def add_nameCorpCreatorLocal_column(row: pd.Series) -> pd.Series:
    """
    Process a row of a DataFrame to create a new column, 
    populating it with the 'Organization Name' from the 'sources' sheet if neither 'LCNAF' nor 'VIAF' names are found.

    Args:
        row (pd.Series): The row to process

    Returns:
        pd.Series: The processed row
    """

    log.debug(f'entering add_nameCorpCreatorLocal_column')

    # check if 'nameCorpCreatorLC' is empty
    if row['nameCorpCreatorLC']:
        log.debug(f'nameCorpCreatorLC is not empty, returning row')
        row['nameCorpCreatorLocal'] = ''
        return row
    
    # check if 'namePersonCreatorLC' is empty
    if row['namePersonCreatorLC']:
        log.debug(f'namePersonCreatorLC is not empty, returning row')
        row['nameCorpCreatorLocal'] = ''
        return row
    
    # check if 'nameCorpCreatorVIAF' is empty
    if row['nameCorpCreatorVIAF']:
        log.debug(f'nameCorpCreatorVIAF is not empty, returning row')
        row['nameCorpCreatorLocal'] = ''
        return row
    
    # Try to pull the first value from 'Organization Name_sources'
    sources_name = row['Organization Name_sources'].split('|')[0]
    if sources_name:
        row['nameCorpCreatorLocal'] = sources_name
        return row
    log.debug(f'No Organization Name found in Organization Name_sources, attempting to pull from Organization Name_subjects')
    subjects_name = row['Organization Name_subjects'].split('|')[0]
    if subjects_name:
        row['nameCorpCreatorLocal'] = subjects_name
    else:
        row['nameCorpCreatorLocal'] = ''
        log.warning(f'No Organization Name found for row: {row}')

    log.debug(f'Processed row: {row}')
    return row

# MARK: API CALLS

# Initialize the rate limiter
rate_limiter = RateLimiter({
    'lc': 1,
    'viaf': 1
})

# Initialize the local caches
lc_subject_cache = LocalCache('lc_subject_cache.json')
lc_name_type_cache = LocalCache('lc_name_type_cache.json')
viaf_name_cache = LocalCache('viaf_name_cache.json')

def lc_get_subject_uri(subject_term: str) -> str | None:
    """
    Call the Library of Congress API to get the URI for a subject term

    Args:
        subject_term (str): The subject term to search for

    Returns:
        str: The URI of the subject term or None if not found
    """
    log.debug(f'entering lc_get_subject_uri')

    # Check the local cache
    if subject_term in lc_subject_cache:
        if lc_subject_cache[subject_term] == 'NOT_FOUND':
            return None
        return lc_subject_cache[subject_term]

    # Limit the rate of API calls if necessary
    rate_limiter.rate_limit_api_call('lc')

    if len(subject_term) > 1:
        subject_term_correct_case = subject_term[0].upper() + subject_term[1:].lower()
    else:
        subject_term_correct_case = subject_term

    if subject_term != subject_term_correct_case:
        log.debug(f'Correcting case for {subject_term} to {subject_term_correct_case} for API call')

    try:
        response = requests.head(f'https://id.loc.gov/authorities/subjects/label/{subject_term_correct_case}', allow_redirects=True)
    except requests.exceptions.RequestException as e:
        log.error(f'Error with request: {e}')
        return None
    if response.ok:
        return lc_subject_cache.write_and_return_response(subject_term, response.headers['x-uri'])
    
    if response.status_code == 404:
        log.debug(f'No URI found for {subject_term}')
        lc_subject_cache[subject_term] = 'NOT_FOUND'

    return None

def lc_get_name_type(uri: str) -> str | None:
    """
    Call the Library of Congress API to get the type of a name (Personal or Corporate)

    Args:
        uri (str): The URI to search for

    Returns:
        str: The type of the name or None if not found
    """
    log.debug(f'entering lc_get_name_type')

    # Check the local cache
    if uri in lc_name_type_cache:
        if lc_name_type_cache[uri] == 'NOT_FOUND':
            return None
        return lc_name_type_cache[uri]

    # Limit the rate of API calls if necessary
    rate_limiter.rate_limit_api_call('lc')

    response = requests.get(f'{uri}.json')
    if response.ok:
        log.debug(f'LC API call successful')
        data: list[dict[str, Any]] = response.json()
        # find the dictionary with a key of '@id' and a value of the uri
        matching_dict: dict[str, Any] = [d for d in data if d.get('@id', None) == uri][0]
        log.debug(f'{matching_dict = }')
        # get the values from the '@type' key
        name_types: list[str] = matching_dict.get('@type', None)
        log.debug(f'{name_types = }')
        if not name_types:
            log.warning(f'No name types found for {uri}')
            return None
        if "http://www.loc.gov/mads/rdf/v1#CorporateName" in name_types:
            return lc_name_type_cache.write_and_return_response(uri, 'Corporate')
        elif "http://www.loc.gov/mads/rdf/v1#PersonalName" in name_types:
            return lc_name_type_cache.write_and_return_response(uri, 'Personal')
    else:
        log.warning(f'LC API call failed for ```{uri}```, {response.status_code = }')

    if response.status_code == 404:
        log.warning(f'No name found for {uri}')
        lc_name_type_cache[uri] = 'NOT_FOUND'
        
    return None

def get_viaf_name(uri: str) -> str:
    """
    Call the VIAF API to get the name of a person or organization

    Args:
        uri (str): The URI to search for

    Returns:
        str: The name of the person or organization or 'NOT_FOUND' if not found
    """
    log.debug(f'entering get_viaf_name')

    # Check the local cache
    if uri in viaf_name_cache:
        return viaf_name_cache[uri]

    # Limit the rate of API calls if necessary
    rate_limiter.rate_limit_api_call('viaf')

    response = requests.get(f'{uri}/viaf.json')

    if not response.ok:
        log.warning(f'Error with {uri}')
        return viaf_name_cache.write_and_return_response(uri, 'NOT_FOUND')

    # Parse the JSON response
    response_json: dict[str, Any] = response.json()
    if response_json.get('redirect'):
        try:
            redirect_id = response_json['redirect']['directto']
        except KeyError:
            log.warning(f'Problem following redirect for {uri}')
            return viaf_name_cache.write_and_return_response(uri, 'NOT_FOUND')
        redirect_uri = f'http://viaf.org/viaf/{redirect_id}'
        return viaf_name_cache.write_and_return_response(uri, get_viaf_name(redirect_uri))
    main_headings: dict[str, Any] = response_json.get('mainHeadings', {})
    data: list[dict[str, Any]] | dict[str, Any] = main_headings.get('data', [])

    # Sometimes the data is a single dictionary, instead of a list of dictionaries
    if isinstance(data, dict):
        data = [data]
    for d in data:
        try:
            sources = d.get('sources', None)
        except AttributeError:
            log.warning(f'Error with {uri}')
            raise
        if 'LC' in sources['s']:
            name = d.get('text', None)
            if name:
                return viaf_name_cache.write_and_return_response(uri, name)
            
    log.warning(f'Unable to find name for ``{uri}``')
    return viaf_name_cache.write_and_return_response(uri, 'NOT_FOUND')

def get_unique_values_from_column(column: pd.Series) -> set[str]:
    """
    Get unique values from a column of a DataFrame, separating pipe-separated values

    Args:
        column (pd.Series): The column to process

    Returns:
        set[str]: The unique values
    """
    
    unique_values: set[str] = set()
    for value in column:
        unique_values.update(value.split('|'))
    return unique_values

def build_uri_dict(values: set[str], api_call: Callable) -> dict[str, str]:
    """
    Build a dictionary of URIs from a set of values using an API call

    Args:
        values (set[str]): The values to search for
        api_call (Callable): The function to call to get the URI for a value

    Returns:
        dict[str, str]: The dictionary of values and URIs
    """
    
    uri_dict: dict[str, str] = {}
    for value in tqdm(values):
        uri = api_call(value)
        if uri:
            uri_dict[value] = uri
        # time.sleep(0.2)
    return uri_dict

def add_subjectTopics(row: pd.Series, uri_dict: dict[str, str]) -> pd.Series:
    """
    Process a row of a DataFrame to populate either subjectTopicsLC or subjectTopicsLocal columns. 
    Populates subjectTopicsLC if an LC URI is found, subjectTopicsLocal if not.

    Args:
        row (pd.Series): The row to process

    Returns:
        pd.Series: The processed row
    """

    log.debug(f'entering add_subjectTopics')

    # Create list of subject terms from pipe-separated values in 'Subject Heading'
    subject_terms: list[str] = row['Subject Heading'].split('|')

    # Iterate through subject terms to find URIs
    uri_terms: list[str] = []
    local_terms: list[str] = []
    for term in subject_terms:
        uri = uri_dict.get(term, None)
        if uri:
            uri_terms.append(f'{term} {uri}')
        else:
            local_terms.append(term)
    if not uri_terms:
        uri_terms = ['']
    if not local_terms:
        local_terms = ['']

    # Concatenate URIs and local terms
    row['subjectTopicsLC'] = '|'.join(uri_terms)
    row['subjectTopicsLocal'] = '|'.join(local_terms)

    log.debug(f'Processed row: {row}')
    return row

def make_name_type_column(row: pd.Series, uri_column: str, authority_column: str) -> pd.Series:
    """
    Process a row of a DataFrame to create a new column,
    populating it with either 'Personal' or 'Corporate' based on an LC API call.

    Args:
        row (pd.Series): The row to process

    Returns:
        pd.Series: The processed row
    """

    log.debug(f'entering make_name_type_column')

    # Get authority and URI values
    authorities: list[str] = row[authority_column].split('|')
    if 'LCNAF' not in authorities:
        row['Name Type'] = ''
        return row
    uris: list[str] = row[uri_column].split('|')

    # Find first LC URI
    uri: str | None = None
    for i, authority in enumerate(authorities):
        if authority == 'LCNAF':
            uri = uris[i]
            break

    # Get name type
    if not uri:
        row['Name Type'] = ''
        return row
    name_type: str | None = lc_get_name_type(uri)
    if not name_type:
        name_type = ''
    row['Name Type'] = name_type

    log.debug(f'Processed row: {row}')
    return row

def handle_person_and_corp_lc_names(row: pd.Series) -> pd.Series:
    """
    Creates the namePersonCreatorLC and nameCorpCreatorLC columns by handing off to process_row based on the value
    in the 'Name Type' column.

    Args:
        row (pd.Series): The row to process

    Returns:
        pd.Series: The processed row
    """
    
    log.debug(f'entering handle_person_and_corp_lc_names')

    # Check if 'Name Type' is empty
    if not row['Name Type']:
        row['namePersonCreatorLC'] = ''
        row['nameCorpCreatorLC'] = ''
        return row

    output_format: list[FormattedOutput] = [
        FormattedOutput(text=None, column_name='Organization Name_sources', function=None, kwargs=None),
        FormattedOutput(text=' ', column_name=None, function=None, kwargs=None),
        FormattedOutput(text=None, column_name='URI', function=None, kwargs=None)
    ]

    # Check if 'Name Type' is 'Personal'
    if row['Name Type'] == 'Personal':
        row = process_row(row, 'namePersonCreatorLC', output_format, 'Source', 'LCNAF')
        row['nameCorpCreatorLC'] = ''
        return row

    # Check if 'Name Type' is 'Corporate'
    if row['Name Type'] == 'Corporate':
        row = process_row(row, 'nameCorpCreatorLC', output_format, 'Source', 'LCNAF')
        row['namePersonCreatorLC'] = ''
        return row

    log.debug(f'Processed row: {row}')
    return row

#endregion    
        
#region MAIN FUNCTION
def main():
    # Process command line arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fmp_file', help='The path to the input CSV file')
    parser.add_argument('student_file', help='The path to the "student spreadsheet" CSV file')
    parser.add_argument('output_file', help='''The path to the output CSV file. 
                        Will be created if it does not exist. Will overwrite if
                         it does. Default is ../output/processed_data.csv''',
                        default='../output/processed_data.csv')
    args = parser.parse_args()
    log.info(f'successfully parsed args, ``{args}``')

    # Set up tqdm.pandas() to use progress_apply
    tqdm.pandas()



    # Read the CSV files
    fmp_df: pd.DataFrame = read_csv(args.fmp_file)
    student_df: pd.DataFrame = read_csv(args.student_file)

    # MARK: NEW WORK

    # Clean the student spreadsheet
    student_df = clean_student_spreadsheet(student_df)

    # Print the student columns
    print(student_df.columns)
    print(student_df.head())
    print(student_df.info())

    # Replace null values with empty strings
    student_df = student_df.fillna('')

    # Use groupby to concatenate the values in each column for each HH ID
    student_df = student_df.groupby('ss_HH ID').agg(lambda x: '|'.join(x))
    
    print(student_df.head())

    sys.exit()







    
    # Add the namePersonOtherVIAF column MARK: namePersonOtherVIAF
    log.debug(f'Adding the namePersonOtherVIAF column')
    print('Adding the namePersonOtherVIAF column. This could take a while as it requires an API call for each new VIAF URI.')
    output_format: list[FormattedOutput] = [
        FormattedOutput(text=None, column_name=None, function=get_viaf_name, 
                        kwargs={'uri': 'Authority URI'}),
        FormattedOutput(text=', ', column_name=None, function=None, kwargs=None),
        FormattedOutput(text=None, column_name='Position', function=None, kwargs=None),
        FormattedOutput(text=' ', column_name=None, function=None, kwargs=None),
        FormattedOutput(text=None, column_name=None, function=build_uri, 
                        kwargs={'authority': 'Authority Used', 'id': 'Authority ID'})
    ]
    new_df: pd.DataFrame = df.progress_apply(process_row, 
                                             args=('namePersonOtherVIAF', 
                                                   output_format, 
                                                   'Authority Used', 
                                                   'viaf'), 
                                             axis=1) # type: ignore
    print('Finished adding the namePersonOtherVIAF column')

    # Add the namePersonOtherLocal column MARK: namePersonOtherLocal
    log.debug(f'Adding the namePersonOtherLocal column')
    print('Adding the namePersonOtherLocal column.') 
    output_format: list[FormattedOutput] = [
        FormattedOutput(text=None, column_name='Authoritized Name', function=None, kwargs=None),
        FormattedOutput(text=', ', column_name=None, function=None, kwargs=None),
        FormattedOutput(text=None, column_name='Position', function=None, kwargs=None),
    ]
    new_df: pd.DataFrame = new_df.apply(process_row, args=('namePersonOtherLocal', output_format, 'Authority Used', 'local'), axis=1)

    # Make the nameType column
    print('Adding the (temporary) Name Type column. This could take a while as it requires an API call for each new LCNAF URI.')
    new_df: pd.DataFrame = new_df.progress_apply(make_name_type_column, args=('URI', 'Source'), axis=1) # type: ignore
    print('Finished adding the Name Type column')

    # Add the namePersonCreatorLC and nameCorpCreatorLC columns MARK: namePersonCreatorLC, nameCorpCreatorLC
    print('Adding the namePersonCreatorLC and nameCorpCreatorLC columns')
    new_df: pd.DataFrame = new_df.apply(handle_person_and_corp_lc_names, axis=1)

    # Add the nameCorpCreatorVIAF column MARK: nameCorpCreatorVIAF
    """
    If no LCNAF, find name, Pull only VIAF URIs, ignore all others
    """
    print('Adding the nameCorpCreatorVIAF column.')
    output_format: list[FormattedOutput] = [
        FormattedOutput(text=None, column_name='Organization Name_sources', function=None, kwargs=None),
        FormattedOutput(text=' ', column_name=None, function=None, kwargs=None),
        FormattedOutput(text=None, column_name='URI', function=None, kwargs=None)
    ]
    new_df: pd.DataFrame = new_df.apply(process_row, args=('nameCorpCreatorVIAF', output_format, 'Source', 'VIAF'), axis=1)

    # We only want to keep the nameCorpCreatorVIAF column if the nameCorpCreatorLC and namePersonCreatorLC columns are empty
    new_df['nameCorpCreatorVIAF'] = new_df.apply(
        lambda row: row['nameCorpCreatorVIAF'] 
        if not row['nameCorpCreatorLC'] 
        and not row['namePersonCreatorLC'] 
        else '', 
        axis=1
        )

    # Add the nameCorpCreatorLocal column MARK: nameCorpCreatorLocal
    """
    nameCorpCreatorLocal (FileMakerPro: sources sheet -> Organization Name, Source)
        If no LCNAF and no VIAF, find name, pull just one
        If no name is found as Local, add the Organization Name from the 
        subjects sheet (this will be the same value as in the subjectCorpLocal field)
        Ex: The Presbyterian Journal
    """
    print('Adding the nameCorpCreatorLocal column.')
    new_df: pd.DataFrame = new_df.apply(add_nameCorpCreatorLocal_column, axis=1)

    # Add the subjectTopicsLC and subjectTopicsLocal columns MARK: subjectTopicsLC, subjectTopicsLocal
    print('Adding subjectTopicsLC and subjectTopicsLocal columns (hitting LC API for each new unique subject term)')
    unique_subjects = get_unique_values_from_column(new_df['Subject Heading'])
    uri_dict = build_uri_dict(unique_subjects, lc_get_subject_uri)
    new_df: pd.DataFrame = new_df.apply(add_subjectTopics, args=(uri_dict,), axis=1)
    print('Finished adding subjectTopicsLC and subjectTopicsLocal columns')

    # Add subjectNamesLC MARK: subjectNamesLC
    """
    (FileMakerPro: sources sheet -> Organization Name, Source, URI)
    """
    # this will be the same value as in the namePersonCreatorLC field, so we can just copy that value
    print('Adding subjectNamesLC column')
    new_df['subjectNamesLC'] = new_df['namePersonCreatorLC']

    # Add subjectCorpLC MARK: subjectCorpLC
    """
    (FileMakerPro: sources sheet -> Organization Name, Source, URI)
    """
    # this will be the same value as in the nameCorpCreatorLC field, so we can just copy that value
    print('Adding subjectCorpLC column')
    new_df['subjectCorpLC'] = new_df['nameCorpCreatorLC']

    # Add subjectCorpVIAF MARK: subjectCorpVIAF
    """
    (FileMakerPro: sources sheet -> Organization Name, Source, URI)
    """
    # this will be the same value as in the nameCorpCreatorVIAF field, so we can just copy that value
    print('Adding subjectCorpVIAF column')
    new_df['subjectCorpVIAF'] = new_df['nameCorpCreatorVIAF']

    # Remove the 'Name Type' column
    new_df.drop('Name Type', axis=1, inplace=True)

    # print(new_df.head())
    log.info(f'Finished processing DataFrame, writing to CSV')
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_csv(new_df, args.output_file)
    print('Done!')
#endregion

#region DUNDER MAIN
if __name__ == '__main__':
    main()
#endregion