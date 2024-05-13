# fmp_data_munge

## Description
This script takes a csv spreadsheet of FileMaker Pro data and adds new columns to the csv based on the data in the existing columns and API calls to LCNAF and VIAF.

## Installation
1. Create a directory to hold the project and navigate to it (you can name it what you like):
    ```shell
    mkdir fmp_data_munge_outer
    cd fmp_data_munge_outer
    ```
2. Copy the repository url and clone the repository:
    ```shell
    git clone [repository url]
    ```
3. Create a virtual environment in the root directory of the project (recommended):
    ```shell
    python -m venv [virtual environment name]
    ```
4. If you created a virtual environment, activate it using the following command:
    ```shell
    source [virtual environment name]/bin/activate
    ```
5. Install the required packages using the following command:
    ```shell
    pip install -r requirements.txt
    ```

    ## Usage
    1. If you have not already done so, activate the virtual environment (if you created one):
        ```shell
        source [virtual environment name]/bin/activate
        ```
    2. Run the script using the following command:
        ```shell
        python fmp_data_munge.py [fmp_file] [student_file] [output_file] [orgs_file]
        ```
        The `fmp_file` is the path to the input CSV file. The `student_file` is the path to the "student spreadsheet" CSV file. The `output_file` is the path to the output CSV file. The output file will be created if it does not exist and will be overwritten if it does. The default output file path is `../output/processed_data.csv`. The `orgs_file` is the path to the txt file containing the list of organizations to include. If `orgs_file` is not provided, all organizations will be included. If there are spaces in any file path, enclose the path in quotes.

        example:
        ```shell
        python fmp_data_munge.py "My Files/fmp_data.csv" "My Files/student_data.csv" "My Files/output_data.csv" "My Files/orgs.txt"
    ```

## Contributing
If you would like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
