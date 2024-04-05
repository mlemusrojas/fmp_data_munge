import unittest
from fmp_data_munge import create_lc_name

class TestMunger(unittest.TestCase):

    def test_create_lc_name__single_good_data(self):
        """
        Checkes that the create_lc_name function returns the expected output for single entry
        """
        name = 'John Doe'
        date = '2024-04-05'
        role = 'Developer'
        uri = 'https://example.com'
        expected = 'John Doe, 2024-04-05, Developer https://example.com'
        result = create_lc_name(name, date, role, uri)
        self.assertEqual(expected, result)

    def test_create_lc_name__multiple_good(self):
        """
        Checks that the create_lc_name function returns the expected output for multiple entries
        """
        row_data = [ 
            {'source': ('John Doe', '2024-04-05', 'Developer', 'https://example.com'), 
                'expected': 'John Doe, 2024-04-05, Developer https://example.com'},
            {'source': ('Jane Smith', '2024-04-05', 'Designer', 'https://example.com'),
                'expected': 'Jane Smith, 2024-04-05, Designer https://example.com'},
            {'source': ('David Brown', '2024-04-05', 'Manager', 'https://example.com'),
                'expected': 'David Brown, 2024-04-05, Manager https://example.com'}
            ]
        for row in row_data:
            name, date, role, uri = row['source']
            expected = row['expected']
            result = create_lc_name(name, date, role, uri)
            self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
