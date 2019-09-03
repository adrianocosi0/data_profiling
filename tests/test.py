import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import unittest
import pandas as pd
from src.DataProfiler import DataProfiler
import datetime

data_path = '../data/Acc.csv'

class TestCase(unittest.TestCase):

    def setUp(self):
        self.testdata = pd.read_csv(data_path)
        self.profiler = DataProfiler()

    def test_dataframe_validity(self):
        bad_data = {1:[5], 3:[4,5,6,7]}
        self.assertRaises(ValueError, self.profiler._check_data, *(bad_data,))

    def test_column_convertions(self):
        self.assertEqual(self.profiler.convert_column(self.testdata, 'Date', convert_to='datetime').dtype,
                         pd.to_datetime(self.testdata['Date']).dtype)
        self.assertEqual(self.profiler.convert_column(self.testdata, 'Road_Type', convert_to='categorical').dtype,
                         self.testdata['Road_Type'].astype('category').dtype)

    def test_column_description(self):
        """Check that summary stats is correct and that results are populated correctly"""
        df = self.profiler.prepare_dataframe(self.testdata, date_columns=['Date'],
                                             categorical_columns=['Accident_Severity'])
        res = self.profiler.describe_columns(df, ['Number_of_Vehicles', 'Accident_Severity'],
                                             identify_outliers=True, exclude_outliers_from_graph=True)
        #check outliers for numerical col
        lb = self.testdata['Number_of_Vehicles'].mean() - 1.96*self.testdata['Number_of_Vehicles'].std()
        ub = self.testdata['Number_of_Vehicles'].mean() + 1.96*self.testdata['Number_of_Vehicles'].std()
        outliers = (self.testdata['Number_of_Vehicles'] <= lb) | (self.testdata['Number_of_Vehicles'] >= ub)
        self.assertEqual((outliers == res.outliers['Number_of_Vehicles']).sum(), self.testdata.shape[0])

        #check graphs are populated
        self.assertEqual(set(list(res.graphs['Number_of_Vehicles'].keys())),
                         set(self.profiler.graph_per_category['numerical']))
        self.assertEqual(set(list(res.graphs['Accident_Severity'].keys())),
                         set(self.profiler.graph_per_category['categorical']))

    def test_distribution_over_time(self):
        """Check plots over time periods"""
        dataframe = self.profiler.prepare_dataframe(self.testdata, date_columns=['Date'],
                                             categorical_columns=['Accident_Severity'])
        res = self.profiler.distribution_over_time_for_columns(dataframe, ['Number_of_Casualties', 'Road_Type'], 'Date',
                                                                start_date=pd.to_datetime(
                                                                dataframe['Date'].min() + datetime.timedelta(100)),
                                                                frequency='week',
                                                                frequency_multiplier=2
                                                                )
        self.assertEqual(set(list(res.keys())), set(['Number_of_Casualties', 'Road_Type']))

        #wrong frequency
        self.assertRaises(ValueError, self.profiler.distribution_over_time_for_columns,
                         *(dataframe, ['Number_of_Casualties', 'Road_Type'], 'Date', None, None, '2weeks', 3))

if __name__ == '__main__':
    unittest.main()
