import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from pprint import pprint
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_categorical, is_datetime64_any_dtype, is_datetimetz
import numpy as np
import logging
from collections import defaultdict

class DataProfiler:
    """Class for data profiling"""

    graph_per_category = {'numerical': ['hist', 'density', 'line', 'box'],
                          'categorical': ['bar'],
                          'datetime': ['hist']}
    logger = logging.getLogger('DataProfiler')
    def _check_data(self, data):
        try:
            data = pd.DataFrame(data)
        except ValueError:
            raise ValueError('Provided data is not a dataframe and cannot be converted to it!')

    def convert_column(self, dataframe, column, convert_to='datetime'):
        """
        :param dataframe: Dataframe with column to convert
        :param column: Column to convert
        :param convert_to: One of "datetime" and "categorical". Specify column type to convert data to. Defaults to 'datetime'
        :return: Converted column
        :note: Works only on 'object' (i.e. generic) type dataframe columns.
        """
        available_convertions = ['categorical', 'datetime']
        try:
            assert convert_to in available_convertions
        except AssertionError:
            raise('Convert variable to one of {}'.format(' or '.join(available_convertions)))
        data = dataframe[column]
        print(convert_to)
        if convert_to == 'datetime':
            data = pd.to_datetime(data)
        elif convert_to == 'categorical':
            data = data.astype('category')
            data = data.cat.add_categories('Missing')
            data.fillna('Missing', inplace=True)
        return data

    def prepare_dataframe(self, dataframe, date_columns=[], categorical_columns=[]):
        """
        :param dataframe: Dataframe with columns to convert
        :param date_columns: List of date columns in dataframe
        :param categorical_columns: List of categorical columns in dataframe
        :return: Dataframe with converted columns
        """
        for column in date_columns:
            dataframe[column] = self.convert_column(dataframe, column)
        for column in categorical_columns:
            dataframe[column] = self.convert_column(dataframe, column, convert_to='categorical')
        return dataframe

    def describe_columns(self, dataframe, columns, percentiles=[.25,.50,.75],
                         graph_types = ['hist', 'density', 'pareto'],
                         identify_outliers=True, exclude_outliers_from_graph=True):
        """
        Return a dictionary of column name to associated summary statistics

        Parameters
        ----------
        dataframe
            DataFrame containing column(s) to describe
        columns
            Column(s) to describe in dataframe
        percentiles
            Percentiles (range 0 to 1) to calculate for summary statistics.
            This has effect only on numerical columns
        """
        self._check_data(dataframe)
        try:
            assert isinstance(columns, list)
        except AssertionError:
            raise('A list of columns needs to be passed')
        columns_to_summary_stats = {}
        graph_type_to_graph = defaultdict(dict)
        for column in columns:
            if column not in dataframe.columns:
                raise KeyError('Column {} is not in dataframe'.format(column))
            #summary stats
            column_stats = self.summary_stats(dataframe, column, percentiles=percentiles)
            columns_to_summary_stats[column] = column_stats
            #outliers identification
            outliers_ind = np.zeros(dataframe.shape[0], dtype=bool)
            if identify_outliers:
                outliers_ind = self.identify_outliers(dataframe, column, column_stats=column_stats)
            data_to_graph = df[column]
            #plot graphs
            self.plot_descriptive_graphs_for_column(dataframe, column, outliers_ind=outliers_ind)
        return columns_to_summary_stats, graph_type_to_graph

    def summary_stats(self, dataframe, column, percentiles=[.25,.50,.75]):
        """
        :param dataframe: Input data
        :param column: Column to describe
        :param percentiles: Percentiles for numerical column (range 0 to 1)
        :return: Summary statistics for column
        """
        column_stats = dataframe[column].describe(percentiles=percentiles)
        print('\n\n{} Summary Statistics:\n'.format(column))
        pprint(column_stats)
        return column_stats

    def identify_outliers(self, dataframe, column, column_stats=pd.DataFrame()):
        """
        :param dataframe:
        :param column:
        :param column_stats: If passed avoids calculating summary stats for data
        :return: boolean array for outlier values
        """
        if not column_stats.empty:
            column_stats = self.summary_stats(dataframe, column)
        if 'mean' in column_stats.index:  # check that mean is in stats --> it's a numeric column
            conf_inter = [column_stats.loc['mean'] - (1.96 * column_stats.loc['std']), \
                          column_stats.loc['mean'] + (1.96 * column_stats.loc['std'])]
        else:
            return np.zeros(dataframe.shape[0], dtype=bool)
        outliers_ind = (df[column] < conf_inter[0] | df[column] > conf_inter[1])
        return outliers_ind

    def plot_descriptive_graphs_for_column(self, dataframe, column, outliers_ind=None, show=False):
        graph_type_to_graph = {}
        if is_numeric_dtype(dataframe[column]):
            cat = 'numerical'
            if isinstance(outliers_ind, np.ndarray):
                data_to_plot = dataframe.loc[~outliers_ind, column]
        elif is_categorical(dataframe[column]):
            cat = 'categorical'
            data_to_plot = dataframe[column].value_counts()
        elif is_datetime64_any_dtype(dataframe[column]) or is_datetimetz(dataframe[column]):
            cat = 'datetime'
        else:
            self.logger.warning('''Column {} could not be plotted because of object (generic) type.
            Please convert it to a categorical, date or numerical type. String columns cannot be plotted!''')
        for graph in self.graph_per_category[cat]:
            graph_type_to_graph[graph] = data_to_plot.plot(kind=graph)
            if show:
                graph_type_to_graph[graph].figure.show()

if __name__ == '__main__':
    df = pd.read_csv('../data/Acc.csv')
    print(df.columns)
    profiler = DataProfiler()
    dataframe = profiler.prepare_dataframe(df, date_columns=['Date'], categorical_columns=['Road_Type'])
    profiler.describe_columns(df, ['Road_Type'])
