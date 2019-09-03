import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from pprint import pprint
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_categorical, is_datetime64_any_dtype, is_datetime64tz_dtype
import numpy as np
import logging
from collections import defaultdict, namedtuple
from matplotlib import dates
import datetime

class DataProfiler:
    """Class for data profiling"""

    graph_per_category = {'numerical': ['hist', 'density', 'box'],
                          'categorical': ['bar'],
                          'datetime': ['hist']}
    logger = logging.getLogger('DataProfiler')
    descriptive_res = namedtuple('Columns_Summary', ('summary_stats', 'graphs', 'outliers'))

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
            raise ValueError('Convert variable to one of {}'.format(' or '.join(available_convertions)))
        data = dataframe[column]
        if convert_to == 'datetime':
            data = pd.to_datetime(data)
        elif convert_to == 'categorical':
            data = data.astype('category')
            if data.isnull().sum() > 0:
                data = data.cat.add_categories('Missing')
                data.fillna('Missing', inplace=True)
        return data

    @plt.FuncFormatter
    def _num_to_date(self, x, pos):
        """Custom formater to turn floats into date"""
        return dates.num2date(x).strftime('%Y-%m-%d')

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
                         identify_outliers=True, exclude_outliers_from_graph=False,
                         show_graphs=False):
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
        identify_outliers
            If true finds outliers in each column to plot if numerical
        exclude_outliers_from_graph
            If true it excludes the detected outliers from the graph
        """
        self._check_data(dataframe)
        try:
            assert isinstance(columns, list)
        except AssertionError:
            raise ValueError('A list of columns needs to be passed')
        columns_to_summary_stats = {}
        columns_to_graphs = defaultdict(dict)
        column_to_outliers = {}
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
                column_to_outliers[column] = outliers_ind
            #plot graphs
            if exclude_outliers_from_graph:
                columns_to_graphs[column] = self.plot_descriptive_graphs_for_column(dataframe, column,
                                                                                    outliers_ind=outliers_ind,
                                                                                    show=show_graphs)
            else:
                columns_to_graphs[column] = self.plot_descriptive_graphs_for_column(dataframe, column)
        res = self.descriptive_res(columns_to_summary_stats, columns_to_graphs, column_to_outliers)
        return res

    def summary_stats(self, dataframe, column, percentiles=[.25,.50,.75], print_summary=True):
        """
        :param dataframe: Input data
        :param column: Column to describe
        :param percentiles: Percentiles for numerical column (range 0 to 1)
        :return: Summary statistics for column
        """
        column_stats = dataframe[column].describe(percentiles=percentiles)
        if print_summary:
            print('\n\n{} Summary Statistics:\n'.format(column))
            pprint(column_stats)
        return column_stats

    def identify_outliers(self, dataframe, column, column_stats=pd.DataFrame()):
        """
        :param dataframe: Input dataframe
        :param column: Input column
        :param column_stats: If passed avoids calculating summary stats for data.
        Used to find outliers
        :return: boolean array for indexing outlier values
        """
        if not column_stats.empty:
            column_stats = self.summary_stats(dataframe, column, print_summary=False)
        if 'mean' in column_stats.index:  # check that mean is in stats --> it's a numeric column
            conf_inter = [column_stats.loc['mean'] - (1.96 * column_stats.loc['std']), \
                          column_stats.loc['mean'] + (1.96 * column_stats.loc['std'])]
            outliers_ind = (dataframe[column] < conf_inter[0]) | (dataframe[column] > conf_inter[1])
            return outliers_ind
        else:
            self.logger.warning('{} is not numerical, outliers cannot be calculated'.format(
                column))
        return np.zeros(dataframe.shape[0], dtype=bool)

    def plot_descriptive_graphs_for_column(self, dataframe, column, outliers_ind=None,
                                           show=False):
        """
        :param dataframe: Input dataframe
        :param column: Column to plot
        :param outliers_ind: Boolean array for indexing outliers
        :param show: If to show graphs when running the code
        :return: Mapping of columns plotted to graphs types
        """
        graph_type_to_graph = {}
        data_to_plot = dataframe[column]
        if is_numeric_dtype(dataframe[column]):
            cat = 'numerical'
            if isinstance(outliers_ind, np.ndarray):
                data_to_plot = dataframe.loc[~outliers_ind, column]
        elif is_categorical(dataframe[column]):
            cat = 'categorical'
            data_to_plot = dataframe[column].value_counts()
        elif is_datetime64_any_dtype(dataframe[column]) or is_datetime64tz_dtype(dataframe[column]):
            cat = 'datetime'
            data_to_plot = pd.Series(dates.date2num(data_to_plot))
        else:
            self.logger.warning('''Column "{}" could not be plotted because of {} (generic) type.
    Please convert it to a categorical, date or numerical type. String columns cannot be plotted!'''.format(
                column, dataframe[column].dtype))
            return {}
        n_graphs = len(self.graph_per_category[cat])
        fig, axes = plt.subplots(int(np.ceil(n_graphs/2)), int(np.ceil(n_graphs/2)),
                                 squeeze=0)
        fig.suptitle(column, fontsize='large')
        fig.tight_layout()
        plt.subplots_adjust(top=0.82)
        axes = axes.flatten()
        for i,graph in enumerate(self.graph_per_category[cat]):
            plot = data_to_plot.plot(kind=graph, ax=axes[i])
            axes[i].set_title(graph)
            if cat == 'datetime':
                axes[i].xaxis.set_major_formatter(self._num_to_date)
            graph_type_to_graph[graph] = axes[i]
        if show:
            plt.show(block=False)
        return graph_type_to_graph

    def distribution_over_time_for_columns(self, dataframe, columns_to_plot,
                                           date_column, start_date=None,
                                           end_date=None, frequency='day',
                                           frequency_multiplier=1):
        """
        :param dataframe: Dataframe to plot
        :param columns_to_plot: List of columns to plot
        :param date_column: The date column (does not need to be converted to date type)
        :param start_date: Start plotting from this date
        :param end_date: Limit plot to this date
        :param frequency: One of: day,week,month,year. Frequency at which the data is plotted
        :param frequency_multiplier: Integer, modifies frequency by that integer (e.g. if frequency=day and multiplier=2 final frequency is 2 days)
        :return: Dictionary of column plotted to matplotlib plot
        """
        frequencies = ['day', 'week', 'month', 'year']
        freq_to_pandas_freq = {'day':'d', 'week':'w', 'month':'m', 'year':'y'}
        date = dataframe[date_column]
        try:
            assert frequency in frequencies
        except AssertionError:
            raise ValueError('Can only plot at these frequencies: {}'.format(
                ' , '.join(frequencies)))
        try:
            assert isinstance(frequency_multiplier, int)
        except AssertionError:
            raise ValueError('Please pass an integer value for frequency multiplier')
        is_date = is_datetime64_any_dtype(date) or is_datetime64tz_dtype(date)
        if not is_date:
            try:
                dataframe[date_column] = self.convert_column(dataframe, date_column, convert_to='datetime')
            except:
                raise('{} is not a valid date column'.format(date_column))
        if not start_date:
            start_date = date.min()
        if not end_date:
            end_date = date.max()
        columns_and_graph = {}
        dataframe[date_column] = dataframe[date_column].dt.date
        try:
            assert isinstance(start_date, datetime.date)
        except AssertionError:
            raise ValueError('Start date {} is not a valid date'.format(start_date))
        try:
            assert isinstance(end_date, datetime.date)
        except AssertionError:
            raise ValueError('End date {} is not a valid date'.format(end_date))
            end_date = end_date.date()
        if hasattr(start_date, 'date'):
            start_date = start_date.date()
        if hasattr(end_date, 'date'):
            end_date = end_date.date()
        period_to_plot = (dataframe[date_column] >= start_date) | \
                         (dataframe[date_column] <= end_date)
        dataframe = dataframe.loc[period_to_plot].set_index(date_column)
        for column in columns_to_plot:
            dataframe_to_plot = dataframe[[column]]
            if not (is_numeric_dtype(dataframe_to_plot[column]) or is_categorical(dataframe_to_plot[column])):
                self.logger.warning('Skipping {} of dtype {}. Only numerical or categorical data can be plotted'.format(
                    column, dataframe_to_plot[column].dtype
                ))
                continue
            if is_categorical(dataframe_to_plot[column]):
                dataframe_to_plot.index = pd.to_datetime(dataframe_to_plot.index)
                df_fin = dataframe_to_plot.groupby([pd.Grouper(freq='{}{}'.format(frequency_multiplier,
                                                   freq_to_pandas_freq[frequency])), column]
                                    ).size()
                plot = df_fin.unstack(level=1).plot.bar()
                every_nth = 3
                for n, label in enumerate(plot.xaxis.get_ticklabels()):
                    if n % every_nth != 0:
                        label.set_visible(False)
            else:
                plot = dataframe_to_plot.plot()
            columns_and_graph[column] = plot
        return columns_and_graph

if __name__ == '__main__':
    df = pd.read_csv('../data/Acc.csv')
    print(df.columns)
    profiler = DataProfiler()
    dataframe = profiler.prepare_dataframe(df, date_columns=['Date'], categorical_columns=['Road_Type'])
    res = profiler.describe_columns(df, ['Date', 'Number_of_Vehicles', 'Junction_Detail'], show_graphs=False)
    print(res)
    plot = profiler.distribution_over_time_for_columns(dataframe, ['Number_of_Casualties', 'Road_Type'], 'Date',
                                                   start_date=pd.to_datetime(dataframe['Date'].min() + datetime.timedelta(100)))
