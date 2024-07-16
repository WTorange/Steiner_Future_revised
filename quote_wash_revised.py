from datetime import datetime, timedelta
import pandas as pd

import logging


def get_logger_quote_wash(name, debug=False):
    """
    初始化一个日志记录器。

    参数:
    - name (str): 日志记录器的名称。

    返回:
    - logger (logging.Logger): 配置好的日志记录器。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        handler = logging.FileHandler('log_file.log')
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
class DataProcessor:
    def __init__(self, future_index, opening_hours_file='future_information.csv', debug=False):
        self.future_index = future_index
        self.opening_hours_df = pd.read_csv(opening_hours_file)
        self.logger = get_logger_quote_wash('QuoteProcessor', debug)
        self.contract_hours_cache = self._cache_opening_hours()

    def _cache_opening_hours(self):
        return dict(zip(self.opening_hours_df['code'], self.opening_hours_df['hours']))

    def quote_wash(self, day_quote, future_index):
        if day_quote is None or len(day_quote) == 0:
            return None

        required_columns = ['datetime', 'date', 'time']
        missing_columns = [col for col in required_columns if col not in day_quote.columns]

        day_quote['datetime'] = pd.to_datetime(day_quote['datetime'])
        if 'date' in missing_columns or 'time' in missing_columns:
            day_quote['time'] = day_quote['datetime'].dt.strftime('%H%M%S%f').str.slice(0, 9)
            day_quote['date'] = day_quote['datetime'].dt.strftime('%Y%m%d')

        contract_code = future_index
        contract_hours = self.contract_hours_cache.get(contract_code)

        if contract_hours is None:
            self.logger.warning(f"No opening hours found for {contract_code}")
            return None

        def time_to_ms(hour, minute, second=0, ms=0):
            return hour * 3600 * 1000 + minute * 60 * 1000 + second * 1000 + ms

        def filter_trading_data(row):
            trading_time = str(row['time']).zfill(9)
            trading_total_ms = time_to_ms(int(trading_time[:2]), int(trading_time[2:4]), int(trading_time[4:6]),
                                          int(trading_time[6:9]))

            for period in contract_hours.split():
                start, end = period.split('-')
                start_hour, start_minute = map(int, start.split(':'))
                end_hour, end_minute = map(int, end.split(':'))

                start_total_ms = time_to_ms(start_hour, start_minute)
                end_total_ms = time_to_ms(end_hour, end_minute) if end_hour >= start_hour else time_to_ms(end_hour + 24,
                                                                                                          end_minute)

                if start_total_ms <= trading_total_ms <= end_total_ms:
                    return True

            return False

        day_quote = day_quote[day_quote.apply(filter_trading_data, axis=1)]
        return day_quote

    def resample_data(self, data, contract_code, freq='500L'):
        data = data.copy()
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['date'] = data['datetime'].dt.date

        resampled_data = []
        opening_hours_str = self.contract_hours_cache.get(contract_code)
        if opening_hours_str is None:
            self.logger.warning(f"No opening hours found for {contract_code}")
            return None

        for date in data['date'].unique():
            periods = opening_hours_str.split()
            date_data = data[data['date'] == date].copy()
            original_datetime = date_data['datetime'].copy()
            date_data['resample_time'] = original_datetime

            for period in periods:
                start_time_str, end_time_str = period.split('-')
                start_datetime = datetime.strptime(date.strftime('%Y-%m-%d') + start_time_str, '%Y-%m-%d%H:%M')
                end_datetime = datetime.strptime(date.strftime('%Y-%m-%d') + end_time_str, '%Y-%m-%d%H:%M')

                if end_datetime < start_datetime:
                    end_datetime += timedelta(days=1)

                all_times = pd.date_range(start=start_datetime, end=end_datetime, freq=freq)
                resampled = date_data.set_index('resample_time').reindex(all_times, method='pad')

                resampled['date'] = date
                resampled['resample_time'] = resampled.index
                resampled = resampled[resampled['last_prc'].notna()]
                resampled_data.append(resampled)

        return pd.concat(resampled_data, ignore_index=True)

    def process(self, day_quote, future_index):
        washed_quote = self.quote_wash(day_quote, future_index)
        if washed_quote is not None:
            resampled_quote = self.resample_data(washed_quote, future_index)
            return resampled_quote
        return None


if __name__ == '__main__':
    future_index = 'RU'
    processor = DataProcessor(future_index)

    day_quote = pd.read_csv('test1.csv')
    cleaned_quote = processor.process(day_quote, future_index)
    cleaned_quote.to_csv("check.csv")