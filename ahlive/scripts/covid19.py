# import pandas as pd
# import ahlive as ah

# url = (
#     'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
# )
# df = pd.read_csv(url)
# df = df.drop([
#     'UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2',
#     'Country_Region'
# ], axis=1)
# df.iloc[:, 4:] = df.iloc[:, 4:].diff(axis=1).clip(0).rolling(7, axis=1).mean()
# df.columns = df.columns.str.lower().str.rstrip('_')
# df = df.melt(
#     id_vars=['lat', 'long', 'combined_key', 'province_state'],
#     var_name='date', value_name='new_cases'
# ).dropna()
# df = df.loc[df['province_state'] == 'Illinois']
# df['date'] = pd.to_datetime(df['date'])
# df = df.loc[df['date'] > '2020-03-01']
# df['new_cases'] = df['new_cases'].round(0)
# df_sum = df.groupby('date')[['new_cases']].sum().reset_index()

# timeseries = ah.DataFrame(
#     df_sum, 'date', 'new_cases',
#     annotation_kwds={'format': '.0f'}, ytick_kwds={'format': '.0f'})
# condition = df_sum['date'].isin(
#     pd.date_range('2020-03-15', '2020-11-15', freq='1MS'))
# timeseries = timeseries.add_annotations(
#     annotations='y', delays=1, condition=condition)
# geomap = ah.DataFrame(
#     df, 'long', 'lat', state_labels='date', figsize=(24, 9),
#     xlim0s=-92, xlim1s=-86.5, ylim0s=37, ylim1s=43, clabel='',
#     interp='cubic', states=True, ocean=False, batch=True,
#     chart='scatter', c='new_cases', s='new_cases', label='combined_key',
#     projection='platecarree', plot_kwds={'alpha': 0.6}, join='overlay',
#     suptitle='New Confirmed COVID-19 Cases per Day in IL (7-Day Rolling Average)',
# )
# layout = (geomap + timeseries)
# layout.animate()
