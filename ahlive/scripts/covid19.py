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

# import ahlive as ah
# import pandas as pd

# df = pd.read_csv('covid19_normalized.csv').melt(
#     id_vars='location', var_name='date')
# df['date'] = pd.to_datetime(df['date'])
# df = df.loc[df['location'].isin([
#     'United States', 'China'
# ])]
# categories = df.groupby('location').max().sort_values('value').index
# df['location'] = pd.Categorical(df['location'], categories=categories)
# df = df.loc[df['date'] > '2020-03-15'].sort_values(['location', 'date'])
# df = df.loc[df['date'] < '2020-04-15']

# ahdf = ah.DataFrame(
#     df, 'date', 'value', label='location', join='cascade',
#     inline_labels='value', state_labels='date',
#     ylabel='Confirmed Cases / 100k people',
#     title='COVID-19, 7-Day Average, Normalized by Population',
#     watermark='Animated using Ahlive | Data: http://91-divoc.com/'
# )

# ahdf.add_references(x0s='min', label='Reference').animate()


# import pandas as pd
# from tropycal import tracks

# import ahlive as ah

# track_ds = tracks.TrackDataset(
#     basin='north_atlantic', source='hurdat', include_btk=True)

# season_ds = track_ds.get_season(2020)
# meta_df = season_ds.to_dataframe()
# storm_df = pd.concat(
#     season_ds.get_storm(
#         storm_id).to_dataframe().assign(**{'name': storm_name})
#     for storm_id, storm_name in zip(meta_df['id'], meta_df['name']))
# storm_df = storm_df.dropna()

# storm_df['mslp'] /= 20

# ahdf = ah.DataFrame(
#     storm_df, 'lon', 'lat', join='cascade', chart_type='trail',
#     xlim0s='fixed', ylim0s='fixed', xlim1s='fixed', ylim1s='fixed',
#     chart='scatter', label='name', c='vmax', s='mslp',
#     inline_labels='name', state_labels='date', legend=False,
#     frames=20, crs='PlateCarree', land=True, states=True, ocean=True
# )
# ahdf.animate()
