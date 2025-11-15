import pandas as pd
import numpy as np

def load_and_prepare_data():

    personal = pd.read_csv('/home/ililo/archive/spotify_data.csv')
    personal['ts'] = pd.to_datetime(personal['ts']).dt.tz_convert("Africa/Addis_Ababa")
    personal = personal[['ts', 'ms_played', 'master_metadata_track_name', 'master_metadata_album_artist_name', 'master_metadata_album_album_name', 'reason_start', 'reason_end', 'shuffle', 'skipped']]
    personal['master_metadata_album_artist_name'] = personal['master_metadata_album_artist_name'].str.lower()
    personal['master_metadata_track_name'] = personal['master_metadata_track_name'].str.lower()
    personal['id'] = personal['master_metadata_album_artist_name'] + '_' + personal['master_metadata_track_name']

    metadata = pd.read_csv('/home/ililo/spotify-metadata/clean_metadata.csv')

    merged_final = pd.merge(metadata, personal, on='id', how='inner')

    return merged_final
