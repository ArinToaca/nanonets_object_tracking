import streamlit as st
import plotly.express as px
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import os
VID_FOLDER = '.'
st.title('Abandoned luggage prototype')
videos_list = [el for el in os.listdir(VID_FOLDER) if 'mp4' in el]
option = st.selectbox('Select analyzed video',
                      tuple(videos_list))

csv_option_filepath = option.split('.')[0] + '.csv'
df = pd.read_csv(csv_option_filepath)
df.loc[df['events'] != 0, 'events'] = 1
st.video(option, format='video/mp4', start_time=0)
fig = px.line(df, x="timestamp", y="events", title='Events Chart', color='events')
st.write(fig)