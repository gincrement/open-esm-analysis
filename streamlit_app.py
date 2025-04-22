#
# Update package history based on latest available statistics from ecosyste.ms and anaconda
#
# (C) Open Energy Transition (OET)
# License: MIT / CC0 1.0
#

# define the path of the CSV file listing the packages to assess
url_api = 'https://ost.ecosyste.ms/api/v1/projects/esd'

# import required packages
import os.path
import json
import io
from datetime import datetime, timedelta
from time import gmtime, strftime

import streamlit as st

try:
    import pandas as pd
except:
    !pip install pandas
    import pandas as pd

try:
    from urllib.request import urlopen
except:
    !pip install urllib
    from urllib.request import urlopen
try:
    import requests
except:
    !pip install requests
    import requests

try:
    import itables
    from itables import init_notebook_mode
    from itables import to_html_datatable
except:
    !pip install itables
    import itables
    from itables import init_notebook_mode
    from itables import to_html_datatable

# define variables
names = []
urls = []
descriptions = []
categories = []
sub_categories = []
languages = []
licenses = []
download_counts = []
total_dependent_repos_counts = []
stars = []
citations = []
forks = []
contributors = []
develop_distr_scores = []
past_year_issues_counts = []
creates = []
updates = []

# get the JSON file from the ost.ecosyste.ms
json_url = url_api
r = requests.get(json_url)
all_data = r.json()

for i in range(len(all_data)):
    json_data = all_data[i]
    package_downloads = 0
    dependent_repos_count = 0
    latest_release_published_at = None
    for package_manager in range(len(json_data['packages'])):
        if json_data['packages'][package_manager]['downloads']:
                if json_data['packages'][package_manager]['downloads_period'] == "last-month":
                    package_downloads += json_data['packages'][package_manager]['downloads']

        if json_data['packages'][package_manager]['dependent_repos_count']:
                dependent_repos_count += json_data['packages'][package_manager]['dependent_repos_count']

        if latest_release_published_at is None or latest_release_published_at < json_data['packages'][package_manager]['latest_release_published_at']:
            latest_release_published_at = json_data['packages'][package_manager]['latest_release_published_at']

    if package_downloads:
        download_counts.append(package_downloads)
    else:
        download_counts.append(0)

    # store necessary details
    names.append(json_data['name'])
    urls.append(json_data['url'])
    descriptions.append(json_data['description'])
    categories.append(json_data['category'])
    sub_categories.append(json_data['sub_category'])
    languages.append(json_data['language'])
    licenses.append(json_data['repository']['license'])
    total_dependent_repos_counts.append(dependent_repos_count)
    stars.append(json_data['repository']['stargazers_count'])
    citations.append(json_data['total_citations'])
    forks.append(json_data['repository']['forks_count'])
    contributors.append(json_data['repository']['commit_stats']['total_committers'])
    develop_distr_scores.append(("%.3f" % json_data['commits']['dds']))
    past_year_issues_counts.append(json_data['issues_stats']['past_year_issues_count'])
    creates.append(datetime.strptime(json_data['repository']['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y/%m'))
    # updates.append(datetime.strptime(json_data['repository']['updated_at'], '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y/%m'))
    updates.append(datetime.strptime(latest_release_published_at, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y/%m'))

# create a dataframe containing all collected data
df = pd.DataFrame()
df['Project Name'] = names
df['Project Name'] = names
df['Category'] = categories
df['Sub Category'] = sub_categories
df['Created'] = creates
df['Updated'] = updates
df['License'] = licenses
df['Language'] = languages
df['Citations'] = citations
df['Stars'] = stars
df['Contribs'] = contributors
df['DDS'] = develop_distr_scores
df['Forks'] = forks
df['Dependents'] = total_dependent_repos_counts
df['PM Downloads'] = download_counts
df['PY Issues']= past_year_issues_counts

# adjust some details
df.loc[df['Project Name'] == 'Antares Simulator', 'License'] = 'mpl-2.0'
df.loc[df['Project Name'] == 'FINE', 'License'] = 'mit'
df.loc[df['Project Name'] == 'Minpower', 'License'] = 'mit'
df.loc[df['Project Name'] == 'pandapower', 'License'] = 'bsd-3-clause'
df.loc[df['Project Name'] == 'switch-model', 'License'] = 'apache-2.0'
df.loc[df['Project Name'] == 'Temoa', 'Language'] = 'Python'
df.loc[df['Project Name'] == 'PyPowSyBl', 'Language'] = 'Python'

# delete some columns not needed yet
df.drop(columns=[
    'Category', 'Sub Category', 'Language',
], axis=1, errors='ignore', inplace=True)

# show the nice table
html(
#    itables.show(
    to_html_datatable(
        # df_extract.loc[:, df_extract.columns != 'Repository'],
        df,
        #buttons=["copyHtml5", "csvHtml5", "excelHtml5"],
        #lengthMenu=[25, 50],
        #order=[[0, "asc"]]
    ),
)

# Remark:
#   Contribs .. contributors
#   DDS ... development distribution score (the smaller the number the better; 0 means no data available)
#   PM .. previous month (0 means either no downloads or not tracked/shared from the repository owner)
#   PY .. previous year (0 means either no issues or not tracked/shared from the repository owner)

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
