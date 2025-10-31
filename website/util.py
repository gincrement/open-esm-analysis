# SPDX-FileCopyrightText: openmod-tracker contributors
#
# SPDX-License-Identifier: MIT


"""Utility functions to support rendering the web app."""

from pathlib import Path

import pandas as pd
import streamlit as st


def is_datetime_column(series: pd.Series) -> bool:
    """Check if a column is datetime."""
    return pd.api.types.is_datetime64_any_dtype(series)


def is_numeric_column(series: pd.Series) -> bool:
    """Check if a column is numeric."""
    return pd.api.types.is_numeric_dtype(series)


def is_categorical_column(series: pd.Series) -> bool:
    """Check if a column should be treated as categorical."""
    return isinstance(series.dtype, pd.StringDtype | pd.CategoricalDtype)


def is_list_column(series: pd.Series) -> bool:
    """Check if a column contains lists of items."""
    return all(series.dropna().apply(lambda x: isinstance(x, list)))


def nan_filter(col: pd.Series) -> pd.Series:
    """Remove rows with NaNs in column.

    Args:
        col (pd.Series): Column potentially containing NaNs

    Returns:
        pd.Series: Filtered `col`.
    """
    return col.notna()


def get_state(key, default=None):
    """Get a value from Streamlit session state, or set it to default if not present."""
    return st.session_state.get(key, default)


def set_state(key, value, subkey: str | None = None):
    """Set a value in Streamlit session state."""
    if subkey is None:
        st.session_state[key] = value
    else:
        st.session_state[key][subkey] = value


def init_state(key, default, subkey: str | None = None):
    """Initialise a key in Streamlit session state."""
    if subkey is None:
        set_state(key, get_state(key, default))
    else:
        set_state(key, get_state(key, default)[subkey], subkey)


@st.cache_data
def map_repo_to_tool(user_stats_df: pd.DataFrame, repo_col: str) -> list[dict]:
    """Map repository names to tool names.

    Args:
        user_stats_df (pd.DataFrame): User stats dataframe.
        repo_col (str): Name of the column containing repository names.

    Returns:
        list[dict]: List of dictionaries mapping repository names to tool names.
    """
    available_repos = set(
        (",".join(user_stats_df[repo_col].str.lower().values)).split(",")
    )
    tools_df = pd.read_csv(
        Path(__file__).parent.parent / "inventory" / "output" / "filtered.csv"
    )
    urls = {repo: "https://github.com/" + repo.lower() for repo in available_repos}
    repo_to_tool_map = [
        {
            "repo": repo,
            "name": tools_df.loc[tools_df.url == urls[repo], "name"]
            .item()
            .split(",")[0],
        }
        for repo, url in urls.items()
        if url in tools_df.url.values
    ]
    return repo_to_tool_map
