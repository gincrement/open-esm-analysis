# SPDX-FileCopyrightText: openmod-tracker contributors
#
# SPDX-License-Identifier: MIT


"""Repository Interaction Collector.

This script fetches comprehensive repository interaction data from GitHub or GitLab
based on the repository URL, using the appropriate API client with rate limiting,
pagination, and error handling.
"""

import logging
from pathlib import Path
from urllib.parse import urlparse

import click
import pandas as pd
import util
from dotenv import load_dotenv
from github_api import GitHubRepositoryCollectorGH
from gitlab_api import GitLabRepositoryCollectorGL
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

PAGINATION_CACHE = util.read_yaml("pagination_cache", exists=False)

COLS = [
    "username",
    "interaction",
    "subtype",
    "number",
    "created",
    "closed",
    "merged",
    "repo",
    "host",
]


def detect_host(url: str) -> str | None:
    """Detect the git hosting platform from a URL.

    Args:
        url: Repository URL

    Returns:
        'gh', 'gl', or None if unknown
    """
    parsed = urlparse(url.lower())
    netloc = parsed.netloc

    if netloc.endswith("github.com"):
        return "gh"
    elif "gitlab" in netloc:
        return "gl"
    return None


@click.command()
@click.option(
    "--stats-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the CSV file containing repository URLs in the first column.",
    default="inventory/output/stats.csv",
)
@click.option(
    "--out-path",
    type=click.Path(exists=False, dir_okay=False, file_okay=True, path_type=Path),
    help="Output path for the user interactions data file.",
    default="user_analysis/output/user_interactions.csv",
)
def cli(stats_file: Path, out_path: Path):
    """CLI entry point to collect all users who interact with repositories listed in a stats file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing_users = pd.read_csv(out_path)
    else:
        existing_users = pd.DataFrame(columns=COLS, index=[])
        existing_users.to_csv(out_path, index=False)

    repos_df = pd.read_csv(stats_file, index_col="id")

    # Initialize collectors
    github_collector = GitHubRepositoryCollectorGH()
    gitlab_collector = GitLabRepositoryCollectorGL()

    for repo_id, repo in tqdm(repos_df.iterrows(), desc="Collecting users"):
        repo_url = str(repo.html_url).lower()
        host = detect_host(repo_url)

        if host is None:
            LOGGER.warning(
                f"Skipping user collection for {repo_id} ({repo_url}) - unknown host."
            )
            continue

        url_parts = urlparse(repo_url)
        repo_path = url_parts.path.strip("/")

        LOGGER.warning(f"Collecting users for {repo_path} from {host}")

        # Route to appropriate collector
        if host == "gh":
            df = github_collector.collect_repo_data(repo_path)
        elif host == "gl":
            df = gitlab_collector.collect_repo_data(repo_path)
        else:
            LOGGER.warning(f"Unsupported host: {host}")
            continue

        if df.empty:
            LOGGER.warning(f"No users found for {repo_path}.")
            continue

        # Add host column
        df["host"] = host

        existing_users = pd.concat([existing_users, df]).reindex(columns=COLS)

        # Clean up data by removing data that has been downloaded already
        # and merging rows with updated data (e.g. issues/PRs that have since been closed/merged)
        no_dups_users = (
            existing_users[existing_users["repo"] == repo_path]
            .groupby(
                ["username", "interaction", "subtype", "number", "repo", "host"],
                group_keys=False,
                dropna=False,
            )
            .apply(lambda x: x.ffill().bfill())
            .drop_duplicates()
            .dropna(subset=["username", "created", "repo"])
        )
        existing_users = pd.concat(
            [existing_users[existing_users["repo"] != repo_path], no_dups_users]
        )
        existing_users["number"] = existing_users["number"].astype("Int32")
        existing_users.sort_values(["repo", "created"]).to_csv(out_path, index=False)
        util.dump_yaml("pagination_cache", PAGINATION_CACHE)


if __name__ == "__main__":
    cli()
