# SPDX-FileCopyrightText: openmod-tracker contributors
#
# SPDX-License-Identifier: MIT

"""GitLab Repository Interaction Collector.

This script fetches repository interaction data from GitLab REST API (issues, merge requests, comments, forks, commits).
It mirrors the GitHub collector shape as closely as GitLab's API allows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import click
import pandas as pd
import requests
from dotenv import load_dotenv
from gitlab_api import GitLabClient
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

load_dotenv()

COLS = [
    "username",
    "interaction",
    "subtype",
    "number",
    "created",
    "closed",
    "merged",
    "repo",
]


class GitLabRepositoryCollector:
    """Collects GitLab repository activity using REST API endpoints."""

    def __init__(self, token: str | None = None, base_url: str | None = None):
        """Initialize collector with underlying GitLab client."""
        self.client = GitLabClient(token=token, base_url=base_url)

    def _parse_username(self, author: dict[str, Any] | None) -> str | None:
        if not author:
            return None
        # Prefer username, fallback to name
        return author.get("username") or author.get("name")

    def collect_repo_data(self, repo_path: str) -> pd.DataFrame:
        """Analyze a GitLab project and return activity data.

        Args:
            repo_path: Full path of project (e.g. group/subgroup/project)
        """
        results: list[dict[str, Any]] = []
        try:
            self.client.get_project(repo_path)  # Validate project exists
        except requests.HTTPError as e:
            LOGGER.error(f"Failed to access GitLab project {repo_path}: {e}")
            return pd.DataFrame(columns=COLS)
        # Issues + comments
        for issue in self.client.list_issues(repo_path):
            iid = issue.get("iid")
            results.append(
                {
                    "interaction": "issue",
                    "subtype": "author",
                    "number": iid,
                    "username": self._parse_username(issue.get("author")),
                    "created": issue.get("created_at"),
                    "closed": issue.get("closed_at"),
                }
            )
            for note in self.client.list_issue_notes(repo_path, iid):
                results.append(
                    {
                        "interaction": "issue",
                        "subtype": "comment",
                        "number": iid,
                        "username": self._parse_username(note.get("author")),
                        "created": note.get("created_at"),
                    }
                )

        # Merge requests + comments
        for mr in self.client.list_merge_requests(repo_path):
            iid = mr.get("iid")
            results.append(
                {
                    "interaction": "pr",  # keep column naming consistent with GitHub (treated as PRs)
                    "subtype": "author",
                    "number": iid,
                    "username": self._parse_username(mr.get("author")),
                    "created": mr.get("created_at"),
                    "closed": mr.get("closed_at") if not mr.get("merged_at") else None,
                    "merged": mr.get("merged_at"),
                }
            )
            for note in self.client.list_mr_notes(repo_path, iid):
                results.append(
                    {
                        "interaction": "pr",
                        "subtype": "comment",
                        "number": iid,
                        "username": self._parse_username(note.get("author")),
                        "created": note.get("created_at"),
                    }
                )

        # Forks
        for fork in self.client.list_forks(repo_path):
            # fork project payload contains namespace info; best effort username
            ns = fork.get("namespace") or {}
            username = ns.get("full_path") or fork.get("path_with_namespace")
            results.append(
                {
                    "interaction": "fork",
                    "username": username,
                    "created": fork.get("created_at"),
                }
            )

        # Stargazers
        for starrer in self.client.list_stargazers(repo_path):
            # starrer payload contains user info; best effort username
            user = starrer.get("user") or {}
            username = user.get("username")
            results.append(
                {
                    "interaction": "stargazer",
                    "username": user.get("username"),
                    "created": starrer.get("starred_since"),
                }
            )
        # Commits
        for commit in self.client.list_commits(repo_path):
            username = commit.get("author_email") or commit.get("author_name")
            results.append(
                {
                    "interaction": "commit",
                    "subtype": "default_branch",  # GitLab endpoint returns default ordering; we don't distinguish here
                    "username": username,
                    "created": commit.get("created_at"),
                }
            )
        if not results:
            LOGGER.info(f"No interactions found for {repo_path}")
            return pd.DataFrame(columns=COLS)

        df = pd.DataFrame(results).assign(repo=repo_path)
        # Harmonize columns
        for col in ("number", "closed", "merged", "subtype"):
            if col not in df:
                df[col] = pd.Series(dtype="float") if col == "number" else None
        # Parse timestamps (keep naive)
        for ts_col in ["created", "closed", "merged"]:
            if ts_col in df:
                df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.tz_localize(None)
        if "number" in df:
            df["number"] = df["number"].astype("Int16")
        return df[COLS]


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
    help="Output path for the user interactions data file (GitLab).",
    default="user_analysis/output/user_interactions_gitlab.csv",
)
def cli(stats_file: Path, out_path: Path):
    """CLI entry point to collect GitLab users who interact with repositories listed in a stats file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = pd.read_csv(out_path)
    else:
        existing = pd.DataFrame(columns=COLS, index=[])
        existing.to_csv(out_path, index=False)

    repos_df = pd.read_csv(stats_file, index_col="id")
    collector = GitLabRepositoryCollector()

    for repo_id, repo in tqdm(repos_df.iterrows(), desc="Collecting GitLab users"):
        repo_url = str(repo.html_url).lower()
        url_parts = urlparse(repo_url)
        if "gitlab" not in url_parts.netloc:
            continue
        repo_path = url_parts.path.strip("/")
        LOGGER.warning(f"Collecting users for {repo_path}")

        df = collector.collect_repo_data(repo_path)
        if df.empty:
            continue

        existing = pd.concat([existing, df]).reindex(columns=COLS)
        # De-duplicate per repo
        no_dups = (
            existing[existing["repo"] == repo_path]
            .groupby(
                ["username", "interaction", "subtype", "number", "repo"],
                group_keys=False,
                dropna=False,
            )
            .apply(lambda x: x.ffill().bfill())
            .drop_duplicates()
            .dropna(subset=["username", "created", "repo"])
        )
        existing = pd.concat([existing[existing["repo"] != repo_path], no_dups])
        existing["number"] = existing["number"].astype("Int32")
        existing.sort_values(["repo", "created"]).to_csv(out_path, index=False)


if __name__ == "__main__":
    cli()
