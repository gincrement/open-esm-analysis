# SPDX-FileCopyrightText: openmod-tracker contributors
#
# SPDX-License-Identifier: MIT

"""Get detailed information about GitLab users who interacted with repositories."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from gitlab_api import GitLabClient
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

USER_COLS = [
    "company",
    "blog",
    "location",
    "email_domain",
    "bio",
    "twitter_username",
    "repos",
    "readme",
]


def _domain_from_email(email: str | None) -> str | None:
    if not email or "@" not in email:
        return None
    return email.split("@", 1)[1]


def get_user_details_gl(
    username: str, repos: set[str], client: GitLabClient
) -> pd.DataFrame:
    """Get user details for a GitLab user by username.

    Returns user_df (index=username).
    Missing fields compared to GitHub are returned empty.
    """
    user = client.find_user_by_username(username)
    if not user:
        LOGGER.warning(f"Could not find GitLab user for username: {username}")
        return pd.DataFrame()

    # Basic profile fields
    company = user.get("organization")
    blog = user.get("website_url")
    location = user.get("location")
    bio = user.get("bio")
    public_email = user.get("public_email")
    twitter = user.get("twitter")
    job_title = user.get("job_title") or ""
    work_info = user.get("work_information") or ""
    if job_title or work_info:
        readme = " - ".join([job_title, work_info])
    else:
        readme = ""

    user_data = {
        "company": company,
        "blog": blog,
        "location": location,
        "email_domain": _domain_from_email(public_email),
        "bio": bio,
        "twitter_username": twitter,
        "repos": ",".join(sorted(repos)),
        "readme": readme,
    }

    user_df = pd.DataFrame(user_data, index=pd.Index([username], name="username"))
    return user_df


@click.command()
@click.option(
    "--user-interactions",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    help="Path to the GitLab user interactions CSV file (from get_gitlab_repo_users.py).",
    default="user_analysis/output/user_interactions_gitlab.csv",
)
@click.option(
    "--outdir",
    type=click.Path(exists=False, dir_okay=True, file_okay=False, path_type=Path),
    help="Output directory for GitLab user details files.",
    default="user_analysis/output",
)
@click.option(
    "--refresh-cache", is_flag=True, help="If set, rebuild user files from scratch."
)
def cli(
    user_interactions: Path,
    outdir: Path,
    refresh_cache: bool,
    base_url: str | None,
    token: str | None,
):
    """Collect GitLab user details for all users listed in the interactions CSV."""
    load_dotenv()
    outdir.mkdir(parents=True, exist_ok=True)

    user_details_path = outdir / "user_details_gitlab.csv"

    if user_details_path.exists() and not refresh_cache:
        existing_users = pd.read_csv(user_details_path, index_col=0)
    else:
        existing_users = pd.DataFrame(columns=USER_COLS, index=[])
        existing_users.to_csv(user_details_path)

    client = GitLabClient()

    users_df = pd.read_csv(user_interactions)
    # Only fetch for users we don't already have
    users_df = users_df[~users_df.username.isin(existing_users.index)]
    user_repo_map: dict[str, set[str]] = (
        users_df.groupby("username")["repo"].agg(lambda x: set(x)).to_dict()
    )

    LOGGER.warning(f"Collecting details for {len(user_repo_map)} unique GitLab users")

    for username, repos in tqdm(
        user_repo_map.items(), desc="Collecting GitLab user details"
    ):
        user_df = get_user_details_gl(username, repos, client)
        # Only add new users
        if not user_df.empty:
            user_df[USER_COLS].to_csv(user_details_path, mode="a", header=False)


if __name__ == "__main__":
    cli()
