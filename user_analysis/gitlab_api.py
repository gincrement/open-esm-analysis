# SPDX-FileCopyrightText: openmod-tracker contributors
#
# SPDX-License-Identifier: MIT

"""Lightweight GitLab API client using requests.

Supports GitLab.com by default and self-hosted GitLab instances via GITLAB_BASE_URL.
Authentication via a personal access token provided in GITLAB_TOKEN.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests
import util
from dotenv import load_dotenv

load_dotenv()
LOGGER = logging.getLogger(__name__)

# Default base API URL for GitLab.com REST v4
DEFAULT_BASE_URL = "https://gitlab.com/api/v4"

PAGINATION_CACHE = util.read_yaml("gitlab_pagination_cache", exists=False)


@dataclass
class GitLabRateLimit:
    """Represents simplistic rate handling window.

    GitLab does not return a single global core limit like GitHub.
    We conservatively sleep between paginated requests when a 429 is hit.
    """

    sleep_seconds: float = 1.0


class GitLabClient:
    """Minimal GitLab REST API client with pagination and basic retries."""

    def __init__(self, token: str | None = None, base_url: str | None = None):
        """Create a GitLab client.

        Args:
            token: Personal access token (PAT) for authentication. If not provided,
                will attempt to read from environment variable ``GITLAB_TOKEN``.
            base_url: Override for self-hosted GitLab API base (e.g. https://gitlab.example.com/api/v4).
                If omitted, uses environment variable ``GITLAB_BASE_URL`` or the public gitlab.com API.
        """
        self.base_url = (
            base_url or os.environ.get("GITLAB_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.session = requests.Session()
        headers = {"Accept": "application/json"}
        token = token or os.environ.get("GITLAB_TOKEN")
        if token:
            LOGGER.info("Using provided GitLab token for authentication")
            # Private-token is acceptable, also 'Authorization: Bearer <token>' works for PATs
            headers["PRIVATE-TOKEN"] = token
        else:
            LOGGER.warning("No GitLab token provided; proceeding unauthenticated")
        self.session.headers.update(headers)
        self.rate = GitLabRateLimit()

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = self.session.request(method, url, timeout=120, **kwargs)
            resp.raise_for_status()
        except Exception as e:
            LOGGER.error(f"GitLab API request error: {e}")
            pass
        return resp

    def _paginate(
        self, path: str, params: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Paginate a GitLab list endpoint, yielding items."""
        page = PAGINATION_CACHE.get(path, 0)
        params = params.copy() if params else {}
        params.setdefault("per_page", 100)
        while True:
            page += 1
            params["page"] = page
            resp = self._request("GET", path, params=params)
            items = resp.json()
            if not isinstance(items, list) or not items:
                break
            yield from items
            # Stop when fewer than per_page returned
            if len(items) < params["per_page"]:
                break
        if page > 1:
            PAGINATION_CACHE[path] = page
            util.dump_yaml("gitlab_pagination_cache", PAGINATION_CACHE)

    @staticmethod
    def encode_project_path(path: str) -> str:
        """URL-encode a project path (namespace/project) for use as :id in /projects/:id endpoints."""
        # GitLab expects full path URL-encoded, e.g. group/subgroup%2Fproject
        return quote(path, safe="")

    # --- Project-level endpoints ---

    def get_project(self, full_path: str) -> dict[str, Any]:
        """Retrieve a project record by its full path (namespace/project)."""
        return self._request(
            "GET", f"projects/{self.encode_project_path(full_path)}"
        ).json()

    def list_issues(self, full_path: str) -> Iterable[dict[str, Any]]:
        """Iterate issues for a project (ascending by creation time)."""
        project = self.encode_project_path(full_path)
        return self._paginate(
            f"projects/{project}/issues",
            params={"order_by": "created_at", "sort": "asc"},
        )

    def list_issue_notes(
        self, full_path: str, issue_iid: int
    ) -> Iterable[dict[str, Any]]:
        """Iterate issue notes (comments) for an issue IID within a project."""
        project = self.encode_project_path(full_path)
        return self._paginate(f"projects/{project}/issues/{issue_iid}/notes")

    def list_merge_requests(self, full_path: str) -> Iterable[dict[str, Any]]:
        """Iterate merge requests for a project (ascending by creation time)."""
        project = self.encode_project_path(full_path)
        return self._paginate(
            f"projects/{project}/merge_requests",
            params={"order_by": "created_at", "sort": "asc"},
        )

    def list_mr_notes(self, full_path: str, mr_iid: int) -> Iterable[dict[str, Any]]:
        """Iterate merge request notes (comments) for an MR IID within a project."""
        project = self.encode_project_path(full_path)
        return self._paginate(f"projects/{project}/merge_requests/{mr_iid}/notes")

    def list_forks(self, full_path: str) -> Iterable[dict[str, Any]]:
        """Iterate forks for a project."""
        project = self.encode_project_path(full_path)
        return self._paginate(f"projects/{project}/forks")

    def list_commits(self, full_path: str) -> Iterable[dict[str, Any]]:
        """Iterate commits for a project (default branch first by GitLab ordering)."""
        project = self.encode_project_path(full_path)
        return self._paginate(f"projects/{project}/repository/commits")

    def list_stargazers(self, full_path: str) -> Iterable[dict[str, Any]]:
        """Get all project starrers."""
        project = self.encode_project_path(full_path)
        return self._paginate(f"projects/{project}/starrers")

    # NOTE: GitLab does not expose stargazers list via public API; skipping

    # --- User endpoints ---

    def find_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Find a user by username (returns the first matching user or None)."""
        resp = self._request("GET", "users", params={"username": username}).json()
        if isinstance(resp, list) and resp:
            return self.get_user(resp[0]["id"])
        return None

    def get_user(self, user_id: int) -> dict[str, Any]:
        """Get a user by numeric ID."""
        return self._request("GET", f"users/{user_id}").json()
