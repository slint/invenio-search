# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""Run Invenio's search indices on Postgres with ParadeDB ``pg_search``.

For small instances (local development, demos, PR previews), not production.
Install the package and set ``SEARCH_HOSTS = "paradedb://"``; the OpenSearch
client stays, and :class:`.connection.ParadeDBConnection` answers its
requests from Postgres.
"""

from .connection import ParadeDBConnection
from .ext import InvenioSearchParadeDB

__version__ = "0.1.0.dev0"

__all__ = ("InvenioSearchParadeDB", "ParadeDBConnection", "__version__")
