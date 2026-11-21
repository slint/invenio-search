# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""Flask extension that points invenio-search's client at Postgres."""

from .connection import ParadeDBConnection

SCHEME = "paradedb://"


def _url(hosts):
    if isinstance(hosts, (list, tuple)) and len(hosts) == 1:
        hosts = hosts[0]
    return hosts if isinstance(hosts, str) and hosts.startswith(SCHEME) else None


class InvenioSearchParadeDB:
    """Use ParadeDB when ``SEARCH_HOSTS`` is a ``paradedb://`` URL.

    ``paradedb://`` alone uses the application database
    (``SQLALCHEMY_DATABASE_URI``); ``paradedb://user:pass@host:5432/db`` uses
    that database. Any other ``SEARCH_HOSTS`` leaves invenio-search alone.
    """

    def __init__(self, app=None):
        """Extension initialization."""
        if app:
            self.init_app(app)

    def init_app(self, app):
        """Flask application initialization."""
        url = _url(app.config.get("SEARCH_HOSTS"))
        if url is None:
            return
        rest = url[len(SCHEME) :]
        config = dict(app.config.get("SEARCH_CLIENT_CONFIG") or {})
        # invenio-search passes SEARCH_CLIENT_CONFIG to the OpenSearch client,
        # so a connection class is all it takes; no HTTP host is involved.
        config["connection_class"] = ParadeDBConnection
        if rest.strip("/"):
            config["dsn"] = "postgresql://" + rest
        app.config["SEARCH_CLIENT_CONFIG"] = config
        app.config["SEARCH_HOSTS"] = None
        app.extensions["invenio-search-paradedb"] = self
