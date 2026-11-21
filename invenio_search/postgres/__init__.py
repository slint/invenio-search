# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""Search backend on plain Postgres.

For small instances (local development, demos, test environments) that should
run without an OpenSearch cluster. The client keeps talking the OpenSearch
API; :class:`~invenio_search.postgres.connection.PostgresConnection` answers
it from Postgres instead of sending HTTP requests:

.. code-block:: python

    SEARCH_CLIENT_CONFIG = {
        "connection_class": "invenio_search.postgres.connection:PostgresConnection",
    }

It stores the indices in the ``invenio_search`` schema of the application
database (``SQLALCHEMY_DATABASE_URI``), or of ``SEARCH_POSTGRES_DSN`` if set.

Documents are analyzed with the analyzers of the index settings
(:mod:`.analysis`) into postings tables (:mod:`.schema`), and text queries are
matched and scored with BM25 per shard like Lucene does (:mod:`.scoring`,
:mod:`.lucene`). Filters run as exact SQL/JSON path conditions
(:mod:`.query`). Requests using a query, aggregation or endpoint the backend
does not implement fail with a 400 error.
"""
