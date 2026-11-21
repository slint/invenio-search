# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""Search backend on Postgres with the ParadeDB ``pg_search`` extension.

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
The database needs the ``pg_search`` extension, for example from the
``paradedb/paradedb`` Docker image, and the first connection needs the
privileges to create it.

Text queries use ParadeDB's BM25 index with its default tokenizer, not the
analyzers of the index mappings, so relevance scores and the set of documents
a text query matches can differ from OpenSearch. Filters, permissions, facets
and sorting on fields run as exact SQL/JSON path conditions. Requests using a
query, aggregation or endpoint the backend does not implement fail with a 400
error.
"""
