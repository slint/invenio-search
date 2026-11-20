# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""Search backend on Postgres with the ParadeDB ``pg_search`` extension.

For small instances (local development, demos, test environments) that should
run without an OpenSearch cluster. The client keeps talking the OpenSearch
API; :class:`~invenio_search.paradedb.connection.ParadeDBConnection` answers
it from Postgres instead of sending HTTP requests:

.. code-block:: python

    SEARCH_CLIENT_CONFIG = {
        "connection_class": "invenio_search.paradedb.connection:ParadeDBConnection",
    }

It stores the indices in the ``invenio_search`` schema of the application
database (``SQLALCHEMY_DATABASE_URI``), or of ``SEARCH_PARADEDB_DSN`` if set.
The database needs the ``pg_search`` extension, for example from the
``paradedb/paradedb`` Docker image, and the first connection needs the
privileges to create it.

Text queries use ParadeDB's BM25 index, filters run as exact SQL/JSON path
conditions. Analyzers from the index mappings are not applied: all text uses
ParadeDB's default tokenizer. Requests using a query, aggregation or endpoint
the backend does not implement fail with a 400 error.
"""
