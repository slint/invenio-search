=========================
 invenio-search-paradedb
=========================

Run Invenio's search indices on Postgres with the
`ParadeDB <https://github.com/paradedb/paradedb>`_ ``pg_search`` extension
(AGPL), so a small instance needs no OpenSearch cluster.

It is meant for local development, demos and PR previews. Production, and any
work on search ranking, analyzers or autocomplete, should use OpenSearch.

Setup
=====

.. code-block:: console

    $ docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres paradedb/paradedb
    $ pip install invenio-search-paradedb

.. code-block:: python

    # invenio.cfg (or INVENIO_SEARCH_HOSTS='paradedb://')
    SEARCH_HOSTS = "paradedb://"                         # the app database
    # SEARCH_HOSTS = "paradedb://user:pass@host:5432/db"  # or another database

Then run ``invenio index init`` as usual. Indices live in the
``invenio_search`` schema; the first connection creates the ``pg_search``
extension, so the database user needs the right to do that. Remove the
setting to go back to OpenSearch.

How it works
============

invenio-search keeps building its OpenSearch client. This package registers an
Invenio extension that, when ``SEARCH_HOSTS`` is a ``paradedb://`` URL, gives
that client a connection class which answers the OpenSearch REST calls from
Postgres instead of over HTTP. No change to invenio-search is needed.

What behaves differently
========================

- Filters, permissions, facets/aggregations, sorting on fields, versions and
  percolation are exact SQL and match OpenSearch.
- Text queries use ParadeDB's tokenizer, not the index analyzers. Relevance
  order differs, and some searches find fewer or other records, notably
  partial-word autocomplete. See the caveats in the project notes.
- Geo queries, scripts, highlighting, suggesters, point in time and
  ``more_like_this`` return a 400 error.

Tests
=====

.. code-block:: console

    $ PARADEDB_DSN=postgresql://postgres:postgres@localhost/test pytest
