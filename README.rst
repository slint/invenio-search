..
    SPDX-FileCopyrightText: 2015-2018 CERN.
    SPDX-License-Identifier: MIT

================
 Invenio-Search
================

.. image:: https://img.shields.io/github/license/inveniosoftware/invenio-search.svg
        :target: https://github.com/inveniosoftware/invenio-search/blob/master/LICENSE

.. image:: https://github.com/inveniosoftware/invenio-search/workflows/CI/badge.svg
        :target: https://github.com/inveniosoftware/invenio-search/actions?query=workflow%3ACI

.. image:: https://img.shields.io/coveralls/inveniosoftware/invenio-search.svg
        :target: https://coveralls.io/r/inveniosoftware/invenio-search

.. image:: https://img.shields.io/pypi/v/invenio-search.svg
        :target: https://pypi.org/pypi/invenio-search


Search management for Invenio (for Elasticsearch and OpenSearch).

Features:

- Allows Invenio modules to register indexes, aliases and index templates.
- Manages the creation and deletion of indices, aliases and templates.
- API for providing stable searches (e.g. prevents bouncing of search results).
- Maps JSONSchema URLs to Elasticsearch/OpenSearch indexes.
- Supports Elasticsearch v7 and OpenSearch v1.
- Optional Postgres search backend (ParadeDB ``pg_search``) for small
  instances, see below.

Postgres-only search (ParadeDB)
===============================

For local development, demos and test environments, the search indices can
live in Postgres instead of an OpenSearch cluster. The OpenSearch client is
kept; a connection class answers its requests from Postgres with the
`ParadeDB <https://github.com/paradedb/paradedb>`_ ``pg_search`` extension
(AGPL). Production setups should keep using OpenSearch.

.. code-block:: console

    $ docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=invenio paradedb/paradedb
    $ pip install "invenio-search[opensearch2,paradedb]"

.. code-block:: python

    # invenio.cfg
    SEARCH_CLIENT_CONFIG = {
        "connection_class": "invenio_search.paradedb.connection:ParadeDBConnection",
    }

Then run ``invenio index init`` as usual. Indices are stored in the
``invenio_search`` schema of ``SQLALCHEMY_DATABASE_URI`` (or
``SEARCH_PARADEDB_DSN``). Index analyzers are not applied, and queries or
aggregations the backend does not implement fail with a 400 error.

Further documentation is available at https://invenio-search.readthedocs.io/.
