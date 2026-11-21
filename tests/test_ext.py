# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""Picking the backend from SEARCH_HOSTS."""

import pytest
from flask import Flask
from invenio_search import InvenioSearch, current_search_client

from invenio_search_paradedb import InvenioSearchParadeDB, ParadeDBConnection


def client_for(hosts):
    app = Flask("test")
    app.config.update(SEARCH_HOSTS=hosts, SQLALCHEMY_DATABASE_URI="sqlite://")
    InvenioSearchParadeDB(app)
    InvenioSearch(app)
    with app.app_context():
        return current_search_client.transport


@pytest.mark.parametrize("hosts", ["paradedb://", ["paradedb://"]])
def test_app_database(hosts):
    transport = client_for(hosts)
    assert transport.connection_class is ParadeDBConnection
    assert transport.get_connection().dsn is None


def test_own_database():
    transport = client_for("paradedb://u:p@db:5433/search")
    assert transport.get_connection().dsn == "postgresql://u:p@db:5433/search"


def test_opensearch_untouched():
    transport = client_for(["localhost:9200"])
    assert transport.connection_class is not ParadeDBConnection
