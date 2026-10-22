"""Normal test execution must never reset the application's own DB/cache."""
import pytest
from tests import postgres_support as support


def test_postgres_reset_requires_explicit_dedicated_database(monkeypatch):
    monkeypatch.delenv('INSIGHTFLOW_ALLOW_TEST_DB_RESET', raising=False)
    monkeypatch.delenv('INSIGHTFLOW_TEST_DATABASE_URL', raising=False)
    with pytest.raises(RuntimeError):
        support.require_isolated_postgres()
    monkeypatch.setenv('INSIGHTFLOW_ALLOW_TEST_DB_RESET', '1')
    monkeypatch.setenv('INSIGHTFLOW_TEST_DATABASE_URL', 'postgresql://user:pass@127.0.0.1:1/insightflow')
    with pytest.raises(RuntimeError):
        support.require_isolated_postgres()
    monkeypatch.setenv('INSIGHTFLOW_TEST_DATABASE_URL', 'postgresql://user:pass@127.0.0.1:1/insightflow_test')
    support.require_isolated_postgres()


def test_redis_reset_rejects_default_database(monkeypatch):
    monkeypatch.setenv('INSIGHTFLOW_ALLOW_TEST_REDIS_RESET', '1')
    monkeypatch.setenv('INSIGHTFLOW_TEST_REDIS_URL', 'redis://127.0.0.1:1/0')
    with pytest.raises(RuntimeError):
        support.require_isolated_redis()
    monkeypatch.setenv('INSIGHTFLOW_TEST_REDIS_URL', 'redis://127.0.0.1:1/15')
    support.require_isolated_redis()
