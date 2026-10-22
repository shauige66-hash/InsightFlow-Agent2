# Safe regression testing

Ordinary tests use temporary local databases, script-driven model responses and a network guard. Run them with bytecode and external model downloads disabled. A passing mock workflow is evidence about engineering behavior, not model accuracy.

PostgreSQL fixtures drop tables and Redis fixtures flush their selected logical database. They are disabled unless all reset safeguards are explicitly supplied. Never point them at the application's database or shared cache.

- Provision a separate PostgreSQL database whose name ends in `_test` or starts with `test_`. Set `INSIGHTFLOW_TEST_DATABASE_URL` to it and `INSIGHTFLOW_ALLOW_TEST_DB_RESET=1` only for the test process.
- Use a separate Redis test instance with a nonzero logical database, set `INSIGHTFLOW_TEST_REDIS_URL` explicitly, and opt in with `INSIGHTFLOW_ALLOW_TEST_REDIS_RESET=1`. A nonzero DB on a production/shared instance is still not recommended.
- Without opt-in, infrastructure tests skip before attempting a connection. No fixture automatically creates or resets a deployed database.
- Real model regression remains opt-in and incurs provider calls. Its verdict now also requires a machine-readable answer block to match independent CSV/Decimal business oracles; report completion and chart counts alone cannot pass.

Targeted offline checks for the audit repairs:

```powershell
python -B -m pytest -q -p no:cacheprovider tests/test_execution_boundary_regressions.py tests/test_business_answer_regressions.py tests/test_test_infrastructure_isolation.py
```

Docker runtime, Linux bind permissions, PostgreSQL concurrency and Redis faults require an explicit disposable environment. Skips must be reported separately from passes.
