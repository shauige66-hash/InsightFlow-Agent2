"""PostgreSQL 持久化：迁移、会话、记忆、知识索引、Run 与 Checkpoint。

这些用例跑真实容器。跳过意味着本机没有可用的 PostgreSQL（见 skip 原因），
而不是"用 SQLite 假装通过了"。
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

import storage
from tests.postgres_support import (  # noqa: F401
    drop_tables,
    postgres,
    requires_postgres,
)

pytestmark = [requires_postgres]


class TestMigration:
    def test_migration_is_idempotent(self, postgres) -> None:
        url = storage.database_url()
        # 从空 schema 开始，才能看出第一次迁移确实建了表
        storage.dispose_all()
        drop_tables()

        first = storage.migrate(url, knowledge=True)
        second = storage.migrate(url, knowledge=True)

        assert "core_tables" in first["applied"]
        assert "knowledge_tables" in first["applied"]
        # 第二次没有新东西可建
        assert second["applied"] == []

    def test_schema_version_is_recorded_once(self, postgres) -> None:
        from sqlalchemy import text

        url = storage.database_url()
        with storage.session_scope(url) as session:
            count = session.execute(
                text("SELECT COUNT(*) FROM schema_migrations")
            ).scalar()
        assert count == 1

    def test_healthcheck_reports_tables(self, postgres) -> None:
        payload = storage.healthcheck(storage.database_url())

        assert payload["dialect"] == "postgresql"
        assert payload["tables_ready"] is True
        assert payload["knowledge_ready"] is True

    def test_pgvector_extension_and_index_exist(self, postgres) -> None:
        from sqlalchemy import text

        with storage.session_scope(storage.database_url()) as session:
            version = session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname='vector'")
            ).scalar()
            indexes = [
                row[0]
                for row in session.execute(
                    text(
                        "SELECT indexname FROM pg_indexes "
                        "WHERE tablename='knowledge_chunks'"
                    )
                )
            ]
        assert version
        assert "ix_knowledge_chunks_embedding" in indexes


class TestConversations:
    def test_a_session_and_its_messages_survive_a_restart(self, postgres) -> None:
        from storage.conversations import ConversationRepository

        url = storage.database_url()
        with storage.session_scope(url) as session:
            repository = ConversationRepository(session, owner="pg-test")
            created = repository.create_session(title="季度复盘", dataset_refs=["a.csv"])
            repository.add_message(
                session_id=created.id, role="user", content="分析销售额"
            )
            session_id = created.id

        # 模拟进程重启：丢弃 Engine 与连接池，重新建
        storage.dispose_all()

        with storage.session_scope(storage.database_url()) as session:
            repository = ConversationRepository(session, owner="pg-test")
            restored = repository.get_session(session_id)
            messages = repository.list_messages(session_id)

        assert restored is not None
        assert restored.title == "季度复盘"
        assert [message.content for message in messages] == ["分析销售额"]

    def test_client_message_id_is_unique_per_session(self, postgres) -> None:
        from sqlalchemy.exc import IntegrityError

        from storage.conversations import ConversationRepository

        url = storage.database_url()

        # 第一个事务：两个会话 + 一条带客户端 id 的消息
        with storage.session_scope(url) as session:
            repository = ConversationRepository(session, owner="pg-test")
            first = repository.create_session(title="a", dataset_refs=["a.csv"])
            second = repository.create_session(title="b", dataset_refs=["a.csv"])
            repository.add_message(
                session_id=first.id,
                role="user",
                content="你好",
                client_message_id="client-1",
            )
            first_id, second_id = first.id, second.id

        # 第二个事务：同一会话重复写入必须被唯一约束挡住
        with pytest.raises(IntegrityError):
            with storage.session_scope(url) as session:
                repository = ConversationRepository(session, owner="pg-test")
                repository.add_message(
                    session_id=first_id,
                    role="user",
                    content="你好",
                    client_message_id="client-1",
                )

        # 第三个事务：不同会话可以用同一个客户端 id
        with storage.session_scope(url) as session:
            repository = ConversationRepository(session, owner="pg-test")
            repository.add_message(
                session_id=second_id,
                role="user",
                content="你好",
                client_message_id="client-1",
            )
            assert len(repository.list_messages(second_id)) == 1

    def test_lookup_by_run_id(self, postgres) -> None:
        from storage.conversations import ConversationRepository

        with storage.session_scope(storage.database_url()) as session:
            repository = ConversationRepository(session, owner="pg-test")
            created = repository.create_session(title="t", dataset_refs=["a.csv"])
            turn = repository.create_turn(session_id=created.id, run_id="run_x")
            repository.link_run(
                session_id=created.id, turn_id=turn.id, run_id="run_x"
            )
            assert repository.session_for_run("run_x").id == created.id
            assert repository.turn_for_run("run_x").id == turn.id

    def test_owner_scope_is_enforced(self, postgres) -> None:
        from storage.conversations import ConversationRepository

        with storage.session_scope(storage.database_url()) as session:
            created = ConversationRepository(session, owner="alice").create_session(
                title="alice 的会话", dataset_refs=["a.csv"]
            )
            session_id = created.id

        with storage.session_scope(storage.database_url()) as session:
            other = ConversationRepository(session, owner="bob")
            assert other.get_session(session_id) is None
            assert other.list_sessions() == []

    def test_deleting_a_session_keeps_messages_but_hides_it(self, postgres) -> None:
        from storage.conversations import ConversationRepository

        with storage.session_scope(storage.database_url()) as session:
            repository = ConversationRepository(session, owner="pg-test")
            created = repository.create_session(title="t", dataset_refs=["a.csv"])
            repository.add_message(session_id=created.id, role="user", content="x")
            assert repository.delete_session(created.id) is True
            assert repository.get_session(created.id) is None
            # 软删除：消息仍可追溯
            assert len(repository.list_messages(created.id)) == 1


class TestKnowledgeIndex:
    def _put(self, index, *, namespace: str, source: str, content: str, vector):
        from storage.knowledge import IndexedChunk

        return index.put_document(
            namespace=namespace,
            owner="pg-test",
            source=source,
            content_hash=f"hash-{content[:8]}",
            knowledge_version="v1",
            embedding_model_version="model@384",
            chunks=[IndexedChunk(index=0, content=content, embedding=vector)],
        )

    @staticmethod
    def _vector(seed: float) -> list[float]:
        vector = [0.0] * 384
        vector[0] = seed
        vector[1] = 1.0 - seed
        return vector

    def test_vectors_survive_a_restart_and_are_searchable(self, postgres) -> None:
        from storage.knowledge import PostgresKnowledgeIndex

        with storage.session_scope(storage.database_url()) as session:
            index = PostgresKnowledgeIndex(session)
            self._put(
                index,
                namespace="kb:test:a",
                source="vip.md",
                content="VIP 客户年消费超过 10000 元",
                vector=self._vector(0.9),
            )

        storage.dispose_all()

        with storage.session_scope(storage.database_url()) as session:
            index = PostgresKnowledgeIndex(session)
            hits = index.search(
                namespaces=["kb:test:a"],
                embedding=self._vector(0.9),
                top_k=3,
                min_score=0.0,
            )
        assert [hit.source for hit in hits] == ["vip.md"]
        assert hits[0].score > 0.9

    def test_search_is_filtered_by_namespace(self, postgres) -> None:
        from storage.knowledge import PostgresKnowledgeIndex

        with storage.session_scope(storage.database_url()) as session:
            index = PostgresKnowledgeIndex(session)
            self._put(
                index,
                namespace="kb:a",
                source="a.md",
                content="A 知识库内容",
                vector=self._vector(0.9),
            )
            self._put(
                index,
                namespace="kb:b",
                source="b.md",
                content="B 知识库内容",
                vector=self._vector(0.9),
            )
            hits = index.search(
                namespaces=["kb:a"],
                embedding=self._vector(0.9),
                top_k=5,
                min_score=0.0,
            )

        assert {hit.namespace for hit in hits} == {"kb:a"}
        assert [hit.source for hit in hits] == ["a.md"]

    def test_indexing_is_idempotent_for_the_same_content(self, postgres) -> None:
        from storage.knowledge import PostgresKnowledgeIndex

        with storage.session_scope(storage.database_url()) as session:
            index = PostgresKnowledgeIndex(session)
            _, reused_first = self._put(
                index,
                namespace="kb:a",
                source="a.md",
                content="内容",
                vector=self._vector(0.5),
            )
            _, reused_second = self._put(
                index,
                namespace="kb:a",
                source="a.md",
                content="内容",
                vector=self._vector(0.5),
            )
            stats = index.stats("kb:a")

        assert reused_first is False
        assert reused_second is True
        assert stats["documents"] == 1
        assert stats["chunks"] == 1

    def test_a_new_version_deactivates_the_old_chunks(self, postgres) -> None:
        from storage.knowledge import PostgresKnowledgeIndex

        with storage.session_scope(storage.database_url()) as session:
            index = PostgresKnowledgeIndex(session)
            self._put(
                index,
                namespace="kb:a",
                source="a.md",
                content="旧内容",
                vector=self._vector(0.9),
            )
            version_before = index.namespace_version("kb:a")

            index.put_document(
                namespace="kb:a",
                owner="pg-test",
                source="a.md",
                content_hash="hash-new",
                knowledge_version="v2",
                embedding_model_version="model@384",
                chunks=[],
            )
            version_after = index.namespace_version("kb:a")
            hits = index.search(
                namespaces=["kb:a"],
                embedding=self._vector(0.9),
                top_k=5,
                min_score=0.0,
            )

        assert version_before != version_after
        assert hits == [], "旧版本的向量仍然被检索到"

    def test_delete_namespace_removes_documents_and_chunks(self, postgres) -> None:
        from storage.knowledge import PostgresKnowledgeIndex

        with storage.session_scope(storage.database_url()) as session:
            index = PostgresKnowledgeIndex(session)
            self._put(
                index,
                namespace="kb:gone",
                source="a.md",
                content="要删除的知识",
                vector=self._vector(0.7),
            )
            removed = index.delete_namespace("kb:gone")
            stats = index.stats("kb:gone")

        assert removed == 1
        assert stats == {"documents": 0, "chunks": 0}

    def test_wrong_dimension_is_rejected(self, postgres) -> None:
        from storage.knowledge import IndexedChunk, PostgresKnowledgeIndex

        with storage.session_scope(storage.database_url()) as session:
            index = PostgresKnowledgeIndex(session)
            with pytest.raises(ValueError, match="维度"):
                index.put_document(
                    namespace="kb:a",
                    owner="pg-test",
                    source="a.md",
                    content_hash="h",
                    knowledge_version="v1",
                    embedding_model_version="model@128",
                    chunks=[
                        IndexedChunk(index=0, content="x", embedding=[0.1, 0.2])
                    ],
                )


class TestRunStore:
    def test_run_lifecycle_survives_a_restart(self, postgres) -> None:
        from services import storage_factory

        store = storage_factory.run_store("outputs")
        store.create("run-pg-1", data_sources=["a.csv"], analysis_focus="x")
        store.update_status("run-pg-1", "running")
        store.touch("run-pg-1")
        store.record_approval("run-pg-1", {"approval_id": "a1", "decision": "approve"})

        storage.dispose_all()

        reopened = storage_factory.run_store("outputs")
        record = reopened.get("run-pg-1")

        assert record is not None
        assert record.status == "running"
        assert len(record.approvals) == 1
        assert reopened.is_stale_running("run-pg-1") is False

    def test_create_is_an_upsert(self, postgres) -> None:
        from services import storage_factory

        store = storage_factory.run_store("outputs")
        first = store.create("run-dup", data_sources=["a.csv"])
        second = store.create("run-dup", data_sources=["b.csv"])

        assert first.created_at == second.created_at
        assert second.data_sources == ["b.csv"]

    def test_unknown_status_is_rejected(self, postgres) -> None:
        from services import storage_factory

        store = storage_factory.run_store("outputs")
        store.create("run-bad", data_sources=["a.csv"])
        with pytest.raises(ValueError):
            store.update_status("run-bad", "not-a-status")

    def test_checkpointer_uses_postgres_in_this_mode(self, postgres) -> None:
        from core.checkpoints import checkpoint_backend

        assert checkpoint_backend() == "postgres"


class TestCheckpointResume:
    def test_approval_resume_after_a_new_connection(self, postgres, tmp_path: Path) -> None:
        """跨连接恢复：关闭 Checkpointer 再打开，等于一次服务重启。"""
        from langgraph.types import Command

        from core.checkpoints import open_checkpointer
        from services.analysis_service import run_analysis
        from tests.test_report_approval import build_database
        from tests.test_e2e_business_scenarios import ScriptedFactory

        database = tmp_path / "customers.sqlite"
        build_database(database)
        run_id = f"pg-resume-{uuid.uuid4().hex[:8]}"
        replies = {
            "planner": {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "description": "统计客户总数",
                        "target_tables": ["customers.customers"],
                        "expected_output": "客户数",
                    }
                ]
            },
            "executor": {"code": "SELECT COUNT(*) AS n FROM customers"},
            "critic": {"validation_passed": True, "feedback": "ok"},
            "visualizer": {"charts": []},
            "reporter": "# 报告\n\n共 3 位客户。\n",
        }

        with open_checkpointer() as saver:
            paused = run_analysis(
                run_id=run_id,
                data_sources=[str(database)],
                model_factory=ScriptedFactory(run_id, replies).build(),
                output_root=tmp_path,
                require_plan_approval=True,
                checkpointer=saver,
            )
        assert paused["__interrupt__"]

        storage.dispose_all()

        with open_checkpointer() as saver:
            resumed = run_analysis(
                run_id=run_id,
                data_sources=[str(database)],
                model_factory=ScriptedFactory(run_id, replies).build(),
                output_root=tmp_path,
                require_plan_approval=True,
                checkpointer=saver,
                resume=Command(resume={"decision": "approve"}),
            )

        assert resumed["run_status"] == "completed"
        assert resumed["analysis_status"] == "completed"

    def test_list_checkpoint_threads_reads_postgres(self, postgres, tmp_path: Path) -> None:
        from core.checkpoints import list_checkpoint_threads, open_checkpointer
        from services.analysis_service import run_analysis
        from tests.test_report_approval import build_database
        from tests.test_e2e_business_scenarios import ScriptedFactory

        database = tmp_path / "customers.sqlite"
        build_database(database)
        run_id = f"pg-threads-{uuid.uuid4().hex[:8]}"
        replies = {
            "planner": {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "description": "统计",
                        "target_tables": ["customers.customers"],
                        "expected_output": "数量",
                    }
                ]
            },
            "executor": {"code": "SELECT COUNT(*) AS n FROM customers"},
            "critic": {"validation_passed": True, "feedback": "ok"},
            "visualizer": {"charts": []},
            "reporter": "# 报告\n",
        }
        with open_checkpointer() as saver:
            run_analysis(
                run_id=run_id,
                data_sources=[str(database)],
                model_factory=ScriptedFactory(run_id, replies).build(),
                output_root=tmp_path,
                checkpointer=saver,
            )

        assert run_id in list_checkpoint_threads()


class TestCrossProcessPersistence:
    def test_another_process_sees_the_same_rows(self, postgres, tmp_path: Path) -> None:
        """真正的跨进程验证：子进程读同一个数据库。"""
        from storage.conversations import ConversationRepository

        with storage.session_scope(storage.database_url()) as session:
            repository = ConversationRepository(session, owner="pg-test")
            created = repository.create_session(title="跨进程", dataset_refs=["a.csv"])
            repository.add_message(session_id=created.id, role="user", content="你好")
            session_id = created.id

        script = (
            "import os, sys\n"
            "os.environ['INSIGHTFLOW_STORAGE'] = 'postgres'\n"
            f"os.environ['INSIGHTFLOW_DATABASE_URL'] = {storage.database_url()!r}\n"
            "import storage\n"
            "storage.configure()\n"
            "from storage.conversations import ConversationRepository\n"
            "with storage.session_scope(storage.database_url()) as s:\n"
            "    repo = ConversationRepository(s, owner='pg-test')\n"
            "    session = repo.get_session(%r)\n"
            "    print(session.title, len(repo.list_messages(%r)))\n"
        ) % (session_id, session_id)

        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(__file__).resolve().parent.parent),
            timeout=120,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        assert completed.returncode == 0, completed.stderr
        assert (completed.stdout or "").strip() == "跨进程 1"
