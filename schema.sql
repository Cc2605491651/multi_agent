-- spec v4 §3.1 / §3.3 一次性建表脚本
-- 阶段 1 只用 transcript_turns；阶段 2 用 tasks + dag_nodes
-- dag_nodes 字段按 spec §13 强调"一次到位"，阶段 2 不再改表

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- spec §3.1 对话原文库
CREATE TABLE IF NOT EXISTS transcript_turns (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    turn_index      INTEGER NOT NULL,
    agent_id        TEXT,
    user_input      TEXT NOT NULL,
    agent_output    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE(conversation_id, turn_index)
);

CREATE INDEX IF NOT EXISTS idx_transcript_conv
    ON transcript_turns(conversation_id, turn_index);

-- spec §3.3 状态库 · tasks
CREATE TABLE IF NOT EXISTS tasks (
    id                       TEXT PRIMARY KEY,
    user_id                  TEXT NOT NULL,
    title                    TEXT NOT NULL,
    dag_id                   TEXT NOT NULL,
    handoff_conversation_id  TEXT,
    handoff_turn_range       TEXT,   -- JSON 数组 [start, end]
    status                   TEXT NOT NULL,
    created_at               TEXT NOT NULL
);

-- spec §3.3 状态库 · dag_nodes（字段一次到位）
CREATE TABLE IF NOT EXISTS dag_nodes (
    id                TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL,
    node_name         TEXT NOT NULL,
    depends_on        TEXT,                          -- JSON 数组
    status            TEXT NOT NULL,                 -- pending/running/done/failed/skipped
    failure_policy    TEXT NOT NULL DEFAULT 'fail_retry',
    retry_count       INTEGER NOT NULL DEFAULT 0,
    max_retries       INTEGER NOT NULL DEFAULT 2,
    worker_id         TEXT,
    input_memory_ids  TEXT,                          -- JSON 数组
    output_memory_id  TEXT,
    heartbeat_at      TEXT,
    started_at        TEXT,
    finished_at       TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_dag_nodes_task   ON dag_nodes(task_id);
CREATE INDEX IF NOT EXISTS idx_dag_nodes_status ON dag_nodes(status);
