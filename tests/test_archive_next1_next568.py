import gzip
import json
from pathlib import Path

from src.archive_next1_next568 import stage_tokens, write_visible_transcript


def test_stage_tokens_expand_named_ranges() -> None:
    assert stage_tokens("NEXT399--NEXT402 and next566b") == {399, 400, 401, 402, 566}
    assert stage_tokens("next525-next528") == {525, 526, 527, 528}


def test_visible_transcript_excludes_reasoning_and_system_records(tmp_path: Path) -> None:
    session = tmp_path / "rollout.jsonl"
    records = [
        {
            "timestamp": "2026-08-14T00:00:00Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "question"},
        },
        {
            "timestamp": "2026-08-14T00:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": ["must not appear"],
                "encrypted_content": "secret",
            },
        },
        {
            "timestamp": "2026-08-14T00:00:02Z",
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "phase": "commentary",
                "message": "answer",
            },
        },
        {
            "timestamp": "2026-08-14T00:00:03Z",
            "type": "turn_context",
            "payload": {"developer_instructions": "must not appear"},
        },
    ]
    session.write_text("".join(json.dumps(record) + "\n" for record in records))
    destination = tmp_path / "visible.jsonl.gz"

    count, _ = write_visible_transcript(session, destination)

    with gzip.open(destination, "rt", encoding="utf-8") as handle:
        exported = [json.loads(line) for line in handle]
    assert count == 2
    assert [record["role"] for record in exported] == ["user", "assistant"]
    payload = json.dumps(exported)
    assert "must not appear" not in payload
    assert "encrypted_content" not in payload
    assert "developer_instructions" not in payload
