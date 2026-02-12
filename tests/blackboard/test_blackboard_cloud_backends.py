from io import BytesIO

import pytest

from automa_ai.blackboard.backends.dynamodb_json import DynamoDBJSONBlackboardStore
from automa_ai.blackboard.backends.s3_json import S3JSONBlackboardStore
from automa_ai.blackboard.errors import RevisionConflictError
from automa_ai.blackboard.schema import BlackboardSchemaRegistry, BlackboardSchemaValidator


class FakeS3:
    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = Body


class FakeDynamoTable:
    def __init__(self):
        self.items = {}

    def get_item(self, Key):
        item = self.items.get(Key["session_id"])
        return {"Item": item} if item else {}

    def put_item(self, Item, ConditionExpression, ExpressionAttributeValues=None, ExpressionAttributeNames=None):
        sid = Item["session_id"]
        current = self.items.get(sid)
        if ConditionExpression == "attribute_not_exists(session_id)" and current is not None:
            raise RuntimeError("exists")
        if ConditionExpression == "#rev = :expected":
            expected = ExpressionAttributeValues[":expected"]
            if current is None or current.get("revision") != expected:
                raise RuntimeError("conflict")
        self.items[sid] = Item


def _validator():
    registry = BlackboardSchemaRegistry()
    registry.register("test", "1", {"type": "object", "properties": {"items": {"type": "array"}}})
    return BlackboardSchemaValidator(registry)


def test_s3_backend_mocked_roundtrip():
    store = S3JSONBlackboardStore("bucket", "prefix", _validator(), s3_client=FakeS3())
    doc = store.create("s1", "test", "1", {"items": []})
    assert doc.revision == 1
    loaded = store.load("s1")
    assert loaded.data["items"] == []


def test_dynamodb_backend_conditional_conflict():
    store = DynamoDBJSONBlackboardStore("table", _validator(), dynamodb_table=FakeDynamoTable())
    created = store.create("s1", "test", "1", {"items": []})
    assert created.revision == 1

    with pytest.raises(RevisionConflictError):
        store.save(created, expected_revision=2)
