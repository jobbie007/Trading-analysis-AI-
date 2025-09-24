# Import from the canonical utilities package
from utils.json_utils import extract_json_object  # type: ignore


def test_extract_full_json():
    text = '{"a": 1, "b": 2}'
    expected = {"a": 1, "b": 2}
    obj = extract_json_object(text)
    assert isinstance(obj, dict)
    assert obj["a"] == expected["a"] and obj["b"] == expected["b"]


def test_extract_embedded_json():
    text = "noise before {\n  \"ok\": true, \n  \"n\": 5\n} and after"
    expected_n = 5
    obj = extract_json_object(text)
    assert isinstance(obj, dict)
    assert obj["ok"] is True and obj["n"] == expected_n


def test_extract_none_when_absent():
    assert extract_json_object("no braces here") is None
