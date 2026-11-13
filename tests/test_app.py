"""Tests for the docu-ask upload route: temp docs dir, no real files touched."""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # point uploads at a throwaway folder so docs/ is never touched
    monkeypatch.setattr(app_module, "DOCS_DIR", str(tmp_path))
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _upload(client, name, data=b"some doc text"):
    return client.post(
        "/upload",
        data={"file": (io.BytesIO(data), name)},
        content_type="multipart/form-data",
    )


def test_upload_saves_and_redirects(client, tmp_path):
    resp = _upload(client, "notes.md")
    assert resp.status_code == 302
    assert os.path.isfile(os.path.join(str(tmp_path), "notes.md"))


def test_upload_rejects_fully_non_ascii_name(client, tmp_path):
    # secure_filename turns "日本語.md" into a bare "md" that
    # load_documents would skip silently — surface it instead
    resp = _upload(client, "日本語.md")
    assert resp.status_code == 200
    assert b"plain letters" in resp.data
    assert os.listdir(str(tmp_path)) == []


def test_upload_rejects_empty_file(client, tmp_path):
    resp = _upload(client, "empty.md", data=b"")
    assert resp.status_code == 200
    assert b"That file is empty." in resp.data
    assert os.listdir(str(tmp_path)) == []


def test_upload_rejects_whitespace_only_file(client, tmp_path):
    resp = _upload(client, "blank.txt", data=b"  \n\n  ")
    assert resp.status_code == 200
    assert b"That file is empty." in resp.data
    assert os.listdir(str(tmp_path)) == []


def test_upload_rejects_bad_extension(client, tmp_path):
    resp = _upload(client, "photo.png")
    assert resp.status_code == 200
    assert b"Only .txt and .md files." in resp.data
    assert os.listdir(str(tmp_path)) == []


def test_upload_rejects_non_utf8_file(client, tmp_path):
    # load_documents skips undecodable files, so saving this would lose it silently
    resp = _upload(client, "latin1.txt", data=b"caf\xe9 \x96\x99")
    assert resp.status_code == 200
    # the template autoescapes the apostrophe
    assert b"That file isn&#39;t plain UTF-8 text." in resp.data
    assert os.listdir(str(tmp_path)) == []


def test_upload_accepts_non_ascii_utf8(client, tmp_path):
    resp = _upload(client, "cafe.txt", data="café".encode("utf-8"))
    assert resp.status_code == 302
    assert os.path.isfile(os.path.join(str(tmp_path), "cafe.txt"))


def test_delete_removes_document(client, tmp_path):
    _upload(client, "notes.md", data=b"some doc text")
    assert os.path.isfile(os.path.join(str(tmp_path), "notes.md"))
    resp = client.post("/documents/notes.md/delete")
    assert resp.status_code == 302
    assert not os.path.exists(os.path.join(str(tmp_path), "notes.md"))
    assert b"notes.md" not in client.get("/documents").data


def test_delete_missing_document_returns_404(client, tmp_path):
    resp = client.post("/documents/nope.md/delete")
    assert resp.status_code == 404
    assert os.listdir(str(tmp_path)) == []


def test_delete_rejects_non_document_extension(client, tmp_path):
    # app.py lives next to docs/, not in it - never deletable via this route
    resp = client.post("/documents/app.py/delete")
    assert resp.status_code == 404
    assert os.listdir(str(tmp_path)) == []


def test_deleted_document_stops_answering(client):
    _upload(client, "notes.md", data=b"the quokka codeword is wombat")
    assert b"wombat" in client.post("/", data={"question": "quokka codeword"}).data
    client.post("/documents/notes.md/delete")
    assert b"wombat" not in client.post("/", data={"question": "quokka codeword"}).data
