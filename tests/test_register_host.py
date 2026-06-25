import builtins

import pytest

from services.register_host import get_hosts_file, register_host


def test_get_hosts_file_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("HOSTS_FILE", raising=False)

    assert get_hosts_file() is None


def test_register_host_skips_file_when_hosts_file_unset(monkeypatch):
    monkeypatch.delenv("HOSTS_FILE", raising=False)

    def fail_open(*args, **kwargs):
        pytest.fail("register_host should not touch the filesystem when HOSTS_FILE is unset")

    monkeypatch.setattr(builtins, "open", fail_open)

    assert register_host("couliglig1", "192.168.1.10") is False


def test_register_host_writes_configured_hosts_file(monkeypatch, tmp_path):
    hosts_file = tmp_path / "robots.hosts"
    hosts_file.write_text(
        "192.168.1.10 couliglig1.lan couliglig1\n"
        "192.168.1.11 couliglig2.lan couliglig2\n"
    )
    monkeypatch.setenv("HOSTS_FILE", str(hosts_file))

    assert register_host("couliglig1", "192.168.1.99") is True

    assert hosts_file.read_text() == (
        "192.168.1.11 couliglig2.lan couliglig2\n"
        "192.168.1.99 couliglig1.lan couliglig1\n"
    )
