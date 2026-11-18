import pytest
from ml.preflight import MetadataClient, validate_url


@pytest.mark.parametrize("url", ["http://huggingface.co/a", "https://huggingface.co.evil.example/a", "https://user:pass@huggingface.co/a", "https://www.openslr.org:444/a", "file:///x", "https://example.com/file"])
def test_official_metadata_allowlist_rejects_untrusted_urls(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_official_metadata_url():
    assert validate_url("https://huggingface.co/api/models/x")


def test_exhausted_metadata_budget_never_requests_network():
    c = MetadataClient(max_bytes=0)
    with pytest.raises(RuntimeError, match="budget"):
        c.get("https://huggingface.co/api/models/x")
