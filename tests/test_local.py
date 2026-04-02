"""Unit tests for retry and timeout behavior in jina_cli.api."""

from unittest.mock import Mock, patch

import httpx

from jina_cli import api


def make_response(
    status_code: int, url: str, headers: dict | None = None
) -> httpx.Response:
    request = httpx.Request("GET", url)
    return httpx.Response(status_code, headers=headers, request=request)


class TestRequestRetry:
    def test_retry_on_429(self):
        client = Mock()
        client.get.side_effect = [
            make_response(429, f"{api.READER_BASE}/", {"Retry-After": "0.25"}),
            make_response(200, f"{api.READER_BASE}/"),
        ]

        with patch("jina_cli.api.time.sleep") as sleep:
            resp = api._request_with_retry("GET", f"{api.READER_BASE}/", client)

        assert resp.status_code == 200
        assert client.get.call_count == 2
        sleep.assert_called_once_with(0.5)

    def test_retry_on_jina_5xx(self):
        client = Mock()
        client.post.side_effect = [
            make_response(503, f"{api.API_BASE}/v1/embeddings"),
            make_response(200, f"{api.API_BASE}/v1/embeddings"),
        ]

        with patch("jina_cli.api.time.sleep") as sleep:
            resp = api._request_with_retry(
                "POST", f"{api.API_BASE}/v1/embeddings", client
            )

        assert resp.status_code == 200
        assert client.post.call_count == 2
        sleep.assert_called_once_with(0.5)

    def test_no_retry_on_client_error(self):
        client = Mock()
        client.get.return_value = make_response(404, f"{api.READER_BASE}/")

        with patch("jina_cli.api.time.sleep") as sleep:
            try:
                api._request_with_retry("GET", f"{api.READER_BASE}/", client)
            except httpx.HTTPStatusError as exc:
                assert exc.response.status_code == 404
            else:
                assert False, "expected HTTPStatusError"

        assert client.get.call_count == 1
        sleep.assert_not_called()

    def test_no_retry_on_non_jina_5xx(self):
        client = Mock()
        client.get.return_value = make_response(503, "https://example.com")

        with patch("jina_cli.api.time.sleep") as sleep:
            try:
                api._request_with_retry("GET", "https://example.com", client)
            except httpx.HTTPStatusError as exc:
                assert exc.response.status_code == 503
            else:
                assert False, "expected HTTPStatusError"

        assert client.get.call_count == 1
        sleep.assert_not_called()

    def test_retry_on_timeout_exception(self):
        client = Mock()
        client.get.side_effect = [
            httpx.TimeoutException("timed out"),
            make_response(200, f"{api.READER_BASE}/"),
        ]

        with patch("jina_cli.api.time.sleep") as sleep:
            resp = api._request_with_retry("GET", f"{api.READER_BASE}/", client)

        assert resp.status_code == 200
        assert client.get.call_count == 2
        sleep.assert_called_once_with(0.5)

    def test_retry_after_is_capped(self):
        client = Mock()
        client.get.side_effect = [
            make_response(429, f"{api.READER_BASE}/", {"Retry-After": "45"}),
            make_response(200, f"{api.READER_BASE}/"),
        ]

        with patch("jina_cli.api.time.sleep") as sleep:
            api._request_with_retry("GET", f"{api.READER_BASE}/", client)

        sleep.assert_called_once_with(30.0)


class TestTimeoutHelpers:
    def test_effective_timeout_uses_default(self):
        assert api._effective_timeout(None) == api.DEFAULT_TIMEOUT

    def test_effective_timeout_prefers_override(self):
        assert api._effective_timeout(12.5) == 12.5
