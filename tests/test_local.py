"""Unit tests for retry and timeout behavior in jina_cli.api."""

from datetime import datetime, timedelta, timezone
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
        sleep.assert_called_once_with(1.0)

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
        sleep.assert_called_once_with(1.0)

    def test_exhausted_retries_use_general_backoff(self):
        client = Mock()
        client.get.side_effect = [
            make_response(503, f"{api.READER_BASE}/")
            for _ in range(len(api.RETRY_BACKOFF))
        ]

        with patch("jina_cli.api.time.sleep") as sleep:
            try:
                api._request_with_retry("GET", f"{api.READER_BASE}/", client)
            except httpx.HTTPStatusError as exc:
                assert exc.response.status_code == 503
            else:
                assert False, "expected HTTPStatusError"

        assert client.get.call_count == len(api.RETRY_BACKOFF)
        assert sleep.call_args_list == [((wait,),) for wait in api.RETRY_BACKOFF[:-1]]

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
        sleep.assert_called_once_with(1.0)

    def test_retry_after_is_capped(self):
        client = Mock()
        client.get.side_effect = [
            make_response(429, f"{api.READER_BASE}/", {"Retry-After": "45"}),
            make_response(200, f"{api.READER_BASE}/"),
        ]

        with patch("jina_cli.api.time.sleep") as sleep:
            api._request_with_retry("GET", f"{api.READER_BASE}/", client)

        sleep.assert_called_once_with(api.MAX_RETRY_AFTER_WAIT)

    def test_retry_after_http_date_uses_future_delay(self):
        client = Mock()
        retry_at = datetime.now(timezone.utc) + timedelta(seconds=2)
        client.get.side_effect = [
            make_response(
                503,
                f"{api.READER_BASE}/",
                {"Retry-After": retry_at.strftime("%a, %d %b %Y %H:%M:%S GMT")},
            ),
            make_response(200, f"{api.READER_BASE}/"),
        ]

        with patch("jina_cli.api.time.sleep") as sleep:
            api._request_with_retry("GET", f"{api.READER_BASE}/", client)

        sleep.assert_called_once()
        wait = sleep.call_args.args[0]
        assert 1.0 <= wait <= 3.0

    def test_retry_after_http_date_in_past_uses_backoff(self):
        client = Mock()
        retry_at = datetime.now(timezone.utc) - timedelta(seconds=2)
        client.get.side_effect = [
            make_response(
                503,
                f"{api.READER_BASE}/",
                {"Retry-After": retry_at.strftime("%a, %d %b %Y %H:%M:%S GMT")},
            ),
            make_response(200, f"{api.READER_BASE}/"),
        ]

        with patch("jina_cli.api.time.sleep") as sleep:
            api._request_with_retry("GET", f"{api.READER_BASE}/", client)

        sleep.assert_called_once_with(1.0)

    def test_retry_after_invalid_http_date_uses_backoff(self):
        client = Mock()
        client.get.side_effect = [
            make_response(
                503,
                f"{api.READER_BASE}/",
                {"Retry-After": "not-a-date"},
            ),
            make_response(200, f"{api.READER_BASE}/"),
        ]

        with patch("jina_cli.api.time.sleep") as sleep:
            api._request_with_retry("GET", f"{api.READER_BASE}/", client)

        sleep.assert_called_once_with(1.0)

    def test_exhausted_retries_on_502_use_general_backoff(self):
        client = Mock()
        client.get.side_effect = [
            make_response(502, f"{api.READER_BASE}/"),
            make_response(502, f"{api.READER_BASE}/"),
            make_response(502, f"{api.READER_BASE}/"),
            make_response(502, f"{api.READER_BASE}/"),
            make_response(502, f"{api.READER_BASE}/"),
            make_response(502, f"{api.READER_BASE}/"),
        ]

        with patch("jina_cli.api.time.sleep") as sleep:
            try:
                api._request_with_retry("GET", f"{api.READER_BASE}/", client)
            except httpx.HTTPStatusError as exc:
                assert exc.response.status_code == 502
            else:
                assert False, "expected HTTPStatusError"

        assert client.get.call_count == len(api.RETRY_BACKOFF)
        assert sleep.call_args_list == [((wait,),) for wait in api.RETRY_BACKOFF[:-1]]


class TestTimeoutHelpers:
    def test_effective_timeout_uses_default(self):
        assert api._effective_timeout(None) == api.DEFAULT_TIMEOUT

    def test_effective_timeout_prefers_override(self):
        assert api._effective_timeout(12.5) == 12.5
