"""Local tests for API helpers and CLI behavior without live network calls."""

import json
from click.testing import CliRunner
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import httpx

from jina_cli import api
from jina_cli.main import cli


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


class TestCliLocal:
    def test_search_human_readable_formats_results(self):
        runner = CliRunner()
        result_payload = {
            "results": [
                {
                    "title": "Jina AI",
                    "url": "https://jina.ai",
                    "snippet": "Search foundation models.",
                }
            ]
        }

        with patch(
            "jina_cli.main.api.search_web", return_value=result_payload
        ) as search_web:
            result = runner.invoke(cli, ["search", "jina ai", "-n", "2"])

        assert result.exit_code == 0
        assert "Jina AI" in result.output
        assert "https://jina.ai" in result.output
        assert "Search foundation models." in result.output
        search_web.assert_called_once()
        _, kwargs = search_web.call_args
        assert search_web.call_args.args == ("jina ai",)
        assert kwargs["num"] == 2
        assert kwargs["tbs"] is None
        assert kwargs["location"] is None
        assert kwargs["gl"] is None
        assert kwargs["hl"] is None
        assert kwargs["as_json"] is True
        assert kwargs["timeout"] is None

    def test_search_arxiv_routes_to_arxiv_api(self):
        runner = CliRunner()

        with patch(
            "jina_cli.main.api.search_arxiv",
            return_value={"results": [{"title": "Attention Is All You Need"}]},
        ) as search_arxiv:
            result = runner.invoke(cli, ["search", "--arxiv", "attention", "--json"])

        assert result.exit_code == 0
        assert (
            json.loads(result.output)["results"][0]["title"]
            == "Attention Is All You Need"
        )
        search_arxiv.assert_called_once()
        _, kwargs = search_arxiv.call_args
        assert search_arxiv.call_args.args == ("attention",)
        assert kwargs["num"] == 5
        assert kwargs["tbs"] is None
        assert kwargs["as_json"] is True
        assert kwargs["timeout"] is None

    def test_embed_reads_stdin_and_formats_output(self):
        runner = CliRunner()

        with patch(
            "jina_cli.main.api.embed",
            return_value=[{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
        ) as embed:
            result = runner.invoke(cli, ["embed"], input="hello world\n")

        assert result.exit_code == 0
        assert "dim=3" in result.output
        embed.assert_called_once()
        _, kwargs = embed.call_args
        assert embed.call_args.args == (["hello world"],)
        assert kwargs["model"] == "jina-embeddings-v5-text-small"
        assert kwargs["task"] == "text-matching"
        assert kwargs["dimensions"] is None
        assert kwargs["timeout"] is None

    def test_rerank_uses_stdin_documents(self):
        runner = CliRunner()

        with patch(
            "jina_cli.main.api.rerank",
            return_value=[{"index": 1, "relevance_score": 0.9}],
        ) as rerank:
            result = runner.invoke(cli, ["rerank", "pet"], input="cat\ndog\n")

        assert result.exit_code == 0
        assert "[0.9000] dog" in result.output
        rerank.assert_called_once()
        _, kwargs = rerank.call_args
        assert rerank.call_args.args == ("pet", ["cat", "dog"])
        assert kwargs["model"] == "jina-reranker-v3"
        assert kwargs["top_n"] is None
        assert kwargs["timeout"] is None

    def test_dedup_uses_stdin_lines(self):
        runner = CliRunner()

        with patch(
            "jina_cli.main.api.deduplicate",
            return_value=[{"text": "hello world"}],
        ) as deduplicate:
            result = runner.invoke(cli, ["dedup"], input="hello world\nhello world\n")

        assert result.exit_code == 0
        assert result.output.strip() == "hello world"
        deduplicate.assert_called_once()
        _, kwargs = deduplicate.call_args
        assert deduplicate.call_args.args == (["hello world", "hello world"],)
        assert kwargs["k"] is None
        assert kwargs["timeout"] is None

    def test_classify_parses_comma_separated_labels(self):
        runner = CliRunner()

        with patch(
            "jina_cli.main.api.classify",
            return_value=[{"prediction": "positive", "score": 0.9}],
        ) as classify:
            result = runner.invoke(
                cli,
                ["classify", "great movie", "--labels", "positive,negative"],
            )

        assert result.exit_code == 0
        assert "positive (0.9000)" in result.output
        classify.assert_called_once()
        _, kwargs = classify.call_args
        assert classify.call_args.args == (["great movie"], ["positive", "negative"])
        assert kwargs["model"] == "jina-embeddings-v5-text-small"
        assert kwargs["timeout"] is None

    def test_expand_prints_each_query(self):
        runner = CliRunner()

        with patch(
            "jina_cli.main.api.expand_query",
            return_value=["embedding models", {"query": "vector search"}],
        ) as expand_query:
            result = runner.invoke(cli, ["expand", "search"])

        assert result.exit_code == 0
        assert result.output.splitlines() == ["embedding models", "vector search"]
        expand_query.assert_called_once()
        _, kwargs = expand_query.call_args
        assert expand_query.call_args.args == ("search",)
        assert kwargs["timeout"] is None

    def test_screenshot_without_output_prints_image_url(self):
        runner = CliRunner()

        with patch(
            "jina_cli.main.api.screenshot_url",
            return_value={
                "data": {"screenshotUrl": "https://cdn.example/screenshot.png"}
            },
        ) as screenshot_url:
            result = runner.invoke(cli, ["screenshot", "https://example.com"])

        assert result.exit_code == 0
        assert result.output.strip() == "https://cdn.example/screenshot.png"
        screenshot_url.assert_called_once()
        _, kwargs = screenshot_url.call_args
        assert screenshot_url.call_args.args == ("https://example.com",)
        assert kwargs["full_page"] is False
        assert kwargs["timeout"] is None

    def test_global_timeout_reaches_subcommand(self):
        runner = CliRunner()

        with patch(
            "jina_cli.main.api.read_url", return_value="Example Domain"
        ) as read_url:
            result = runner.invoke(
                cli,
                ["--timeout", "45", "read", "https://example.com"],
            )

        assert result.exit_code == 0
        assert "Example Domain" in result.output
        read_url.assert_called_once()
        _, kwargs = read_url.call_args
        assert read_url.call_args.args == ("https://example.com",)
        assert kwargs["with_links"] is False
        assert kwargs["with_images"] is False
        assert kwargs["as_json"] is False
        assert kwargs["timeout"] == 45.0
