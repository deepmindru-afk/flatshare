//! An upstream on a Unix socket, as the engine serves on its GPU node and the tunnel on the control host: requests and
//! streams, pooled connections and their idle timeout, cancellation, and a socket nobody serves.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::BodyExt;

use crate::common::*;

#[tokio::test]
async fn requests_and_streams_reach_an_engine_on_a_unix_socket() {
    let chunks: Vec<Bytes> = [
        &b"data: {\"choices\":[{\"delta\":{\"content\":\"h\"},\"finish_reason\":null}]}\n\n"[..],
        b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":1}}\n\n",
        b"data: [DONE]\n\n",
    ]
    .into_iter()
    .map(Bytes::from_static)
    .collect();
    let expected: Vec<u8> = chunks.concat();
    let mock = Mock::start_unix(Script::Sse {
        chunks,
        gap: Duration::from_millis(5),
    })
    .await;
    let gw = Gateway::start_upstream(&mock.upstream, "").await;
    assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 200);

    let headers = [("x-session-id", "voice-RM_a1B2c3")];
    let reply = call(&gw.unix, "POST", "/v1/chat/completions", &headers, CHAT).await;
    assert_eq!(reply.status, 200);
    assert_eq!(reply.header("content-type"), Some("text/event-stream"));
    assert_eq!(&reply.body[..], &expected[..]);
    let seen = mock.last();
    assert_eq!(seen.path, "/v1/chat/completions");
    assert_eq!(seen.header("x-vllm-priority"), Some("0"));
    assert_eq!(seen.header("x-session-id"), Some("voice-RM_a1B2c3"));
    assert_eq!(seen.json()["cache_salt"], VOICE_SALT);

    let models = call(&gw.tcp, "GET", "/v1/models", &[BENCH_AUTH], "").await;
    assert_eq!(models.status, 200);
    assert_eq!(mock.last().path, "/v1/models");
}

#[tokio::test]
async fn connections_to_a_unix_upstream_are_pooled_until_the_idle_timeout() {
    let mock = Mock::start_unix(Script::Json("{}")).await;
    let gw = Gateway::start_upstream(&mock.upstream, "upstream_idle_timeout_s = 0.3").await;
    let send = async || {
        let reply = call(
            &gw.tcp,
            "POST",
            "/v1/chat/completions",
            &[AGENTS_AUTH],
            CHAT,
        )
        .await;
        assert_eq!(reply.status, 200);
    };
    for _ in 0..3 {
        send().await;
    }
    let accepted = || mock.connections.load(Ordering::SeqCst);
    assert_eq!(
        accepted(),
        1,
        "one pooled connection carried the three requests"
    );
    // Idle past upstream_idle_timeout_s, the connection is let go and the next request opens another.
    tokio::time::sleep(Duration::from_millis(700)).await;
    send().await;
    assert_eq!(accepted(), 2);
}

#[tokio::test]
async fn a_client_that_leaves_mid_stream_closes_the_unix_upstream_within_100_ms() {
    let chunk = Bytes::from_static(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n");
    let mut mock = Mock::start_unix(Script::Endless {
        chunk,
        gap: Duration::from_millis(20),
    })
    .await;
    let gw = Gateway::start_upstream(&mock.upstream, "").await;
    let (response, connection) = open(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    let mut body = response.into_body();
    for _ in 0..3 {
        body.frame().await.unwrap().unwrap();
    }
    let left = Instant::now();
    connection.abort();
    drop(body);
    let dropped = mock.next_drop(Duration::from_secs(2)).await;
    let after = dropped.saturating_duration_since(left);
    assert!(
        after < Duration::from_millis(100),
        "the upstream body was dropped {after:?} after the client left"
    );
    gw.await_metric(
        "lh_gateway_request_outcomes_total{class=\"background\",outcome=\"cancelled\"}",
        1.0,
    )
    .await;
}

#[tokio::test]
async fn a_unix_upstream_nobody_serves_is_502_and_not_ready() {
    // A socket file left by an engine that died, and a path with no file at all.
    let stale = socket_path();
    drop(std::os::unix::net::UnixListener::bind(&stale).unwrap());
    for (path, cause) in [
        (stale, "Connection refused"),
        (socket_path(), "No such file or directory"),
    ] {
        let gw = Gateway::start_upstream(&format!("unix:{}", path.display()), "").await;
        let reply = call(
            &gw.tcp,
            "POST",
            "/v1/chat/completions",
            &[AGENTS_AUTH],
            CHAT,
        )
        .await;
        assert_eq!(reply.status, 502);
        let error = &reply.json()["error"];
        assert_eq!(error["code"], "upstream_unavailable");
        let message = error["message"].as_str().unwrap();
        assert!(message.contains(cause), "{message}");
        gw.await_metric("lh_gateway_upstream_errors_total{kind=\"connect\"}", 1.0)
            .await;
        assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 503);
    }
}
