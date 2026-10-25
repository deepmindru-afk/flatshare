//! Listening sockets the gateway inherits, as systemd's socket activation passes them (`LISTEN_FDS`), so that they
//! outlive any one gateway process: in process, and with the binary as its unit runs it.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use command_fds::{CommandFdExt, FdMapping};
use lh_gateway::{Config, Inherited, LogSink, Server};
use tokio::net::UnixDatagram;

use crate::common::*;

/// A chat completion of the agents tenant on its own connection, which closes after the answer.
fn raw_request() -> Vec<u8> {
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: gateway\r\nAuthorization: Bearer {AGENTS_KEY}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{CHAT}",
        CHAT.len()
    )
    .into_bytes()
}

/// Connects to `addr` now and sends a request, before any gateway accepts there; the thread gives the answer.
fn early_client(addr: SocketAddr) -> std::thread::JoinHandle<String> {
    let mut stream = TcpStream::connect(addr).expect("a listening socket queues the connection");
    stream.write_all(&raw_request()).unwrap();
    std::thread::spawn(move || {
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let mut answer = String::new();
        stream.read_to_string(&mut answer).unwrap();
        answer
    })
}

#[tokio::test]
async fn inherited_sockets_serve_the_connections_made_before_and_stay_after_the_stop() {
    let mock = Mock::start(Script::Json("{}")).await;
    let tcp = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = tcp.local_addr().unwrap();
    let path = socket_path();
    let unix = UnixListener::bind(&path).unwrap();
    let early = early_client(addr);
    let config = Config::from_toml(&listeners_toml(&mock.upstream, addr, &path, "")).unwrap();
    let mut gw = Gateway::from_inherited(config, Inherited::default().tcp(tcp).unix(unix)).await;
    let answer = tokio::task::spawn_blocking(move || early.join().unwrap())
        .await
        .unwrap();
    assert!(answer.starts_with("HTTP/1.1 200 OK\r\n"), "{answer}");
    let reply = call(&gw.unix, "POST", "/v1/chat/completions", &[], CHAT).await;
    assert_eq!(reply.status, 200);
    gw.stopped().await;
    assert!(
        path.exists(),
        "an inherited socket's file is its owner's to remove"
    );
    std::fs::remove_file(&path).unwrap();
}

#[tokio::test]
async fn a_listener_without_an_inherited_socket_is_bound_and_an_unnamed_socket_is_refused() {
    let mock = Mock::start(Script::Json("{}")).await;
    let path = socket_path();
    let unix = UnixListener::bind(&path).unwrap();
    // The TCP listener is the gateway's own (127.0.0.1:0); the Unix one is inherited.
    let config = Config::from_toml(&gateway_toml(mock.addr, &path, "")).unwrap();
    let mut gw = Gateway::from_inherited(config.clone(), Inherited::default().unix(unix)).await;
    assert_eq!(call(&gw.tcp, "GET", "/health", &[], "").await.status, 200);
    assert_eq!(call(&gw.unix, "GET", "/health", &[], "").await.status, 200);
    gw.stopped().await;

    let stray = TcpListener::bind("127.0.0.1:0").unwrap();
    let stray_addr = stray.local_addr().unwrap();
    let err = Server::bind_with(config, Arc::new(LogSink), Inherited::default().tcp(stray))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        err.to_string().contains(&format!(
            "inherited sockets that no [[listen]] entry names, so that nothing would accept on them: tcp {stray_addr}"
        )),
        "{err}"
    );
    // The refused start bound nothing: the socket file the first gateway left is as it was.
    assert!(path.exists());
    std::fs::remove_file(&path).unwrap();
}

/// The binary's configuration file, in the target's temporary directory.
fn config_file(toml: &str) -> PathBuf {
    let path = socket_path().with_extension("toml");
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(path.file_name().unwrap());
    std::fs::write(&path, toml).unwrap();
    path
}

/// Sends SIGTERM to `pid`.
fn terminate(pid: u32) {
    let status = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .unwrap();
    assert!(status.success());
}

#[tokio::test]
async fn the_binary_serves_the_sockets_systemd_passes_and_says_when_it_is_ready() {
    let mock = Mock::start(Script::Json("{}")).await;
    let tcp = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = tcp.local_addr().unwrap();
    let path = socket_path();
    let unix = UnixListener::bind(&path).unwrap();
    let early = early_client(addr);
    let manager_path = socket_path();
    let manager = UnixDatagram::bind(&manager_path).unwrap();
    let config = config_file(&listeners_toml(&mock.upstream, addr, &path, ""));

    // As systemd starts a socket-activated service: the sockets from fd 3 on, LISTEN_FDS, and a NOTIFY_SOCKET.
    let mut command = Command::new(env!("CARGO_BIN_EXE_lh-gateway"));
    command
        .arg("--config")
        .arg(&config)
        .env("LISTEN_FDS", "2")
        .env_remove("LISTEN_PID")
        .env("NOTIFY_SOCKET", &manager_path)
        .stdout(Stdio::null())
        .fd_mappings(vec![
            FdMapping {
                parent_fd: OwnedFd::from(tcp),
                child_fd: 3,
            },
            FdMapping {
                parent_fd: OwnedFd::from(unix),
                child_fd: 4,
            },
        ])
        .unwrap();
    let mut child = command.spawn().unwrap();
    drop(command); // the child holds the sockets now

    let mut buf = [0; 256];
    let n = tokio::time::timeout(Duration::from_secs(10), manager.recv(&mut buf))
        .await
        .expect("READY=1 within 10 s")
        .unwrap();
    assert_eq!(&buf[..n], b"READY=1");
    let answer = tokio::task::spawn_blocking(move || early.join().unwrap())
        .await
        .unwrap();
    assert!(answer.starts_with("HTTP/1.1 200 OK\r\n"), "{answer}");
    let reply = call(
        &Target::Unix(path.clone()),
        "POST",
        "/v1/chat/completions",
        &[],
        CHAT,
    )
    .await;
    assert_eq!(reply.status, 200);
    assert_eq!(
        mock.last().header("x-vllm-priority"),
        Some("0"),
        "the voice socket's tenant"
    );

    terminate(child.id());
    let status = tokio::task::spawn_blocking(move || child.wait().unwrap())
        .await
        .unwrap();
    assert!(status.success(), "{status}");
    assert!(path.exists(), "the socket unit's file stays");
    for file in [&path, &manager_path, &config] {
        std::fs::remove_file(file).unwrap();
    }
}
