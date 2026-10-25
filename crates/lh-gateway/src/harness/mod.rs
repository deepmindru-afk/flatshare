//! E2's experiment harness: the admission point and the policies it runs, E2's baselines and Leaseholder's own
//! (policy leaseholder, E2 arm 9's gateway half), the realtime sessions' turns that policy leaseholder follows, and
//! their configuration (`[admission]`) and metric families.
//!
//! The configuration to deploy runs none of it (ADR-0006: the measured configuration ran without an admission
//! policy). Without an `[admission]` policy the gateway has no admission point and registers none of these metric
//! families. With one, the proxy reaches the harness only through a request's [`Ticket`], and the lifecycle events
//! through [`Admission::events`].

mod admission;
pub(crate) mod config;
mod metrics;
mod realtime;

pub(crate) use admission::{Admission, Ticket};
pub(crate) use metrics::{Metrics, Refusal};
