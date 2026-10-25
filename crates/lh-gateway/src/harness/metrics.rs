//! The harness's metric families: the admission point's, and those of the realtime sessions' turns that policy
//! leaseholder follows. A gateway with an admission point registers them in its registry, after its own; one without
//! has none of them, so a scraper reads them as absent, not zero.

use std::sync::atomic::AtomicU64;
use std::time::Duration;

use lh_core::admission::Deferral;
use lh_proto::SloClass;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::{Registry, Unit};

use super::realtime::Phase;
use crate::metrics::{GaugeGuard, Histograms, LATENCY_BUCKETS, create};

/// Why the admission point refused a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// Its class's queue was full (429).
    QueueFull,
    /// It waited `queue_timeout_s` without being admitted (503).
    Timeout,
}

impl Refusal {
    const ALL: [Self; 2] = [Self::QueueFull, Self::Timeout];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QueueFull => "queue_full",
            Self::Timeout => "timeout",
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ClassLabels {
    class: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RefusalLabels {
    class: &'static str,
    reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ReasonLabels {
    reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct OversizeLabels {
    oversize: bool,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct StateLabels {
    state: &'static str,
}

/// The harness's series: cloning a handle shares every one.
#[derive(Clone, Debug)]
pub(crate) struct Metrics {
    admitted: Family<ClassLabels, Counter>,
    waiting: Family<ClassLabels, Gauge>,
    admission_wait: Histograms<ClassLabels>,
    refused: Family<RefusalLabels, Counter>,
    resubmits: Family<ClassLabels, Counter>,
    discarded_tokens: Family<ClassLabels, Counter>,
    deferred: Family<ReasonLabels, Counter>,
    deferral: Histograms<ReasonLabels>,
    aged: Family<OversizeLabels, Counter>,
    degraded: Counter,
    false_alarms: Counter,
    false_alarm_token_seconds: Counter<f64, AtomicU64>,
    realtime_sessions: Family<StateLabels, Gauge>,
    state_timeouts: Family<StateLabels, Counter>,
}

impl Metrics {
    /// Every series, at zero.
    pub fn new() -> Self {
        let metrics = Self {
            admitted: Family::default(),
            waiting: Family::default(),
            admission_wait: Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS)),
            refused: Family::default(),
            resubmits: Family::default(),
            discarded_tokens: Family::default(),
            deferred: Family::default(),
            deferral: Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS)),
            aged: Family::default(),
            degraded: Counter::default(),
            false_alarms: Counter::default(),
            false_alarm_token_seconds: Counter::default(),
            realtime_sessions: Family::default(),
            state_timeouts: Family::default(),
        };
        for class in SloClass::ALL.map(SloClass::as_str) {
            for reason in Refusal::ALL.map(Refusal::as_str) {
                create(&metrics.refused, RefusalLabels { class, reason });
            }
            create(&metrics.admitted, ClassLabels { class });
            create(&metrics.waiting, ClassLabels { class });
            create(&metrics.admission_wait, ClassLabels { class });
            create(&metrics.resubmits, ClassLabels { class });
            create(&metrics.discarded_tokens, ClassLabels { class });
        }
        for reason in Deferral::ALL.map(Deferral::as_str) {
            create(&metrics.deferred, ReasonLabels { reason });
            create(&metrics.deferral, ReasonLabels { reason });
        }
        for oversize in [false, true] {
            create(&metrics.aged, OversizeLabels { oversize });
        }
        for state in Phase::ALL.map(Phase::as_str) {
            create(&metrics.realtime_sessions, StateLabels { state });
            if state != Phase::Idle.as_str() {
                create(&metrics.state_timeouts, StateLabels { state });
            }
        }
        metrics
    }

    /// Registers every series in `r`, the gateway's registry, in the order of the exposition.
    pub fn register(&self, r: &mut Registry) {
        r.register(
            "admission_admitted",
            "Requests the admission point let go upstream, by class",
            self.admitted.clone(),
        );
        r.register(
            "admission_waiting",
            "Requests waiting at the admission point",
            self.waiting.clone(),
        );
        r.register_with_unit(
            "admission_wait",
            "Time from reaching the admission point to being admitted",
            Unit::Seconds,
            self.admission_wait.clone(),
        );
        r.register(
            "admission_refused",
            "Requests the admission point refused, by class and reason",
            self.refused.clone(),
        );
        r.register(
            "admission_resubmits",
            "Requests sent upstream again after a preemption dropped their attempt",
            self.resubmits.clone(),
        );
        r.register(
            "admission_discarded_tokens",
            "Estimated prompt tokens of the upstream attempts preemptions dropped: at most the prefill they \
             discarded",
            self.discarded_tokens.clone(),
        );
        r.register(
            "admission_deferred",
            "Background requests policy leaseholder deferred, by what held them back when they arrived: the \
             headroom of the armed realtime sessions, or the critical window's prefill budget",
            self.deferred.clone(),
        );
        r.register_with_unit(
            "admission_deferral",
            "Time from arrival to admission of the background requests policy leaseholder deferred, by what \
             held them back when they arrived",
            Unit::Seconds,
            self.deferral.clone(),
        );
        r.register(
            "admission_aged",
            "Deferred requests admitted regardless after max_defer_s; oversize: their estimate alone was over \
             what the headroom leaves background requests",
            self.aged.clone(),
        );
        r.register(
            "admission_degraded",
            "Realtime requests whose reply policy leaseholder capped, because the background requests in flight \
             left less than one turn's headroom free",
            self.degraded.clone(),
        );
        r.register(
            "admission_false_alarms",
            "Headroom armings that no realtime request of their session followed: the session left the armed \
             states, timed out or closed first",
            self.false_alarms.clone(),
        );
        r.register(
            "admission_false_alarm_token_seconds",
            "Estimated background tokens the headroom held back during false alarms, times how long",
            self.false_alarm_token_seconds.clone(),
        );
        r.register(
            "realtime_sessions",
            "Realtime sessions policy leaseholder follows, by the state of their turn",
            self.realtime_sessions.clone(),
        );
        r.register(
            "realtime_state_timeouts",
            "Realtime sessions that stayed in a state of their turn for state_timeout_s without an event, by the \
             state: a lost or missing event",
            self.state_timeouts.clone(),
        );
    }

    /// Counts a request as waiting at the admission point until the returned guard drops.
    pub fn waiting(&self, class: SloClass) -> GaugeGuard {
        GaugeGuard::new(self.waiting.get_or_create_owned(&class_labels(class)))
    }

    /// A request was admitted after waiting `wait`.
    pub fn admitted(&self, class: SloClass, wait: Duration) {
        let labels = class_labels(class);
        self.admitted.get_or_create(&labels).inc();
        self.admission_wait
            .get_or_create(&labels)
            .observe(wait.as_secs_f64());
    }

    /// A preemption dropped an upstream attempt of `tokens` estimated prompt tokens.
    pub fn preempted(&self, class: SloClass, tokens: u64) {
        let labels = class_labels(class);
        self.discarded_tokens.get_or_create(&labels).inc_by(tokens);
    }

    /// A preempted request was sent upstream again.
    pub fn resubmitted(&self, class: SloClass) {
        let labels = class_labels(class);
        self.resubmits.get_or_create(&labels).inc();
    }

    /// Policy leaseholder deferred a background request.
    pub fn deferred(&self, reason: Deferral) {
        let labels = ReasonLabels {
            reason: reason.as_str(),
        };
        self.deferred.get_or_create(&labels).inc();
    }

    /// A deferred background request was admitted after waiting `wait`.
    pub fn deferral(&self, reason: Deferral, wait: Duration) {
        let labels = ReasonLabels {
            reason: reason.as_str(),
        };
        self.deferral
            .get_or_create(&labels)
            .observe(wait.as_secs_f64());
    }

    /// A deferred request was admitted regardless.
    pub fn aged(&self, oversize: bool) {
        self.aged.get_or_create(&OversizeLabels { oversize }).inc();
    }

    /// A realtime request's reply was capped.
    pub fn degraded(&self) {
        self.degraded.inc();
    }

    /// A headroom arming ended with no realtime request, having held back `token_seconds` of background work.
    pub fn false_alarm(&self, token_seconds: f64) {
        self.false_alarms.inc();
        self.false_alarm_token_seconds
            .inc_by(token_seconds.max(0.0));
    }

    /// The gauge of the realtime sessions in `phase`.
    pub fn realtime_sessions(&self, phase: Phase) -> Gauge {
        self.realtime_sessions.get_or_create_owned(&StateLabels {
            state: phase.as_str(),
        })
    }

    /// A realtime session's `phase` timed out.
    pub fn state_timeout(&self, phase: Phase) {
        let labels = StateLabels {
            state: phase.as_str(),
        };
        self.state_timeouts.get_or_create(&labels).inc();
    }

    pub fn refused(&self, class: SloClass, reason: Refusal) {
        let labels = RefusalLabels {
            class: class.as_str(),
            reason: reason.as_str(),
        };
        self.refused.get_or_create(&labels).inc();
    }

    /// The false alarms, and their token-seconds.
    #[cfg(test)]
    pub fn false_alarms(&self) -> (u64, f64) {
        (
            self.false_alarms.get(),
            self.false_alarm_token_seconds.get(),
        )
    }

    #[cfg(test)]
    pub fn state_timeouts(&self, phase: Phase) -> u64 {
        let labels = StateLabels {
            state: phase.as_str(),
        };
        self.state_timeouts
            .get(&labels)
            .map_or(0, |counter| counter.get())
    }
}

fn class_labels(class: SloClass) -> ClassLabels {
    ClassLabels {
        class: class.as_str(),
    }
}
