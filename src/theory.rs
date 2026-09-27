//! Closed-form reference results.
//!
//! This module is the "answer key". It must never call into the
//! implementation modules ([`crate::cfar`], [`crate::matched_filter`], ...),
//! so that a bug in the implementation cannot silently propagate into the
//! values it is checked against.

pub mod exponential;
