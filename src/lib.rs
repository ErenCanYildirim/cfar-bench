//! Radar signal processing primitives: waveform generation, matched
//! filtering, ambiguity analysis and CFAR detection.
//!
//! Every component is validated numerically against closed-form theory
//! (see [`theory`]) rather than by visual inspection.
//!
//! # Module layout
//!
//! | Module              | Contents                                           |
//! |---------------------|----------------------------------------------------|
//! | [`sim`]             | Synthetic data: noise, and later targets/scenes    |
//! | [`stats`]           | Estimators and hypothesis tests used for validation|
//! | [`theory`]          | Closed-form reference results (never shares logic with the code under test) |
//! | [`waveform`]        | Transmit waveforms (LFM, Barker)                   |
//! | [`matched_filter`]  | Pulse compression                                  |
//! | [`ambiguity`]       | Delay-Doppler ambiguity function                   |
//! | [`cfar`]            | CA-CFAR and OS-CFAR detectors                      |

pub mod ambiguity;
pub mod cfar;
pub mod matched_filter;
pub mod sim;
pub mod stats;
pub mod theory;
pub mod waveform;
