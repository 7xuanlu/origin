// SPDX-License-Identifier: Apache-2.0
//! Router submodules — query/content classification helpers.
//!
//! The `intent` module and the `bundle`/`keywords` modules that depend on
//! `sensor::vision` live in the app crate for now. They'll move into
//! origin-core once `sensor::vision::WindowOcrResult` + friends land here.
pub mod classify;
pub mod content_score;
