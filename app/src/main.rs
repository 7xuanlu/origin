// SPDX-License-Identifier: AGPL-3.0-only
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    #[cfg(feature = "review-fixtures")]
    wenlan_lib::run_review();

    #[cfg(not(feature = "review-fixtures"))]
    wenlan_lib::run()
}

// CI routing probe: app-only change must run app-check and no rust test lanes (PR closed unmerged).
