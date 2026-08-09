// SPDX-License-Identifier: AGPL-3.0-only

pub fn run_review() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running Wenlan Review");
}
