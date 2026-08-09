# Extract from origin monorepo

This repository was created on 2026-05-07T04:42:58Z by extracting
the `app/` Tauri crate plus the React frontend from
[7xuanlu/origin](https://github.com/7xuanlu/origin) at SHA `1be677bd26417c5ff1b33b449bc1e2922568c3ca`.

For commit history before that point, see the origin repo. The reason for the
split was documented in `docs/superpowers/decision_extract_tauri_to_origin_app_repo.md` <!-- drift-ok --> in the origin repo.

**Reversed 2026-07-20:** the app was folded back into the monorepo as the
`app` workspace crate (this repo). The 2026-05-07 split above is history.

License: AGPL-3.0-only (vs Apache-2.0 for the daemon and shared types).
