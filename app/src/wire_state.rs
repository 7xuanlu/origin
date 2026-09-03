// SPDX-License-Identifier: AGPL-3.0-only
//! `wire_state()` — the real, resolved wiring of Wenlan on this machine:
//! whether the daemon answers, which `wenlan-mcp` binary would actually be
//! written into a client config (plus the full candidate trail, missing
//! paths included), and per-client MCP routing. Single source of truth
//! behind the onboarding wizard's "Setting up" step and Settings →
//! Diagnostics.
//!
//! The bug this exists to make visible: the app once resolved `wenlan-mcp`
//! to a maintainer's cargo build output, wrote that absolute path into a
//! user's `claude_desktop_config.json`, and the binary was later deleted by
//! `cargo clean`. Claude Desktop failed with "cannot connect mcpserver
//! wenlan" and nothing in the app surfaced it. A trail that only lists paths
//! that exist can't show a missing one — so `mcp_binary_wire` below keeps
//! every candidate `mcp_config::wenlan_mcp_candidate_sources` returns,
//! existent or not.

use crate::api::WenlanClient;
use crate::mcp_config;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WireState {
    pub daemon: DaemonWire,
    pub mcp_binary: BinaryWire,
    pub clients: Vec<ClientWire>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DaemonWire {
    pub base_url: String,
    pub reachable: bool,
    pub version: Option<String>,
    pub error: Option<String>,
    /// Whether the daemon this app spawned is held by the kill-on-close job
    /// object — i.e. whether a hard kill of the app really ends it. `None`
    /// means this app owns no sidecar (a service holds the daemon, or it
    /// exited); it is not a stand-in for "bound". Carried here because a
    /// binding that failed is otherwise invisible outside the log, and an
    /// orphaned daemon is exactly the kind of wiring this command exists to
    /// make visible.
    pub sidecar_job_binding: Option<crate::daemon_start::JobBinding>,
    /// Whether this app ever started a sidecar without being able to measure
    /// whether launchd already owned the daemon. `false` is a measured "no";
    /// `true` means two daemons are possible for a reason nothing else on this
    /// surface would show. See `daemon_start::spawned_on_unknown_owner`.
    pub sidecar_spawned_on_unknown_owner: bool,
    /// What the most recent `daemon_start::stop_sidecar` established, or
    /// `None` if the app has not tried to stop one in this process. Carried
    /// for the same reason as `sidecar_job_binding`: a stop that could not
    /// confirm the daemon ended is otherwise invisible, and the next launch
    /// meets it as a port that is already held.
    pub last_sidecar_stop: Option<crate::daemon_start::SidecarStopOutcome>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BinaryWire {
    /// The command setup would write, or `null` when the binary search could
    /// not be completed and setup would therefore write NOTHING. A reader that
    /// prints this must handle the null; there is no placeholder string that
    /// would be honest here, because no command was chosen.
    pub command: Option<String>,
    pub args: Vec<String>,
    /// Present only alongside `command: null`: the candidates that could not be
    /// looked at, with the OS error, and the message the user is shown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unresolved: Option<UnresolvedBinary>,
    /// Inputs the candidate paths hang off that could not be determined, so
    /// those candidates were never built and never probed. These have NO entry
    /// in `candidates` -- that is exactly what makes them dangerous, and why
    /// they are carried separately rather than left to be inferred from a short
    /// trail.
    ///
    /// ROUND 5, D4: this used to live inside `unresolved`, i.e. only ever
    /// alongside `command: null`. That made it structurally impossible to
    /// report an undetermined input on a search that FOUND something -- and a
    /// find is precisely when the omission is invisible, because the command
    /// beside it looks like a complete answer. It sits here, beside `command`
    /// rather than under a failure, because it is a property of the SEARCH:
    /// non-empty with a command means "this command was chosen, and these
    /// inputs were never read"; empty with a command means the search really
    /// did cover everything it names.
    pub undetermined: Vec<mcp_config::UndeterminedInput>,
    pub candidates: Vec<BinaryCandidate>,
}

/// Why no command was chosen. Mirrors `mcp_config::McpEntryDecision::
/// PreserveExisting`, whose whole point is that no entry may be written.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UnresolvedBinary {
    pub message: String,
    /// Candidate PATHS that could not be looked at.
    pub unreadable: Vec<UnreadableCandidate>,
    // `undetermined` used to live here. It now lives on `BinaryWire` itself --
    // see the field comment there for why a failure that only exists on the
    // no-command path is a failure a successful search can hide.
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UnreadableCandidate {
    pub path: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BinaryCandidate {
    pub path: String,
    /// The measurement: `file`, `not_a_file`, `not_executable` with the reason,
    /// `absent`, or `unreadable` with the OS error.
    ///
    /// Round 4, defect F: an `exists: bool` used to sit beside this, and it was
    /// the old boolean collapse still on the wire — `false` for "absent", for
    /// "a directory is squatting on the name", and for "the OS refused to look".
    /// Any reader that took the easy field got the conflation back for free, and
    /// the Diagnostics panel did exactly that (it labelled an unreadable
    /// candidate "Missing"). Removed; readers match on `state`.
    pub state: mcp_config::CandidateProbe,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClientWire {
    pub client_type: String,
    pub name: String,
    /// ROUND 5, DEFECT 4: these four were `bool`, and every one of them was a
    /// `Path::exists()` or a `read_to_string(..).unwrap_or(false)` — `false`
    /// for "measured: no" AND for "the OS would not tell me". Readers match on
    /// [`mcp_config::Reading`], exactly as they already do on `CandidateProbe`
    /// for the binary trail, and for the same reason (round 4, defect F): a
    /// boolean beside a tri-state hands the conflation back to any reader that
    /// takes the shorter field, so there is no boolean.
    pub detected: mcp_config::Reading,
    /// `None` when the directory the path hangs off could not be determined.
    /// The row still exists — that is the point — but there is no path to show.
    pub config_path: Option<String>,
    pub has_raw_entry: mcp_config::Reading,
    /// Config holds BOTH a `wenlan` and a legacy `origin` raw entry — the
    /// raw+raw duplicate. Load-bearing for no-plugin clients (cursor,
    /// gemini_cli): they never trip the plugin+raw double-registration
    /// (`has_plugin && has_raw_entry`), so this is the only way their
    /// duplicate surfaces, and it routes to a fix that removes only `origin`.
    pub has_raw_duplicate: mcp_config::Reading,
    pub has_plugin: mcp_config::Reading,
    /// `plugin` | `config` | `skip` | `unknown`.
    pub route: String,
}

/// `route` for a client — what setup would do *now*, not what is already there.
///
/// The plugin always wins over a raw MCP entry, because every Wenlan plugin
/// declares its own `mcpServers`: writing `~/.claude.json` /
/// `[mcp_servers.wenlan]` / `claude_desktop_config.json` as well registers the
/// server twice.
///
/// `has_plugin` is load-bearing here, not decoration. Claude Desktop has two
/// independent plugin surfaces, and its *chat*-side plugin ships an MCP server
/// of its own — so a Desktop that already has the plugin must be `"skip"`, not
/// `"config"`. Routing it to `"config"` is exactly the double-registration that
/// broke a real machine (see commit 5d7a364); `claude_code` / `codex_cli` reach
/// the same conclusion via their own `"plugin"` arm.
/// ROUND 5, DEFECT 4, AT THE ROUTE. A route is an INSTRUCTION, so computing one
/// from an unread input is worse than reporting nothing: `!detected` used to
/// yield `"skip"` for a client whose config the OS refused to stat ("this isn't
/// here, do nothing" — about a client that may well be there), and an unread
/// `has_plugin` yielded `"config"` for a Claude Desktop that already has the
/// chat-side plugin, which is precisely the double registration that broke a
/// real machine. `"unknown"` is the fourth value: no instruction, because none
/// was established.
fn route_for(
    client_type: &str,
    detected: &mcp_config::Reading,
    has_plugin: &mcp_config::Reading,
) -> &'static str {
    use mcp_config::Reading;
    match detected {
        Reading::No => "skip",
        Reading::Unreadable { .. } => "unknown",
        Reading::Yes => {
            if client_type == "claude_code" || client_type == "codex_cli" {
                // These two route to their own plugin installer regardless of
                // what the plugin reading says, so an unread plugin cannot
                // mislead here.
                "plugin"
            } else {
                match has_plugin {
                    Reading::Yes => "skip",
                    Reading::No => "config",
                    Reading::Unreadable { .. } => "unknown",
                }
            }
        }
    }
}

/// Builds the daemon's wire state from an already-attempted health check —
/// pure, so it's testable without a live daemon. Always succeeds: an
/// unreachable daemon is `reachable: false` with `error` set, never a
/// propagated `Err` — surfacing exactly that without crashing is
/// `wire_state`'s whole point.
fn daemon_wire_for(
    base_url: String,
    health: Result<String, String>,
    sidecar_job_binding: Option<crate::daemon_start::JobBinding>,
    sidecar_spawned_on_unknown_owner: bool,
    last_sidecar_stop: Option<crate::daemon_start::SidecarStopOutcome>,
) -> DaemonWire {
    let (reachable, version, error) = match health {
        Ok(version) => (true, Some(version), None),
        Err(error) => (false, None, Some(error)),
    };
    DaemonWire {
        base_url,
        reachable,
        version,
        error,
        sidecar_job_binding,
        sidecar_spawned_on_unknown_owner,
        last_sidecar_stop,
    }
}

/// Builds the mcp-binary wire state from ONE resolution: the decision setup
/// would act on, and the probe readings that decision was made from.
///
/// ROUND 4, DEFECT F. The comment that used to sit here claimed the command and
/// the trail "can never disagree" because both came from the same *candidate
/// list*. That was false, and the falseness was the bug: the entry was resolved
/// by one probe pass inside `mcp_config::wenlan_mcp_entry()`, and then this
/// function ran a SECOND, independent probe pass over the same paths. Two
/// passes are two instants. A permission that flipped in between produced
/// `command: npx` printed beside `state: file`, or a local command beside
/// `state: unreadable` — a diagnostics trail contradicting the decision it was
/// supposed to explain, which is worse than no trail.
///
/// Now the caller does one pass and hands both halves here, so the trail
/// describes the decision that was actually made — by construction, not by
/// assertion.
fn mcp_binary_wire(
    decision: mcp_config::McpEntryDecision,
    trail: Vec<mcp_config::ProbedCandidate>,
) -> BinaryWire {
    let candidates = trail
        .into_iter()
        .map(|c| BinaryCandidate {
            path: c.path.to_string_lossy().to_string(),
            state: c.state,
            source: c.source.to_string(),
        })
        .collect();
    // Round 5, D4: `undetermined` comes out of BOTH arms. A chosen command and
    // an input that could not be read are independent facts, and pairing them
    // only with the failure arm is what let a `Found` swallow the second one.
    let (command, args, unresolved, undetermined) = match decision {
        mcp_config::McpEntryDecision::Write {
            entry,
            undetermined,
        } => (Some(entry.command), entry.args, None, undetermined),
        mcp_config::McpEntryDecision::PreserveExisting { unmeasured } => {
            let message = mcp_config::unresolved_message(&unmeasured);
            let mcp_config::Unmeasured {
                unreadable,
                undetermined,
            } = unmeasured;
            (
                None,
                Vec::new(),
                Some(UnresolvedBinary {
                    message,
                    unreadable: unreadable
                        .into_iter()
                        .map(|(path, error)| UnreadableCandidate {
                            path: path.to_string_lossy().to_string(),
                            error,
                        })
                        .collect(),
                }),
                undetermined,
            )
        }
    };
    BinaryWire {
        command,
        args,
        unresolved,
        undetermined,
        candidates,
    }
}

/// Reshapes one `mcp_config::detect_mcp_clients()` row for the wire.
///
/// PURE — it reads nothing. It used to call `client_plugin_enabled`,
/// `client_config_has_raw_entry` and `client_config_has_both_raw_entries` here,
/// re-reading the very files `detect_mcp_clients` had just read, so `detected`
/// and `has_raw_entry` on one row could come from different instants and
/// contradict each other. That is round 4's defect F in the client half, and it
/// is fixed the same way it was for the binary trail: one pass reads, and
/// everything downstream reshapes what that pass returned.
fn client_wire(client: &mcp_config::McpClient) -> ClientWire {
    ClientWire {
        route: route_for(&client.client_type, &client.detected, &client.has_plugin).to_string(),
        client_type: client.client_type.clone(),
        name: client.name.clone(),
        detected: client.detected.clone(),
        has_raw_entry: client.has_raw_entry.clone(),
        has_raw_duplicate: client.has_raw_duplicate.clone(),
        has_plugin: client.has_plugin.clone(),
        config_path: client.config_path.clone(),
    }
}

/// Assembles the full `WireState` for the real machine. The mcp-binary half is
/// ONE call to `mcp_config::wenlan_mcp_decision()`, which returns the decision
/// and the probe readings behind it together — see `mcp_binary_wire` for why
/// two separate probe passes were a defect and not an optimisation. Never
/// errors — a down daemon shows up as `daemon.reachable: false`, and an
/// unresolvable binary as `mcp_binary.command: null`, not a failed command.
pub async fn compute(client: &WenlanClient) -> WireState {
    let health = client.health().await.map(|h| h.version);
    let daemon = daemon_wire_for(
        client.base_url().to_string(),
        health,
        crate::daemon_start::sidecar_job_binding(),
        crate::daemon_start::spawned_on_unknown_owner(),
        crate::daemon_start::last_sidecar_stop(),
    );

    let (decision, trail) = mcp_config::wenlan_mcp_decision();
    let mcp_binary = mcp_binary_wire(decision, trail);

    let clients = mcp_config::detect_mcp_clients()
        .iter()
        .map(client_wire)
        .collect();

    WireState {
        daemon,
        mcp_binary,
        clients,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcp_config::Reading;
    use std::path::{Path, PathBuf};

    // ── route_for ───────────────────────────────────────────────────────

    #[test]
    fn plugin_clients_route_to_plugin_never_config() {
        assert_eq!(
            route_for("claude_code", &Reading::Yes, &Reading::No),
            "plugin"
        );
        assert_eq!(
            route_for("codex_cli", &Reading::Yes, &Reading::No),
            "plugin"
        );
    }

    #[test]
    fn non_plugin_clients_route_to_config() {
        assert_eq!(route_for("cursor", &Reading::Yes, &Reading::No), "config");
        assert_eq!(
            route_for("claude_desktop", &Reading::Yes, &Reading::No),
            "config"
        );
        assert_eq!(
            route_for("gemini_cli", &Reading::Yes, &Reading::No),
            "config"
        );
    }

    #[test]
    fn undetected_clients_route_to_skip_regardless_of_type() {
        assert_eq!(route_for("claude_code", &Reading::No, &Reading::No), "skip");
        assert_eq!(route_for("cursor", &Reading::No, &Reading::No), "skip");
    }

    /// The bug that broke a real machine. Claude Desktop's *chat*-side plugin
    /// ships its own MCP server, so a Desktop that already has the plugin must
    /// not also be written a raw `claude_desktop_config.json` entry — that is
    /// the double registration. `route` has to see `has_plugin` to know this;
    /// a route computed from `(client_type, detected)` alone cannot.
    #[test]
    fn a_client_that_already_has_the_plugin_is_never_routed_to_config() {
        assert_eq!(
            route_for("claude_desktop", &Reading::Yes, &Reading::Yes),
            "skip"
        );
        assert_eq!(route_for("cursor", &Reading::Yes, &Reading::Yes), "skip");
        // …and without the plugin it still needs its config written.
        assert_eq!(
            route_for("claude_desktop", &Reading::Yes, &Reading::No),
            "config"
        );
    }

    // ── mcp_binary_wire ─────────────────────────────────────────────────

    const TEST_NPM_PACKAGE: &str = "wenlan-mcp@^0.12.0";

    /// Build the wire exactly as `compute` does — ONE resolution, decision and
    /// trail from the same readings — but against an explicit candidate tree.
    fn wire_for(
        home: Option<&Path>,
        dev_bin: Option<&str>,
        exe_dir: Option<&Path>,
        probe: impl Fn(&Path) -> mcp_config::CandidateProbe,
    ) -> BinaryWire {
        let (decision, trail) =
            mcp_config::wenlan_mcp_decision_for(home, dev_bin, exe_dir, probe, TEST_NPM_PACKAGE);
        mcp_binary_wire(decision, trail)
    }

    /// The candidate the trail must name, with the platform's executable
    /// suffix. Spelled from `std::env::consts::EXE_SUFFIX` so this stays an
    /// independent check on `wenlan_mcp_candidate_sources` rather than a
    /// restatement of it.
    fn expected_candidate(home: &Path, rel_dir: &str) -> PathBuf {
        home.join(rel_dir)
            .join(format!("wenlan-mcp{}", std::env::consts::EXE_SUFFIX))
    }

    #[test]
    fn candidate_trail_keeps_measured_absent_paths() {
        let home = PathBuf::from("/Users/someone");
        let wire = wire_for(Some(&home), None, None, |_| {
            mcp_config::CandidateProbe::Absent
        });

        // ~/.wenlan/bin/wenlan-mcp and ~/.cargo/bin/wenlan-mcp — neither exists.
        assert_eq!(wire.candidates.len(), 2);
        assert!(
            wire.candidates
                .iter()
                .all(|c| c.state == mcp_config::CandidateProbe::Absent),
            "a candidate measured absent must still be in the trail: {:?}",
            wire.candidates
        );
        for (rel_dir, source) in [(".wenlan/bin", "installed"), (".cargo/bin", "cargo")] {
            let expected = expected_candidate(&home, rel_dir);
            assert!(
                wire.candidates
                    .iter()
                    .any(|c| Path::new(&c.path) == expected && c.source == source),
                "the {source} candidate is not {} on this platform: {:?}",
                expected.display(),
                wire.candidates
            );
        }
    }

    #[test]
    fn candidate_trail_reports_file_when_the_probe_says_so() {
        let home = PathBuf::from("/Users/someone");
        let installed = expected_candidate(&home, ".wenlan/bin");
        let target = installed.clone();
        let wire = wire_for(Some(&home), None, None, move |p| {
            if p == target {
                mcp_config::CandidateProbe::File
            } else {
                mcp_config::CandidateProbe::Absent
            }
        });

        let installed_candidate = wire
            .candidates
            .iter()
            .find(|c| c.source == "installed")
            .expect("installed candidate present");
        assert_eq!(installed_candidate.state, mcp_config::CandidateProbe::File);
        let cargo_candidate = wire
            .candidates
            .iter()
            .find(|c| c.source == "cargo")
            .expect("cargo candidate present");
        assert_eq!(cargo_candidate.state, mcp_config::CandidateProbe::Absent);
        // The command is the file the trail says was found — same reading.
        // Compared as paths: `join` mixes separators on Windows, and the
        // question is which file was chosen, not how it was spelled.
        assert_eq!(
            wire.command.as_deref().map(Path::new),
            Some(installed.as_path())
        );
    }

    #[test]
    fn mcp_binary_wire_carries_the_resolved_command_and_args_through() {
        let wire = wire_for(None, None, None, |_| mcp_config::CandidateProbe::Absent);
        assert_eq!(wire.command.as_deref(), Some("npx"));
        assert_eq!(wire.args, vec!["-y", TEST_NPM_PACKAGE]);
        assert!(wire.unresolved.is_none());
    }

    /// DEFECT F. `compute` used to resolve the entry with one probe pass and
    /// then RE-PROBE every candidate to build the trail. Two passes are two
    /// instants: a permission that flipped in between produced `command: npx`
    /// printed beside `state: file`, or a local command beside `state:
    /// unreadable` — a trail contradicting the decision it exists to explain.
    /// The comment above the function claimed the two "can never disagree".
    ///
    /// The fixture flips exactly that way: the first look at the installed
    /// candidate says `File`, every later look says `Unreadable`. The probe
    /// counts its own calls, so the assertion is on what the code under test
    /// did, not on anything this test computed.
    #[test]
    fn the_trail_carries_the_decisions_own_probe_readings() {
        use std::collections::HashMap;
        use std::sync::Mutex;

        let home = PathBuf::from("/Users/someone");
        let installed = expected_candidate(&home, ".wenlan/bin");
        let target = installed.clone();
        let looks: Mutex<HashMap<PathBuf, usize>> = Mutex::new(HashMap::new());

        let wire = {
            let probe = |p: &Path| {
                let mut looks = looks.lock().unwrap();
                let n = looks.entry(p.to_path_buf()).or_insert(0);
                *n += 1;
                if p != target {
                    return mcp_config::CandidateProbe::Absent;
                }
                if *n == 1 {
                    mcp_config::CandidateProbe::File
                } else {
                    // The ACL changed, the network share dropped, the AV
                    // grabbed the file — whatever it was, it happened between
                    // the two passes the old code made.
                    mcp_config::CandidateProbe::Unreadable {
                        error: "Access is denied. (os error 5)".to_string(),
                    }
                }
            };
            wire_for(Some(&home), None, None, probe)
        };

        assert_eq!(
            looks.lock().unwrap().get(&installed).copied(),
            Some(1),
            "every candidate must be probed ONCE; a second pass is a second instant and is how \
             the trail came to contradict the command"
        );

        let installed_candidate = wire
            .candidates
            .iter()
            .find(|c| c.source == "installed")
            .expect("installed candidate present");
        assert_eq!(
            installed_candidate.state,
            mcp_config::CandidateProbe::File,
            "the trail reported a state the decision never saw"
        );
        assert_eq!(
            wire.command.as_deref().map(Path::new),
            Some(installed.as_path()),
            "the command must be the candidate the trail says was found"
        );
    }

    /// DEFECT D on the wire. An unresolvable search writes nothing, so there is
    /// no command to show. `command: null` is the honest rendering; a
    /// placeholder string would put a command on screen that setup would never
    /// write.
    #[test]
    fn an_unresolvable_search_puts_no_command_on_the_wire() {
        let home = PathBuf::from("/Users/someone");
        let wire = wire_for(Some(&home), None, None, |_| {
            mcp_config::CandidateProbe::Unreadable {
                error: "Access is denied. (os error 5)".to_string(),
            }
        });

        assert_eq!(wire.command, None);
        assert!(wire.args.is_empty());
        let unresolved = wire.unresolved.as_ref().expect("unresolved detail present");
        assert_eq!(unresolved.unreadable.len(), 2);
        assert!(unresolved.message.contains("unchanged"));

        let json = serde_json::to_value(&wire).unwrap();
        assert!(json["command"].is_null());
        assert_eq!(
            json["unresolved"]["unreadable"][0]["error"],
            "Access is denied. (os error 5)"
        );
    }

    // ── daemon_wire_for ─────────────────────────────────────────────────

    /// The daemon wire with nothing of ours to report: no sidecar binding, a
    /// measured "never spawned on an unknown owner", and no stop attempt yet.
    fn plain_daemon_wire(health: Result<String, String>) -> DaemonWire {
        daemon_wire_for(
            "http://127.0.0.1:7878".to_string(),
            health,
            None,
            false,
            None,
        )
    }

    #[test]
    fn daemon_wire_for_reachable_carries_version_and_no_error() {
        let wire = plain_daemon_wire(Ok("0.12.0".to_string()));
        assert!(wire.reachable);
        assert_eq!(wire.version.as_deref(), Some("0.12.0"));
        assert!(wire.error.is_none());
    }

    #[test]
    fn daemon_wire_for_unreachable_carries_error_and_no_version() {
        let wire = plain_daemon_wire(Err("connection refused".to_string()));
        assert!(!wire.reachable);
        assert!(wire.version.is_none());
        assert_eq!(wire.error.as_deref(), Some("connection refused"));
    }

    /// A sidecar the job object refused must reach the wire as `unbound` with
    /// its reason, not as a missing field a reader would take for "fine". A
    /// healthy daemon says nothing about whether killing the app would end
    /// it, so the two facts travel separately.
    #[test]
    fn daemon_wire_carries_an_unbound_sidecar_rather_than_hiding_it() {
        let unbound = crate::daemon_start::JobBinding::Unbound {
            reason: "AssignProcessToJobObject(4321) failed: 5".to_string(),
        };
        let wire = daemon_wire_for(
            "http://127.0.0.1:7878".to_string(),
            Ok("0.12.0".to_string()),
            Some(unbound.clone()),
            false,
            None,
        );
        assert!(wire.reachable, "the daemon is up; that is not the question");
        assert_eq!(wire.sidecar_job_binding.as_ref(), Some(&unbound));

        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["sidecar_job_binding"]["state"], "unbound");
        assert_eq!(
            json["sidecar_job_binding"]["reason"],
            "AssignProcessToJobObject(4321) failed: 5"
        );

        // No sidecar of ours in the slot is `null` — "nothing to speak for",
        // never the reassuring `bound`.
        let no_sidecar = plain_daemon_wire(Ok("0.12.0".to_string()));
        assert!(serde_json::to_value(&no_sidecar).unwrap()["sidecar_job_binding"].is_null());
    }

    /// A2's record. A sidecar started while launchd ownership could not be
    /// measured has to be visible somewhere a user or a bug report can reach;
    /// a `log::warn!` inside the probe was not that place, and the probe's old
    /// `false` return told every caller the opposite of the truth.
    #[test]
    fn daemon_wire_reports_a_sidecar_started_on_an_unmeasured_owner() {
        let wire = daemon_wire_for(
            "http://127.0.0.1:7878".to_string(),
            Ok("0.12.0".to_string()),
            None,
            true,
            None,
        );
        assert!(wire.sidecar_spawned_on_unknown_owner);
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["sidecar_spawned_on_unknown_owner"], true);

        // …and the ordinary case says so as a measured negative, not by
        // omission.
        let clean = plain_daemon_wire(Ok("0.12.0".to_string()));
        assert!(!clean.sidecar_spawned_on_unknown_owner);
        assert_eq!(
            serde_json::to_value(&clean).unwrap()["sidecar_spawned_on_unknown_owner"],
            false
        );
    }

    /// A3's record. A stop that could not establish the daemon ended is the
    /// state `docs/cross-platform.md` used to claim was impossible; it reaches
    /// the wire with its reason, and it is not the same JSON as `ended`.
    #[test]
    fn daemon_wire_carries_the_last_sidecar_stop_outcome_with_its_reason() {
        let unmeasured = crate::daemon_start::SidecarStopOutcome::CouldNotMeasure {
            reason: "the sidecar's start time was never captured".to_string(),
        };
        let wire = daemon_wire_for(
            "http://127.0.0.1:7878".to_string(),
            Err("connection refused".to_string()),
            None,
            false,
            Some(unmeasured.clone()),
        );
        assert_eq!(wire.last_sidecar_stop.as_ref(), Some(&unmeasured));
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["last_sidecar_stop"]["outcome"], "could_not_measure");
        assert_eq!(
            json["last_sidecar_stop"]["reason"],
            "the sidecar's start time was never captured"
        );

        let ended = daemon_wire_for(
            "http://127.0.0.1:7878".to_string(),
            Err("connection refused".to_string()),
            None,
            false,
            Some(crate::daemon_start::SidecarStopOutcome::Ended),
        );
        assert_eq!(
            serde_json::to_value(&ended).unwrap()["last_sidecar_stop"]["outcome"],
            "ended"
        );

        // Never stopped one in this process: `null`, not a stand-in for
        // `ended`.
        assert!(
            serde_json::to_value(plain_daemon_wire(Ok("0.12.0".to_string()))).unwrap()
                ["last_sidecar_stop"]
                .is_null()
        );
    }

    /// A4's trail half. `exists: false` conflated "not there" with "could not
    /// look" and with "a directory is sitting at that path"; the trail is a
    /// diagnostic surface, so every state has to reach it — and, round 4, the
    /// `exists` bool that re-created the conflation for any reader who took it
    /// is gone from the wire entirely.
    #[test]
    fn candidate_trail_separates_absence_from_a_failed_look_and_a_directory() {
        let home = PathBuf::from("/Users/someone");
        let installed = expected_candidate(&home, ".wenlan/bin");
        let denied = installed.clone();
        let wire = wire_for(Some(&home), None, None, move |p| {
            if p == denied {
                mcp_config::CandidateProbe::Unreadable {
                    error: "Access is denied. (os error 5)".to_string(),
                }
            } else {
                mcp_config::CandidateProbe::NotAFile
            }
        });

        let installed_candidate = wire
            .candidates
            .iter()
            .find(|c| c.source == "installed")
            .expect("installed candidate present");
        assert_eq!(
            installed_candidate.state,
            mcp_config::CandidateProbe::Unreadable {
                error: "Access is denied. (os error 5)".to_string()
            },
            "the trail must show WHY the candidate was skipped, not just that it was"
        );

        let cargo_candidate = wire
            .candidates
            .iter()
            .find(|c| c.source == "cargo")
            .expect("cargo candidate present");
        assert_eq!(cargo_candidate.state, mcp_config::CandidateProbe::NotAFile);

        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["candidates"][0]["state"]["kind"], "unreadable");
        assert_eq!(
            json["candidates"][0]["state"]["error"],
            "Access is denied. (os error 5)"
        );
        assert_eq!(json["candidates"][1]["state"]["kind"], "not_a_file");
        assert!(
            json["candidates"][0].get("exists").is_none(),
            "the `exists` boolean is the old collapse; it must not be back on the wire"
        );
    }

    // ── compute() end to end against an unreachable daemon ─────────────

    /// Port 1 is a privileged port nothing listens on in this sandbox, so the
    /// connection is refused immediately — no live server needed, and no
    /// dependence on the real `WENLAN_PORT`/`ORIGIN_PORT` env vars (which
    /// `WenlanClient::new()` reads and which are process-global, shared with
    /// every other test in this binary).
    #[tokio::test]
    async fn compute_never_errors_when_the_daemon_is_unreachable() {
        let client = WenlanClient::with_base_url("http://127.0.0.1:1".to_string());
        let wire = compute(&client).await;

        assert!(!wire.daemon.reachable);
        assert!(wire.daemon.error.is_some());
        assert_eq!(wire.daemon.base_url, "http://127.0.0.1:1");
        // Reaching this line at all proves compute() returned instead of
        // panicking or being unreachable through a propagated Err — its
        // signature (`-> WireState`, not `-> Result<WireState, _>`) makes
        // that structurally true, and this test exercises the down-daemon
        // path that would have to trigger it.
    }

    // ── client_wire ──────────────────────────────────────────────────────

    fn cursor_client(detected: Reading, has_raw_entry: Reading) -> mcp_config::McpClient {
        mcp_config::McpClient {
            name: "Cursor".to_string(),
            client_type: "cursor".to_string(),
            config_path: Some("/nonexistent/mcp.json".to_string()),
            detected,
            already_configured: has_raw_entry.clone(),
            has_raw_entry,
            has_raw_duplicate: Reading::No,
            has_plugin: Reading::No,
        }
    }

    #[test]
    fn client_wire_carries_detect_mcp_clients_fields_through() {
        let wire = client_wire(&cursor_client(Reading::Yes, Reading::No));
        assert_eq!(wire.client_type, "cursor");
        assert_eq!(wire.name, "Cursor");
        assert_eq!(wire.detected, Reading::Yes);
        assert_eq!(wire.config_path.as_deref(), Some("/nonexistent/mcp.json"));
        assert_eq!(wire.route, "config");
        assert_eq!(wire.has_raw_entry, Reading::No);
        assert_eq!(wire.has_raw_duplicate, Reading::No);
        assert_eq!(wire.has_plugin, Reading::No);
    }

    /// A no-plugin client whose config carries both `wenlan` and `origin`
    /// surfaces the raw+raw duplicate — the case plugin+raw detection can never
    /// reach for cursor/gemini_cli. Staged on the `McpClient` rather than on
    /// disk, because `client_wire` no longer reads anything: it used to re-read
    /// the same files `detect_mcp_clients` had just read, which is round 4's
    /// defect F in the client half (two instants, one row).
    #[test]
    fn client_wire_carries_a_raw_duplicate_through() {
        let mut client = cursor_client(Reading::Yes, Reading::Yes);
        client.has_raw_duplicate = Reading::Yes;
        let wire = client_wire(&client);
        assert_eq!(wire.has_raw_duplicate, Reading::Yes);
        assert_eq!(wire.has_raw_entry, Reading::Yes);
        assert_eq!(wire.has_plugin, Reading::No);
    }

    /// ROUND 5, DEFECT 4, AT THE ROUTE. A route is an instruction. Computing
    /// one from a reading that failed hands the user an action derived from
    /// nothing — `"skip"` ("this client isn't here") for a config the OS
    /// refused to stat, and `"config"` ("write a raw entry") for a Claude
    /// Desktop whose chat-side plugin could not be read, which is exactly the
    /// double registration that broke a real machine.
    #[test]
    fn a_reading_that_failed_produces_no_instruction() {
        let unreadable = Reading::Unreadable {
            error: "Access is denied. (os error 5)".to_string(),
        };
        assert_eq!(
            route_for("cursor", &unreadable, &Reading::No),
            "unknown",
            "a client that could not be looked for must not be reported as absent"
        );
        assert_eq!(
            route_for("claude_desktop", &Reading::Yes, &unreadable),
            "unknown",
            "an unread plugin must not be routed as a missing one"
        );
        // The two controls, so this is not "always unknown".
        assert_eq!(route_for("cursor", &Reading::No, &Reading::No), "skip");
        assert_eq!(
            route_for("claude_desktop", &Reading::Yes, &Reading::No),
            "config"
        );
        // claude_code/codex_cli route to their own plugin installer whatever
        // the plugin reading says, so an unread plugin cannot mislead there.
        assert_eq!(
            route_for("claude_code", &Reading::Yes, &unreadable),
            "plugin"
        );
    }
}
