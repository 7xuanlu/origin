use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const PHASE_FILE: &str = "crates/wenlan-server/src/daemon_structure_phase.txt";
const MAIN_ROOT: &str = "crates/wenlan-server/src/main.rs";
const SCHEDULER_ROOT: &str = "crates/wenlan-server/src/scheduler.rs";
const MAIN_STARTUP_CHILD: &str = "crates/wenlan-server/src/main/startup.rs";
const MAIN_RUNTIME_CHILD: &str = "crates/wenlan-server/src/main/runtime.rs";
const SCHEDULER_AMBIENT_CHILD: &str = "crates/wenlan-server/src/scheduler/ambient.rs";
const BIND_TESTS: &str = "crates/wenlan-server/src/bind_addr_tests.rs";
const SCHEDULER_TESTS: &str = "crates/wenlan-server/src/scheduler/scheduler_tests.rs";

const MAIN_BASELINE_ORDER: &[&str] = &[
    "install_termination_signals(",
    "tokio::net::TcpListener::bind(",
    "init_logging(",
    "wait_at_startup_signal_test_barrier(",
    "DaemonDataLock::acquire(",
    "RepairArtifactStore::new(",
    "select_startup_repair_fence(",
    "let mut server_state = ServerState::new()",
    "let db = if repair_recovery_pending",
    "import_legacy_status_files(",
    "run_migration_55(",
    "reset_in_progress_documents(",
    "enforce_projection_directory_invariant(",
    "PromptRegistry::load(",
    "let config = wenlan_core::config::load_config(",
    "server_state.api_llm = Some(",
    "server_state.synthesis_llm = Some(",
    "server_state.external_llm = Some(",
    "let reranker_cache_dir = wenlan_core::db::resolve_fastembed_cache_dir(",
    "let mut deep_bgebase_pending = false",
    "reranker_mode_resolved(",
    "import_legacy_default_once(",
    "import_legacy_tags(",
    "IngestBatcher::spawn(",
    "maintenance_coordinator.finish_recovery(",
    "let shared: SharedState",
    "if optional_runtime_workers_allowed(repair_recovery_pending) && deep_bgebase_pending",
    "wenlan_core::reranker::RerankerPick::BgeBase,",
    "let selected_model = config",
    "wait_for_startup_model_admission(",
    "OnDeviceProvider::new_with_model(",
    "LLM_READINESS_HOOK.set(",
    "termination_signals.wait(",
    "scheduler::spawn_scheduler(",
    "router::build_repair_router(",
    "listener.local_addr(",
    "std::io::stdout().flush(",
    "if let Ok(port_file) = std::env::var(",
    "std::fs::write(&port_file",
    "axum::serve(",
    "let server_completed = tokio::select!",
    "match tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, scheduler_task).await",
    "let drained = tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, async {",
    "let scheduler_result = scheduler_task.await",
];

const MAIN_STARTUP_STATE_ORDER: &[&str] = &[
    "install_termination_signals(",
    "tokio::net::TcpListener::bind(",
    "init_logging(",
    "wait_at_startup_signal_test_barrier(",
    "DaemonDataLock::acquire(",
    "startup::prepare_startup_state(",
    "maintenance_coordinator.finish_recovery(",
    "let shared: SharedState",
    "if optional_runtime_workers_allowed(repair_recovery_pending) && deep_bgebase_pending",
    "wenlan_core::reranker::RerankerPick::BgeBase,",
    "let selected_model = config",
    "wait_for_startup_model_admission(",
    "OnDeviceProvider::new_with_model(",
    "LLM_READINESS_HOOK.set(",
    "termination_signals.wait(",
    "scheduler::spawn_scheduler(",
    "router::build_repair_router(",
    "listener.local_addr(",
    "std::io::stdout().flush(",
    "if let Ok(port_file) = std::env::var(",
    "std::fs::write(&port_file",
    "axum::serve(",
    "let server_completed = tokio::select!",
    "match tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, scheduler_task).await",
    "let drained = tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, async {",
    "let scheduler_result = scheduler_task.await",
];

const STARTUP_CHILD_ORDER: &[&str] = &[
    "RepairArtifactStore::new(",
    "select_startup_repair_fence(",
    "let mut server_state = ServerState::new()",
    "let db = if repair_recovery_pending",
    "import_legacy_status_files(",
    "run_migration_55(",
    "reset_in_progress_documents(",
    "enforce_projection_directory_invariant(",
    "PromptRegistry::load(",
    "let config = wenlan_core::config::load_config(",
    "server_state.api_llm = Some(",
    "server_state.synthesis_llm = Some(",
    "server_state.external_llm = Some(",
    "let reranker_cache_dir = wenlan_core::db::resolve_fastembed_cache_dir(",
    "let mut deep_bgebase_pending = false",
    "reranker_mode_resolved(",
    "import_legacy_default_once(",
    "import_legacy_tags(",
    "IngestBatcher::spawn(",
];

const MAIN_RUNTIME_ORDER: &[&str] = &[
    "install_termination_signals(",
    "tokio::net::TcpListener::bind(",
    "init_logging(",
    "wait_at_startup_signal_test_barrier(",
    "DaemonDataLock::acquire(",
    "startup::prepare_startup_state(",
    "maintenance_coordinator.finish_recovery(",
    "let shared: SharedState",
    "runtime::register_optional_runtime_workers(",
    "termination_signals.wait(",
    "scheduler::spawn_scheduler(",
    "runtime::serve_and_drain(",
];

const RUNTIME_WORKER_ORDER: &[&str] = &[
    "if optional_runtime_workers_allowed(repair_recovery_pending) && deep_bgebase_pending",
    "let cache = reranker_cache_dir.clone()",
    "wenlan_core::reranker::RerankerPick::BgeBase,",
    "let selected_model = config",
    "wait_for_startup_model_admission(",
    "OnDeviceProvider::new_with_model(",
    "LLM_READINESS_HOOK.set(",
    "let db_for_reconcile = db_arc.clone()",
    "reconcile_supported_pages(startup::SUPPORT_RECONCILE_BATCH)",
];

const RUNTIME_OPTIONAL_WORKER_SCOPES: &[&str] = &[
    "if optional_runtime_workers_allowed(repair_recovery_pending) && deep_bgebase_pending {\n        let shared_for_deep = shared.clone();",
    "if optional_runtime_workers_allowed(repair_recovery_pending) {\n        let selected_model = config",
    "if optional_runtime_workers_allowed(repair_recovery_pending) {\n        let db_for_ready = db_arc.clone();",
];

const RUNTIME_SERVE_ORDER: &[&str] = &[
    "router::build_repair_router(",
    "listener.local_addr(",
    "std::io::stdout().flush(",
    "if let Ok(port_file) = std::env::var(",
    "std::fs::write(&port_file",
    "axum::serve(",
    "let server_completed = tokio::select!",
    "match tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, scheduler_task).await",
    "let drained = tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, async {",
    "let scheduler_result = scheduler_task.await",
];

const RUNTIME_REGISTER_SNAPSHOT_ORDER: &[&str] = &[
    "let reservation = {",
    "state.startup_model_load_reserved.clone()",
    "let mut load_shutdown = {",
    "state.shutdown.subscribe()",
    "let (maintenance_for_ready, shutdown_for_reconcile) = {",
    "state.maintenance_coordinator.clone()",
    "state.shutdown.clone()",
    "let mut reconcile_shutdown = shutdown_for_reconcile.subscribe()",
];

const FACADE_SNAPSHOT_ORDER: &[&str] = &[
    "let shutdown = { shared.read().await.shutdown.clone() };",
    "let signal_shutdown = shutdown.clone()",
    "let write_signal = {",
    "let s = shared.read().await;",
    "s.write_signal.clone()",
    "scheduler::spawn_scheduler(",
];

const SCHEDULER_POLL_ORDER: &[&str] = &[
    "loop {",
    "let coordinator = {",
    "try_begin_background()",
    "let snapshot = {",
    "sync_filesystem_edits(",
    "sync_directory_sources(",
    "fire_steep_phase_safe(",
    "run_derived_receipt_sweep_if_due(",
    "run_ambient_job_safe(",
];

const AMBIENT_ITEMS: &[(&str, ItemKind)] = &[
    ("AmbientJob", ItemKind::Enum),
    ("AmbientAvailability", ItemKind::Struct),
    ("AmbientSchedule", ItemKind::Struct),
    ("ambient_turn_allowed", ItemKind::Function),
    ("should_backoff_ambient_lane", ItemKind::Function),
    ("ambient_work_consumes_thermal_turn", ItemKind::Function),
    ("AmbientTurnReport", ItemKind::Struct),
    ("resolve_ambient_provider", ItemKind::Function),
    ("run_ambient_job_safe", ItemKind::Function),
    ("run_ambient_job", ItemKind::Function),
    ("run_document_enrichment_slice_tick", ItemKind::Function),
];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Phase {
    Baseline,
    TestsExternal,
    StartupState,
    Runtime,
    Ambient,
    Final,
}

impl Phase {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "baseline" => Ok(Self::Baseline),
            "tests-external" => Ok(Self::TestsExternal),
            "startup-state" => Ok(Self::StartupState),
            "runtime" => Ok(Self::Runtime),
            "ambient" => Ok(Self::Ambient),
            "final" => Ok(Self::Final),
            other => Err(format!("unknown daemon structure phase: {other:?}")),
        }
    }

    fn external_tests(self) -> bool {
        self >= Self::TestsExternal
    }

    fn startup_child(self) -> bool {
        self >= Self::StartupState
    }

    fn runtime_child(self) -> bool {
        self >= Self::Runtime
    }

    fn ambient_child(self) -> bool {
        self >= Self::Ambient
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ItemKind {
    Function,
    Struct,
    Enum,
    Module,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Visibility {
    Private,
    PubSuper,
    Restricted,
    Public,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Declaration {
    kind: ItemKind,
    name: String,
    visibility: Visibility,
    position: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ModuleShape {
    visibility: Visibility,
    inline: bool,
    cfg_test: bool,
    path: Option<String>,
    path_count: usize,
}

fn mask_rust_non_code(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = bytes.to_vec();
    let mut index = 0;
    let mut block_depth = 0usize;
    let mut string = false;
    let mut escaped = false;
    let mut character = false;
    let mut raw_end: Option<Vec<u8>> = None;

    while index < bytes.len() {
        if let Some(end) = raw_end.as_ref() {
            if bytes[index..].starts_with(end) {
                for offset in 0..end.len() {
                    out[index + offset] = b' ';
                }
                index += end.len();
                raw_end = None;
            } else {
                if bytes[index] != b'\n' {
                    out[index] = b' ';
                }
                index += 1;
            }
            continue;
        }
        if block_depth > 0 {
            out[index] = b' ';
            if bytes[index..].starts_with(b"/*") {
                block_depth += 1;
                if index + 1 < out.len() {
                    out[index + 1] = b' ';
                }
                index += 2;
            } else if bytes[index..].starts_with(b"*/") {
                block_depth -= 1;
                if index + 1 < out.len() {
                    out[index + 1] = b' ';
                }
                index += 2;
            } else {
                index += 1;
            }
            continue;
        }
        if string {
            if bytes[index] != b'\n' {
                out[index] = b' ';
            }
            if escaped {
                escaped = false;
            } else if bytes[index] == b'\\' {
                escaped = true;
            } else if bytes[index] == b'"' {
                string = false;
            }
            index += 1;
            continue;
        }
        if character {
            if bytes[index] != b'\n' {
                out[index] = b' ';
            }
            if escaped {
                escaped = false;
            } else if bytes[index] == b'\\' {
                escaped = true;
            } else if bytes[index] == b'\'' {
                character = false;
            }
            index += 1;
            continue;
        }
        if bytes[index..].starts_with(b"//") {
            while index < bytes.len() && bytes[index] != b'\n' {
                out[index] = b' ';
                index += 1;
            }
            continue;
        }
        if bytes[index..].starts_with(b"/*") {
            block_depth = 1;
            out[index] = b' ';
            if index + 1 < out.len() {
                out[index + 1] = b' ';
            }
            index += 2;
            continue;
        }
        if bytes[index] == b'r' {
            let mut cursor = index + 1;
            while cursor < bytes.len() && bytes[cursor] == b'#' {
                cursor += 1;
            }
            if cursor < bytes.len() && bytes[cursor] == b'"' {
                let hashes = cursor - index - 1;
                for offset in 0..=(cursor - index) {
                    out[index + offset] = b' ';
                }
                raw_end = Some(
                    std::iter::once(b'"')
                        .chain(std::iter::repeat_n(b'#', hashes))
                        .collect(),
                );
                index = cursor + 1;
                continue;
            }
        }
        match bytes[index] {
            b'"' => {
                string = true;
                out[index] = b' ';
            }
            b'\'' if index + 2 < bytes.len() && bytes[index + 2] == b'\'' => {
                character = true;
                out[index] = b' ';
            }
            b'\''
                if index + 3 < bytes.len()
                    && bytes[index + 1] == b'\\'
                    && bytes[index + 3] == b'\'' =>
            {
                character = true;
                out[index] = b' ';
            }
            _ => {}
        }
        index += 1;
    }

    String::from_utf8(out).expect("mask preserves UTF-8 byte widths")
}

fn strip_visibility(line: &str) -> (Visibility, &str) {
    let line = line.trim_start();
    if let Some(rest) = line.strip_prefix("pub ") {
        return (Visibility::Public, rest);
    }
    if let Some(rest) = line.strip_prefix("pub(") {
        if let Some(close) = rest.find(')') {
            let visibility = if &rest[..close] == "super" {
                Visibility::PubSuper
            } else {
                Visibility::Restricted
            };
            return (visibility, rest[close + 1..].trim_start());
        }
    }
    (Visibility::Private, line)
}

fn item_name(rest: &str, prefix: &str) -> Option<String> {
    let rest = rest.strip_prefix(prefix)?;
    let name: String = rest
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

fn top_level_declarations(source: &str) -> Vec<Declaration> {
    let code = mask_rust_non_code(source);
    let mut declarations = Vec::new();
    let mut depth = 0isize;
    let mut offset = 0usize;
    for line in code.split_inclusive('\n') {
        if depth == 0 {
            let (visibility, mut rest) = strip_visibility(line);
            if let Some(stripped) = rest.strip_prefix("async ") {
                rest = stripped;
            }
            let found = [
                (ItemKind::Function, "fn "),
                (ItemKind::Struct, "struct "),
                (ItemKind::Enum, "enum "),
                (ItemKind::Module, "mod "),
            ]
            .into_iter()
            .find_map(|(kind, prefix)| item_name(rest, prefix).map(|name| (kind, name)));
            if let Some((kind, name)) = found {
                declarations.push(Declaration {
                    kind,
                    name,
                    visibility,
                    position: offset + line.len() - line.trim_start().len(),
                });
            }
        }
        for byte in line.bytes() {
            match byte {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                _ => {}
            }
        }
        offset += line.len();
    }
    declarations
}

fn function_span(source: &str, name: &str) -> Result<(usize, usize), String> {
    let code = mask_rust_non_code(source);
    let matches: Vec<_> = top_level_declarations(source)
        .into_iter()
        .filter(|declaration| declaration.kind == ItemKind::Function && declaration.name == name)
        .collect();
    if matches.len() != 1 {
        return Err(format!(
            "expected exactly one top-level function {name}, found {}",
            matches.len()
        ));
    }
    let start = matches[0].position;
    let open = code[start..]
        .find('{')
        .map(|offset| start + offset)
        .ok_or_else(|| format!("function {name} has no body"))?;
    let mut depth = 0isize;
    for (offset, byte) in code.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok((open, open + offset + 1));
                }
            }
            _ => {}
        }
    }
    Err(format!("function {name} has an unclosed body"))
}

fn module_item_span(source: &str, name: &str) -> Result<(usize, usize), String> {
    let code = mask_rust_non_code(source);
    let matches: Vec<_> = top_level_declarations(source)
        .into_iter()
        .filter(|declaration| declaration.kind == ItemKind::Module && declaration.name == name)
        .collect();
    if matches.len() != 1 {
        return Err(format!(
            "expected exactly one top-level module {name}, found {}",
            matches.len()
        ));
    }
    let start = matches[0].position;
    let delimiter = code.as_bytes()[start..]
        .iter()
        .position(|byte| matches!(byte, b'{' | b';'))
        .map(|offset| start + offset)
        .ok_or_else(|| format!("module {name} has no item delimiter"))?;
    if code.as_bytes()[delimiter] == b';' {
        return Ok((start, delimiter + 1));
    }

    let mut depth = 0isize;
    for (offset, byte) in code.as_bytes()[delimiter..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok((start, delimiter + offset + 1));
                }
            }
            _ => {}
        }
    }
    Err(format!("module {name} has an unclosed body"))
}

fn function_body(source: &str, name: &str) -> Result<String, String> {
    Ok(mask_rust_non_code(&function_source(source, name)?))
}

fn function_source(source: &str, name: &str) -> Result<String, String> {
    let (start, end) = function_span(source, name)?;
    Ok(source[start..end].to_string())
}

fn unique_order_violations(source: &str, anchors: &[&str], owner: &str) -> Vec<String> {
    let mut violations = Vec::new();
    let mut positions = Vec::new();
    for anchor in anchors {
        let starts_with_identifier = anchor
            .chars()
            .next()
            .is_some_and(|character| character.is_ascii_alphanumeric() || character == '_');
        let found: Vec<_> = source
            .match_indices(anchor)
            .filter(|(index, _)| {
                !starts_with_identifier
                    || source[..*index]
                        .chars()
                        .next_back()
                        .is_none_or(|character| {
                            !character.is_ascii_alphanumeric() && character != '_'
                        })
            })
            .map(|(index, _)| index)
            .collect();
        match found.as_slice() {
            [position] => positions.push((*anchor, *position)),
            [] => violations.push(format!("{owner} is missing ordered stage {anchor:?}")),
            _ => violations.push(format!(
                "{owner} duplicates ordered stage {anchor:?} {} times",
                found.len()
            )),
        }
    }
    for pair in positions.windows(2) {
        if pair[0].1 >= pair[1].1 {
            violations.push(format!(
                "{owner} reorders stage {:?} after {:?}",
                pair[0].0, pair[1].0
            ));
        }
    }
    violations
}

fn adjacent_stage_violations(source: &str, first: &str, second: &str, owner: &str) -> Vec<String> {
    let first_positions: Vec<_> = source
        .match_indices(first)
        .map(|(index, _)| index)
        .collect();
    let second_positions: Vec<_> = source
        .match_indices(second)
        .map(|(index, _)| index)
        .collect();
    if first_positions.len() != 1 || second_positions.len() != 1 {
        return vec![format!(
            "{owner} adjacency requires one {first:?} and one {second:?}"
        )];
    }
    let between = &source[first_positions[0] + first.len()..second_positions[0]];
    let compact: String = between
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    if compact != ");" {
        vec![format!(
            "{owner} must keep {first:?} immediately before {second:?}; found {compact:?} between them"
        )]
    } else {
        Vec::new()
    }
}

fn exact_token_count_violations(
    source: &str,
    token: &str,
    expected: usize,
    owner: &str,
) -> Vec<String> {
    let actual = source.matches(token).count();
    if actual != expected {
        vec![format!(
            "{owner} must contain {token:?} exactly {expected} times, found {actual}"
        )]
    } else {
        Vec::new()
    }
}

fn rust_keyword_count(source: &str, keyword: &str) -> usize {
    source
        .match_indices(keyword)
        .filter(|(position, _)| {
            let before = source[..*position].chars().next_back();
            let after = source[*position + keyword.len()..].chars().next();
            let identifier =
                |character: char| character.is_ascii_alphanumeric() || character == '_';
            before.is_none_or(|character| !identifier(character))
                && after.is_none_or(|character| !identifier(character))
        })
        .count()
}

fn exact_keyword_count_violations(
    source: &str,
    keyword: &str,
    expected: usize,
    owner: &str,
) -> Vec<String> {
    let actual = rust_keyword_count(source, keyword);
    if actual != expected {
        vec![format!(
            "{owner} must contain Rust keyword {keyword:?} exactly {expected} times, found {actual}"
        )]
    } else {
        Vec::new()
    }
}

fn exact_identifier_count_violations(
    source: &str,
    identifier: &str,
    expected: usize,
    owner: &str,
) -> Vec<String> {
    let actual = rust_keyword_count(source, identifier);
    if actual != expected {
        vec![format!(
            "{owner} must contain identifier {identifier:?} exactly {expected} times, found {actual}"
        )]
    } else {
        Vec::new()
    }
}

fn direct_call_count(source: &str, path: &str) -> usize {
    source
        .match_indices(path)
        .filter(|(position, _)| {
            let before = source[..*position].chars().next_back();
            let identifier =
                |character: char| character.is_ascii_alphanumeric() || character == '_';
            let after = &source[*position + path.len()..];
            before.is_none_or(|character| !identifier(character))
                && after.trim_start().starts_with('(')
        })
        .count()
}

fn exact_direct_call_count_violations(
    source: &str,
    path: &str,
    expected: usize,
    owner: &str,
) -> Vec<String> {
    let actual = direct_call_count(source, path);
    if actual != expected {
        vec![format!(
            "{owner} must call {path} exactly {expected} times, found {actual}"
        )]
    } else {
        Vec::new()
    }
}

fn suffix_after_unique<'a>(source: &'a str, anchor: &str, owner: &str) -> Result<&'a str, String> {
    let positions: Vec<_> = source
        .match_indices(anchor)
        .map(|(index, _)| index)
        .collect();
    if positions.len() != 1 {
        return Err(format!(
            "{owner} suffix requires one {anchor:?}, found {}",
            positions.len()
        ));
    }
    Ok(&source[positions[0] + anchor.len()..])
}

fn runtime_carried_value_violations(source: &str, owner: &str) -> Vec<String> {
    let normalized_source = source.replace("\r\n", "\n");
    let code = mask_rust_non_code(&normalized_source);
    let mut violations = unique_order_violations(&code, RUNTIME_WORKER_ORDER, owner);
    for scope in RUNTIME_OPTIONAL_WORKER_SCOPES {
        violations.extend(exact_token_count_violations(&code, scope, 1, owner));
    }
    violations.extend(runtime_recompute_violations(&code, owner));
    violations
}

fn runtime_recompute_violations(source: &str, owner: &str) -> Vec<String> {
    let code = mask_rust_non_code(source);
    let mut violations = Vec::new();
    for path in [
        "load_config",
        "resolve_fastembed_cache_dir",
        "reranker_mode_resolved",
        "resolve_reranker_plan",
    ] {
        violations.extend(exact_identifier_count_violations(&code, path, 0, owner));
    }
    violations
}

fn macro_call_positions(source: &str, macro_name: &str) -> Vec<usize> {
    let needle = format!("{macro_name}!");
    source
        .match_indices(&needle)
        .filter_map(|(position, _)| {
            let before = source[..position].chars().next_back();
            let identifier =
                |character: char| character.is_ascii_alphanumeric() || character == '_';
            if before.is_some_and(identifier) {
                return None;
            }
            source[position + needle.len()..]
                .trim_start()
                .starts_with('(')
                .then_some(position)
        })
        .collect()
}

fn macro_between_violations(
    source: &str,
    before: &str,
    macro_name: &str,
    expected_call: &str,
    after: &str,
    owner: &str,
) -> Vec<String> {
    let code = mask_rust_non_code(source);
    let before_positions: Vec<_> = code.match_indices(before).map(|(index, _)| index).collect();
    let after_positions: Vec<_> = code.match_indices(after).map(|(index, _)| index).collect();
    let macro_positions = macro_call_positions(&code, macro_name);
    if before_positions.len() != 1 || after_positions.len() != 1 || macro_positions.len() != 1 {
        return vec![format!(
            "{owner} must keep one {macro_name}! between {before:?} and {after:?}"
        )];
    }
    let macro_position = macro_positions[0];
    let mut violations = Vec::new();
    if !(before_positions[0] < macro_position && macro_position < after_positions[0]) {
        violations.push(format!(
            "{owner} reorders {macro_name}! outside the port announcement boundary"
        ));
    }
    if !source[macro_position..].starts_with(expected_call) {
        violations.push(format!(
            "{owner} {macro_name}! must remain exactly {expected_call:?}"
        ));
    }
    violations
}

fn child_files(root: &Path, directory: &str) -> BTreeSet<String> {
    let directory = root.join(directory);
    if !directory.is_dir() {
        return BTreeSet::new();
    }
    std::fs::read_dir(directory)
        .expect("read daemon child directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|extension| extension.to_str()) == Some("rs"))
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            Some(name.to_string())
        })
        .collect()
}

fn exact_child_files_violations(
    actual: &BTreeSet<String>,
    expected: &BTreeSet<String>,
    owner: &str,
) -> Vec<String> {
    if actual != expected {
        vec![format!(
            "{owner} child files are {actual:?}, expected {expected:?}"
        )]
    } else {
        Vec::new()
    }
}

fn module_shapes(source: &str, module: &str) -> Vec<ModuleShape> {
    let code = mask_rust_non_code(source);
    top_level_declarations(source)
        .into_iter()
        .filter(|declaration| declaration.kind == ItemKind::Module && declaration.name == module)
        .map(|declaration| {
            let line_end = code[declaration.position..]
                .find('\n')
                .map_or(code.len(), |offset| declaration.position + offset);
            let declaration_line = &code[declaration.position..line_end];
            let module_prefix = format!("mod {module}");
            let after_module = declaration_line
                .find(&module_prefix)
                .map(|offset| &declaration_line[offset + module_prefix.len()..])
                .unwrap_or_default()
                .trim_start();

            let mut attributes = Vec::new();
            let line_start = source[..declaration.position]
                .rfind('\n')
                .map_or(0, |index| index + 1);
            for line in source[..line_start].lines().rev() {
                let line = line.trim();
                if line.starts_with("#[") {
                    attributes.push(line.to_string());
                } else {
                    break;
                }
            }
            attributes.reverse();
            let paths: Vec<_> = attributes
                .iter()
                .filter_map(|attribute| {
                    attribute
                        .strip_prefix("#[path = \"")
                        .and_then(|rest| rest.strip_suffix("\"]"))
                        .map(str::to_string)
                })
                .collect();
            ModuleShape {
                visibility: declaration.visibility,
                inline: after_module.starts_with('{'),
                cfg_test: attributes
                    .iter()
                    .filter(|line| *line == "#[cfg(test)]")
                    .count()
                    == 1,
                path: paths.first().cloned(),
                path_count: paths.len(),
            }
        })
        .collect()
}

fn test_module_contract_violations(
    source: &str,
    module: &str,
    expected_inline: bool,
    expected_path: Option<&str>,
    owner: &str,
) -> Vec<String> {
    let shapes = module_shapes(source, module);
    if shapes.len() != 1 {
        return vec![format!(
            "{owner} must declare test module {module} exactly once, found {}",
            shapes.len()
        )];
    }
    let shape = &shapes[0];
    let mut violations = Vec::new();
    if shape.visibility != Visibility::Private {
        violations.push(format!("{owner} test module {module} must stay private"));
    }
    if !shape.cfg_test {
        violations.push(format!(
            "{owner} test module {module} must have exactly one adjacent #[cfg(test)]"
        ));
    }
    if shape.inline != expected_inline {
        violations.push(format!(
            "{owner} test module {module} inline={} expected {expected_inline}",
            shape.inline
        ));
    }
    if shape.path.as_deref() != expected_path {
        violations.push(format!(
            "{owner} test module {module} path={:?} expected {expected_path:?}",
            shape.path
        ));
    }
    if shape.path_count != usize::from(expected_path.is_some()) {
        violations.push(format!(
            "{owner} test module {module} has {} adjacent path attributes, expected {}",
            shape.path_count,
            usize::from(expected_path.is_some())
        ));
    }
    violations
}

fn normalize_source_newlines(source: String) -> String {
    source.replace("\r\n", "\n")
}

fn read_source(root: &Path, path: &str) -> String {
    let path = root.join(path);
    normalize_source_newlines(
        std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display())),
    )
}

fn read_if_present(root: &Path, path: &str) -> Option<String> {
    root.join(path).is_file().then(|| read_source(root, path))
}

fn declarations_named<'a>(
    sources: &'a [(&'a str, &'a str)],
    kind: ItemKind,
    name: &str,
) -> Vec<(&'a str, Declaration)> {
    sources
        .iter()
        .flat_map(|(path, source)| {
            top_level_declarations(source)
                .into_iter()
                .filter(move |declaration| declaration.kind == kind && declaration.name == name)
                .map(move |declaration| (*path, declaration))
        })
        .collect()
}

fn exact_anchor_owner_violations(
    sources: &[(&str, &str)],
    anchor: &str,
    owner: &str,
) -> Vec<String> {
    let matches: Vec<_> = sources
        .iter()
        .flat_map(|(path, source)| {
            mask_rust_non_code(source)
                .match_indices(anchor)
                .map(move |(position, _)| (*path, position))
                .collect::<Vec<_>>()
        })
        .collect();
    if matches.len() != 1 {
        return vec![format!(
            "{anchor:?} must occur once in {owner}; found {} at {:?}",
            matches.len(),
            matches.iter().map(|(path, _)| *path).collect::<Vec<_>>()
        )];
    }
    if matches[0].0 != owner {
        vec![format!(
            "{anchor:?} is owned by {}, expected {owner}",
            matches[0].0
        )]
    } else {
        Vec::new()
    }
}

fn exact_private_owner_violations(
    sources: &[(&str, &str)],
    kind: ItemKind,
    name: &str,
    owner: &str,
) -> Vec<String> {
    let matches = declarations_named(sources, kind, name);
    if matches.len() != 1 {
        return vec![format!(
            "{name} must have one production definition in {owner}; found {} at {:?}",
            matches.len(),
            matches.iter().map(|(path, _)| *path).collect::<Vec<_>>()
        )];
    }
    let (actual_owner, declaration) = &matches[0];
    let mut violations = Vec::new();
    if *actual_owner != owner {
        violations.push(format!(
            "{name} is owned by {actual_owner}, expected {owner}"
        ));
    }
    if matches!(
        declaration.visibility,
        Visibility::Restricted | Visibility::Public
    ) {
        violations.push(format!("{owner}::{name} must stay private or pub(super)"));
    }
    violations
}

fn absent_item_violations(sources: &[(&str, &str)], kind: ItemKind, name: &str) -> Vec<String> {
    let matches = declarations_named(sources, kind, name);
    if !matches.is_empty() {
        vec![format!(
            "future R7 item {name} exists before its phase at {:?}",
            matches.iter().map(|(path, _)| *path).collect::<Vec<_>>()
        )]
    } else {
        Vec::new()
    }
}

fn module_boundary_violations(
    source: &str,
    module: &str,
    expected: bool,
    owner: &str,
) -> Vec<String> {
    let matches: Vec<_> = top_level_declarations(source)
        .into_iter()
        .filter(|declaration| declaration.kind == ItemKind::Module && declaration.name == module)
        .collect();
    if !expected {
        return if !matches.is_empty() {
            vec![format!("{owner} declares {module} before its phase")]
        } else {
            Vec::new()
        };
    }
    if matches.len() != 1 {
        return vec![format!(
            "{owner} must declare private module {module} exactly once, found {}",
            matches.len()
        )];
    }
    if matches[0].visibility != Visibility::Private {
        vec![format!("{owner} module {module} must stay private")]
    } else {
        Vec::new()
    }
}

fn scheduler_task_locality_violations(scheduler: &str, ambient: Option<&str>) -> Vec<String> {
    let mut violations = Vec::new();
    let mut remainder = mask_rust_non_code(scheduler).into_bytes();
    match module_item_span(scheduler, "tests") {
        Ok((start, end)) => {
            for byte in &mut remainder[start..end] {
                if *byte != b'\n' {
                    *byte = b' ';
                }
            }
        }
        Err(error) => violations.push(error),
    }

    for (function, expected_spawn, expected_loop) in [
        ("spawn_scheduler", 1, 1),
        ("wait_for_startup_model_admission", 0, 1),
    ] {
        let (start, end) = match function_span(scheduler, function) {
            Ok(span) => span,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        let body = mask_rust_non_code(&scheduler[start..end]);
        let owner = format!("{SCHEDULER_ROOT}::{function}");
        violations.extend(exact_direct_call_count_violations(
            &body,
            "tokio::spawn",
            expected_spawn,
            &owner,
        ));
        violations.extend(exact_direct_call_count_violations(
            &body,
            "tokio::task::spawn",
            0,
            &owner,
        ));
        violations.extend(exact_keyword_count_violations(
            &body,
            "loop",
            expected_loop,
            &owner,
        ));
        violations.extend(exact_keyword_count_violations(&body, "while", 0, &owner));
        for byte in &mut remainder[start..end] {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    }
    let remainder = String::from_utf8(remainder).expect("mask preserves UTF-8 byte widths");
    for path in ["tokio::spawn", "tokio::task::spawn"] {
        violations.extend(exact_direct_call_count_violations(
            &remainder,
            path,
            0,
            "scheduler production remainder",
        ));
    }
    for keyword in ["loop", "while"] {
        violations.extend(exact_keyword_count_violations(
            &remainder,
            keyword,
            0,
            "scheduler production remainder",
        ));
    }

    if let Some(ambient) = ambient {
        let code = mask_rust_non_code(ambient);
        for path in ["tokio::spawn", "tokio::task::spawn"] {
            violations.extend(exact_direct_call_count_violations(
                &code,
                path,
                0,
                SCHEDULER_AMBIENT_CHILD,
            ));
        }
        for keyword in ["loop", "while"] {
            violations.extend(exact_keyword_count_violations(
                &code,
                keyword,
                0,
                SCHEDULER_AMBIENT_CHILD,
            ));
        }
    }
    violations
}

fn test_boundary_violations(root: &Path, main: &str, scheduler: &str, phase: Phase) -> Vec<String> {
    let mut violations = Vec::new();
    if phase.external_tests() {
        violations.extend(test_module_contract_violations(
            main,
            "bind_addr_tests",
            false,
            Some("bind_addr_tests.rs"),
            MAIN_ROOT,
        ));
        violations.extend(test_module_contract_violations(
            scheduler,
            "tests",
            false,
            Some("scheduler/scheduler_tests.rs"),
            SCHEDULER_ROOT,
        ));
        if !root.join(BIND_TESTS).is_file() {
            violations.push("bind_addr_tests.rs is missing".into());
        }
        if !root.join(SCHEDULER_TESTS).is_file() {
            violations.push("scheduler/scheduler_tests.rs is missing".into());
        }
    } else {
        violations.extend(test_module_contract_violations(
            main,
            "bind_addr_tests",
            true,
            None,
            MAIN_ROOT,
        ));
        violations.extend(test_module_contract_violations(
            scheduler,
            "tests",
            true,
            None,
            SCHEDULER_ROOT,
        ));
        if root.join(BIND_TESTS).is_file() || root.join(SCHEDULER_TESTS).is_file() {
            violations.push("baseline unexpectedly contains external daemon tests".into());
        }
    }
    violations
}

fn structure_violations(root: &Path, phase: Phase, main: &str, scheduler: &str) -> Vec<String> {
    let startup = read_if_present(root, MAIN_STARTUP_CHILD);
    let runtime = read_if_present(root, MAIN_RUNTIME_CHILD);
    let ambient = read_if_present(root, SCHEDULER_AMBIENT_CHILD);
    structure_violations_with_children(
        root,
        phase,
        main,
        scheduler,
        startup.as_deref(),
        runtime.as_deref(),
        ambient.as_deref(),
    )
}

fn structure_violations_with_children(
    root: &Path,
    phase: Phase,
    main: &str,
    scheduler: &str,
    startup: Option<&str>,
    runtime: Option<&str>,
    ambient: Option<&str>,
) -> Vec<String> {
    let mut violations = test_boundary_violations(root, main, scheduler, phase);

    let expected_main_files: BTreeSet<String> = [
        phase.startup_child().then_some("startup.rs"),
        phase.runtime_child().then_some("runtime.rs"),
    ]
    .into_iter()
    .flatten()
    .map(str::to_string)
    .collect();
    let actual_main_files = child_files(root, "crates/wenlan-server/src/main");
    violations.extend(exact_child_files_violations(
        &actual_main_files,
        &expected_main_files,
        "main",
    ));
    let expected_scheduler_files: BTreeSet<String> = [
        phase.external_tests().then_some("scheduler_tests.rs"),
        phase.ambient_child().then_some("ambient.rs"),
    ]
    .into_iter()
    .flatten()
    .map(str::to_string)
    .collect();
    let actual_scheduler_files = child_files(root, "crates/wenlan-server/src/scheduler");
    violations.extend(exact_child_files_violations(
        &actual_scheduler_files,
        &expected_scheduler_files,
        "scheduler",
    ));

    violations.extend(module_boundary_violations(
        main,
        "startup",
        phase.startup_child(),
        MAIN_ROOT,
    ));
    violations.extend(module_boundary_violations(
        main,
        "runtime",
        phase.runtime_child(),
        MAIN_ROOT,
    ));
    violations.extend(module_boundary_violations(
        scheduler,
        "ambient",
        phase.ambient_child(),
        SCHEDULER_ROOT,
    ));

    if phase.startup_child() != startup.is_some() {
        violations.push(format!(
            "{MAIN_STARTUP_CHILD} presence does not match phase {phase:?}"
        ));
    }
    if phase.runtime_child() != runtime.is_some() {
        violations.push(format!(
            "{MAIN_RUNTIME_CHILD} presence does not match phase {phase:?}"
        ));
    }
    if phase.ambient_child() != ambient.is_some() {
        violations.push(format!(
            "{SCHEDULER_AMBIENT_CHILD} presence does not match phase {phase:?}"
        ));
    }

    let mut owned_sources = vec![(MAIN_ROOT, main), (SCHEDULER_ROOT, scheduler)];
    if let Some(source) = startup {
        owned_sources.push((MAIN_STARTUP_CHILD, source));
    }
    if let Some(source) = runtime {
        owned_sources.push((MAIN_RUNTIME_CHILD, source));
    }
    if let Some(source) = ambient {
        owned_sources.push((SCHEDULER_AMBIENT_CHILD, source));
    }
    let mut main_sources = vec![(MAIN_ROOT, main)];
    if let Some(source) = startup {
        main_sources.push((MAIN_STARTUP_CHILD, source));
    }
    if let Some(source) = runtime {
        main_sources.push((MAIN_RUNTIME_CHILD, source));
    }

    let startup_value_owner = if phase.startup_child() {
        MAIN_STARTUP_CHILD
    } else {
        MAIN_ROOT
    };
    for anchor in [
        "let config = wenlan_core::config::load_config(",
        "let reranker_cache_dir = wenlan_core::db::resolve_fastembed_cache_dir(",
        "let mut deep_bgebase_pending = false",
    ] {
        violations.extend(exact_anchor_owner_violations(
            &main_sources,
            anchor,
            startup_value_owner,
        ));
    }

    if phase.startup_child() {
        violations.extend(exact_private_owner_violations(
            &owned_sources,
            ItemKind::Function,
            "prepare_startup_state",
            MAIN_STARTUP_CHILD,
        ));
    } else {
        violations.extend(absent_item_violations(
            &owned_sources,
            ItemKind::Function,
            "prepare_startup_state",
        ));
    }
    for name in ["register_optional_runtime_workers", "serve_and_drain"] {
        if phase.runtime_child() {
            violations.extend(exact_private_owner_violations(
                &owned_sources,
                ItemKind::Function,
                name,
                MAIN_RUNTIME_CHILD,
            ));
        } else {
            violations.extend(absent_item_violations(
                &owned_sources,
                ItemKind::Function,
                name,
            ));
        }
    }

    let ambient_owner = if phase.ambient_child() {
        SCHEDULER_AMBIENT_CHILD
    } else {
        SCHEDULER_ROOT
    };
    for (name, kind) in AMBIENT_ITEMS {
        violations.extend(exact_private_owner_violations(
            &owned_sources,
            *kind,
            name,
            ambient_owner,
        ));
    }
    for (name, kind) in [
        ("AmbientBudgetProvider", ItemKind::Struct),
        ("with_shared_automatic_budget", ItemKind::Function),
    ] {
        violations.extend(exact_private_owner_violations(
            &owned_sources,
            kind,
            name,
            SCHEDULER_ROOT,
        ));
    }

    let spawn = match function_body(scheduler, "spawn_scheduler") {
        Ok(body) => body,
        Err(error) => {
            violations.push(error);
            String::new()
        }
    };
    let spawn_declarations =
        declarations_named(&owned_sources, ItemKind::Function, "spawn_scheduler");
    if spawn_declarations.len() != 1
        || spawn_declarations[0].0 != SCHEDULER_ROOT
        || spawn_declarations[0].1.visibility != Visibility::Public
    {
        violations.push(
            "spawn_scheduler must remain the single public scheduler-root entry point".into(),
        );
    }
    violations.extend(unique_order_violations(
        &spawn,
        SCHEDULER_POLL_ORDER,
        "spawn_scheduler",
    ));
    if spawn.matches("tokio::spawn").count() != 1 {
        violations.push(format!(
            "spawn_scheduler must own one Tokio scheduler task, found {}",
            spawn.matches("tokio::spawn").count()
        ));
    }
    if spawn.matches("loop {").count() != 1 {
        violations.push(format!(
            "spawn_scheduler must own one serial poll loop, found {}",
            spawn.matches("loop {").count()
        ));
    }
    violations.extend(scheduler_task_locality_violations(scheduler, ambient));

    let run_daemon = match function_body(main, "run_daemon") {
        Ok(body) => body,
        Err(error) => {
            violations.push(error);
            String::new()
        }
    };
    let run_daemon_source = function_source(main, "run_daemon").unwrap_or_default();
    let main_order = match phase {
        Phase::Baseline | Phase::TestsExternal => MAIN_BASELINE_ORDER,
        Phase::StartupState => MAIN_STARTUP_STATE_ORDER,
        Phase::Runtime | Phase::Ambient | Phase::Final => MAIN_RUNTIME_ORDER,
    };
    violations.extend(unique_order_violations(
        &run_daemon,
        main_order,
        "run_daemon",
    ));
    if run_daemon
        .matches("maintenance_coordinator.finish_recovery(")
        .count()
        != 1
    {
        violations.push("finish_recovery must remain exactly once in the run_daemon facade".into());
    }
    violations.extend(adjacent_stage_violations(
        &run_daemon,
        "maintenance_coordinator.finish_recovery(",
        "let shared: SharedState",
        "run_daemon",
    ));
    let expected_main_shared_reads = if phase.runtime_child() { 2 } else { 5 };
    violations.extend(exact_token_count_violations(
        &run_daemon,
        "shared.read().await",
        expected_main_shared_reads,
        "run_daemon",
    ));
    violations.extend(unique_order_violations(
        &run_daemon,
        FACADE_SNAPSHOT_ORDER,
        "run_daemon facade snapshots",
    ));
    if !phase.runtime_child() {
        violations.extend(unique_order_violations(
            &run_daemon,
            RUNTIME_REGISTER_SNAPSHOT_ORDER,
            "run_daemon runtime-registration snapshots",
        ));
    }

    if phase.runtime_child() {
        if let Some(source) = runtime {
            violations.extend(runtime_carried_value_violations(source, MAIN_RUNTIME_CHILD));
        }
        violations.extend(runtime_recompute_violations(
            &run_daemon,
            "run_daemon runtime facade",
        ));
    } else {
        match suffix_after_unique(&run_daemon, "let shared: SharedState", "run_daemon") {
            Ok(runtime_suffix) => violations.extend(runtime_carried_value_violations(
                runtime_suffix,
                "run_daemon runtime suffix",
            )),
            Err(error) => violations.push(error),
        }
    }

    if let Some(source) = startup {
        let body = match function_body(source, "prepare_startup_state") {
            Ok(body) => body,
            Err(error) => {
                violations.push(error);
                String::new()
            }
        };
        violations.extend(unique_order_violations(
            &body,
            STARTUP_CHILD_ORDER,
            "prepare_startup_state",
        ));
        if source.contains("finish_recovery(") {
            violations.push(
                "startup child must return sealed state before facade finish_recovery".into(),
            );
        }
    }
    if let Some(source) = runtime {
        for (function, order) in [
            ("register_optional_runtime_workers", RUNTIME_WORKER_ORDER),
            ("serve_and_drain", RUNTIME_SERVE_ORDER),
        ] {
            let body = match function_body(source, function) {
                Ok(body) => body,
                Err(error) => {
                    violations.push(error);
                    String::new()
                }
            };
            violations.extend(unique_order_violations(&body, order, function));
            if function == "register_optional_runtime_workers" {
                violations.extend(exact_token_count_violations(
                    &body,
                    "shared.read().await",
                    3,
                    "register_optional_runtime_workers",
                ));
                violations.extend(unique_order_violations(
                    &body,
                    RUNTIME_REGISTER_SNAPSHOT_ORDER,
                    "register_optional_runtime_workers snapshots",
                ));
            } else {
                let raw_body = function_source(source, function).unwrap_or_default();
                violations.extend(macro_between_violations(
                    &raw_body,
                    "listener.local_addr(",
                    "println",
                    "println!(\"WENLAN_LISTENING_ON={}\", local_addr);",
                    "std::io::stdout().flush(",
                    "serve_and_drain",
                ));
            }
        }
    } else {
        violations.extend(macro_between_violations(
            &run_daemon_source,
            "listener.local_addr(",
            "println",
            "println!(\"WENLAN_LISTENING_ON={}\", local_addr);",
            "std::io::stdout().flush(",
            "run_daemon",
        ));
    }

    if phase == Phase::Final {
        for forbidden in [
            "RepairArtifactStore::new(",
            "let db = if repair_recovery_pending",
            "wait_for_startup_model_admission(",
            "router::build_repair_router(",
            "axum::serve(",
        ] {
            if run_daemon.contains(forbidden) {
                violations.push(format!(
                    "final run_daemon facade still owns extracted implementation {forbidden:?}"
                ));
            }
        }
    }

    if phase.external_tests() {
        for path in [BIND_TESTS, SCHEDULER_TESTS] {
            if let Some(source) = read_if_present(root, path) {
                for (name, kind) in [
                    ("prepare_startup_state", ItemKind::Function),
                    ("register_optional_runtime_workers", ItemKind::Function),
                    ("serve_and_drain", ItemKind::Function),
                    ("run_ambient_job_safe", ItemKind::Function),
                    ("run_ambient_job", ItemKind::Function),
                ] {
                    if !declarations_named(&[(path, &source)], kind, name).is_empty() {
                        violations.push(format!(
                            "test-only source {path} owns production item {name}"
                        ));
                    }
                }
            }
        }
    }

    violations.sort();
    violations
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn daemon_structure_matches_expected_phase() {
    let root = repo_root();
    let phase_source = read_source(&root, PHASE_FILE);
    let phase = Phase::parse(&phase_source).expect("closed R7 phase vocabulary");
    assert_eq!(
        phase,
        Phase::Final,
        "R7 daemon decomposition is finalized; expected \"final\", got {phase:?}"
    );
    let main = read_source(&root, MAIN_ROOT);
    let scheduler = read_source(&root, SCHEDULER_ROOT);
    let violations = structure_violations(&root, phase, &main, &scheduler);
    assert!(
        violations.is_empty(),
        "R7 daemon structure does not match tracked phase {phase:?}:\n{}",
        violations.join("\n")
    );
}

#[test]
fn daemon_structure_phase_parser_fails_closed_on_unknown() {
    assert_eq!(Phase::parse("baseline").unwrap(), Phase::Baseline);
    assert_eq!(Phase::parse("final").unwrap(), Phase::Final);
    assert!(
        Phase::parse("almost-final").is_err(),
        "unknown phases must not weaken the structure tooth"
    );
}

#[test]
fn daemon_structure_order_guard_rejects_delete_duplicate_and_reorder() {
    let valid = "alpha(); beta(); gamma();";
    assert!(
        unique_order_violations(valid, &["alpha()", "beta()", "gamma()"], "synthetic").is_empty()
    );

    let deleted = "alpha(); gamma();";
    assert!(
        unique_order_violations(deleted, &["alpha()", "beta()", "gamma()"], "synthetic")
            .iter()
            .any(|violation| violation.contains("missing ordered stage"))
    );

    let duplicated = "alpha(); beta(); beta(); gamma();";
    assert!(
        unique_order_violations(duplicated, &["alpha()", "beta()", "gamma()"], "synthetic")
            .iter()
            .any(|violation| violation.contains("duplicates ordered stage"))
    );

    let reordered = "beta(); alpha(); gamma();";
    assert!(
        unique_order_violations(reordered, &["alpha()", "beta()", "gamma()"], "synthetic")
            .iter()
            .any(|violation| violation.contains("reorders stage"))
    );
}

#[test]
fn daemon_structure_visibility_guard_rejects_public_or_duplicate_items() {
    let private = "mod startup;\nasync fn prepare_startup_state() {}\n";
    let sources = [("main.rs", private)];
    assert!(exact_private_owner_violations(
        &sources,
        ItemKind::Function,
        "prepare_startup_state",
        "main.rs"
    )
    .is_empty());
    assert!(module_boundary_violations(private, "startup", true, "main.rs").is_empty());

    let child_visible = "pub(super) async fn prepare_startup_state() {}\n";
    assert!(exact_private_owner_violations(
        &[("startup.rs", child_visible)],
        ItemKind::Function,
        "prepare_startup_state",
        "startup.rs"
    )
    .is_empty());

    let public = "pub mod startup;\npub(crate) async fn prepare_startup_state() {}\n";
    let sources = [("main.rs", public)];
    assert!(exact_private_owner_violations(
        &sources,
        ItemKind::Function,
        "prepare_startup_state",
        "main.rs"
    )
    .iter()
    .any(|violation| violation.contains("must stay private")));
    assert!(
        module_boundary_violations(public, "startup", true, "main.rs")
            .iter()
            .any(|violation| violation.contains("must stay private"))
    );

    let duplicate = [
        ("main.rs", "async fn prepare_startup_state() {}\n"),
        ("startup.rs", "async fn prepare_startup_state() {}\n"),
    ];
    assert!(exact_private_owner_violations(
        &duplicate,
        ItemKind::Function,
        "prepare_startup_state",
        "startup.rs"
    )
    .iter()
    .any(|violation| violation.contains("found 2")));
}

#[test]
fn reviewer_mutation_removed_repair_fence_is_rejected() {
    let root = repo_root();
    let main = read_source(&root, MAIN_ROOT);
    let startup = read_source(&root, MAIN_STARTUP_CHILD);
    let mutated_startup = startup.replacen(
        "select_startup_repair_fence(&pending_repairs",
        "removed_startup_repair_fence(&pending_repairs",
        1,
    );
    assert_ne!(
        mutated_startup, startup,
        "startup repair-fence mutation needle must exist"
    );
    let runtime = read_source(&root, MAIN_RUNTIME_CHILD);
    let scheduler = read_source(&root, SCHEDULER_ROOT);
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &main,
        &scheduler,
        Some(&mutated_startup),
        Some(&runtime),
        None,
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("select_startup_repair_fence")),
        "the live order tooth must reject removal of the durable repair fence: {violations:?}"
    );
}

#[test]
fn reviewer_mutations_reject_startup_runtime_and_drain_false_greens() {
    let root = repo_root();
    let main = read_source(&root, MAIN_ROOT);
    let startup = read_source(&root, MAIN_STARTUP_CHILD);
    let runtime = read_source(&root, MAIN_RUNTIME_CHILD);
    let scheduler = read_source(&root, SCHEDULER_ROOT);

    let crlf_runtime = runtime.replace('\n', "\r\n");
    assert_eq!(
        normalize_source_newlines(crlf_runtime.clone()),
        runtime,
        "structure fixtures must canonicalize Windows CRLF before mutation"
    );
    let violations = runtime_carried_value_violations(&crlf_runtime, MAIN_RUNTIME_CHILD);
    assert!(
        violations.is_empty(),
        "runtime carried-value guard must accept Windows CRLF source: {violations:?}"
    );

    let mutated_startup = startup.replacen(
        "enforce_projection_directory_invariant(",
        "removed_projection_invariant(",
        1,
    );
    assert_ne!(
        mutated_startup, startup,
        "projection invariant mutation needle must exist"
    );
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &main,
        &scheduler,
        Some(&mutated_startup),
        Some(&runtime),
        None,
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("enforce_projection_directory_invariant")),
        "live startup projection mutation must be rejected: {violations:?}"
    );

    let removed_optional_guard = runtime.replacen(
        "if optional_runtime_workers_allowed(repair_recovery_pending) && deep_bgebase_pending {",
        "if deep_bgebase_pending {",
        1,
    );
    assert_ne!(
        removed_optional_guard, runtime,
        "deep-reranker optional-worker guard must exist"
    );
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &main,
        &scheduler,
        Some(&startup),
        Some(&removed_optional_guard),
        None,
    );
    assert!(
        violations.iter().any(|violation| {
            violation.contains("optional_runtime_workers_allowed")
                && violation.contains("register_optional_runtime_workers")
        }),
        "optional-worker guard removal must be rejected by the live runtime tooth: {violations:?}"
    );

    let scope_anchor =
        "if optional_runtime_workers_allowed(repair_recovery_pending) {\n        let selected_model = config";
    let drifted_scope = runtime.replacen(
        scope_anchor,
        "let selected_model = config;\n    if optional_runtime_workers_allowed(repair_recovery_pending) {",
        1,
    );
    assert_ne!(
        drifted_scope, runtime,
        "on-device optional-worker scope must exist"
    );
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &main,
        &scheduler,
        Some(&startup),
        Some(&drifted_scope),
        None,
    );
    assert!(
        violations.iter().any(|violation| {
            violation.contains("let selected_model = config")
                && violation.contains("exactly 1 times, found 0")
        }),
        "optional-worker scope drift must be rejected by the live runtime tooth: {violations:?}"
    );

    for (needle, replacement, expected) in [
        (
            "OnDeviceProvider::new_with_model(",
            "removed_on_device_provider(",
            "OnDeviceProvider::new_with_model",
        ),
        (
            "std::fs::write(&port_file",
            "removed_port_file_write(&port_file",
            "std::fs::write(&port_file",
        ),
        (
            "match tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, scheduler_task).await",
            "match removed_first_drain(SHUTDOWN_DRAIN_TIMEOUT, scheduler_task).await",
            "match tokio::time::timeout",
        ),
        (
            "let drained = tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, async {",
            "let removed_drain = tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, async {",
            "let drained = tokio::time::timeout",
        ),
    ] {
        let mutated_runtime = runtime.replacen(needle, replacement, 1);
        assert_ne!(
            mutated_runtime, runtime,
            "runtime mutation needle must exist: {needle}"
        );
        let violations = structure_violations_with_children(
            &root,
            Phase::Runtime,
            &main,
            &scheduler,
            Some(&startup),
            Some(&mutated_runtime),
            None,
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "live mutation {needle:?} must be rejected: {violations:?}"
        );
    }

    let scheduler_serve_reordered = main
        .replacen("runtime::serve_and_drain(", "removed_serve_and_drain(", 1)
        .replacen(
            "scheduler::spawn_scheduler(",
            "runtime::serve_and_drain(); scheduler::spawn_scheduler(",
            1,
        );
    assert_ne!(
        scheduler_serve_reordered, main,
        "scheduler/serve order anchors must exist"
    );
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &scheduler_serve_reordered,
        &scheduler,
        Some(&startup),
        Some(&runtime),
        None,
    );
    assert!(
        violations.iter().any(|violation| {
            violation.contains("run_daemon")
                && violation.contains("runtime::serve_and_drain(")
                && violation.contains("reorders stage")
        }),
        "scheduler/serve order drift must be rejected with the runtime child present: {violations:?}"
    );

    let router_serve_reordered = runtime
        .replacen(
            "router::build_repair_router(shared)",
            "removed_repair_router(shared)",
            1,
        )
        .replacen(
            "let server = axum::serve(",
            "let _late_router = router::build_repair_router(shared.clone());\n    let server = axum::serve(",
            1,
        );
    assert_ne!(
        router_serve_reordered, runtime,
        "router/serve order anchors must exist"
    );
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &main,
        &scheduler,
        Some(&startup),
        Some(&router_serve_reordered),
        None,
    );
    assert!(
        violations.iter().any(|violation| {
            violation.contains("serve_and_drain")
                && violation.contains("router::build_repair_router(")
                && violation.contains("reorders stage")
        }),
        "router/serve order drift must be rejected with the runtime child present: {violations:?}"
    );

    let finish = "server_state.maintenance_coordinator.finish_recovery();";
    let mutated = main.replacen(
        finish,
        &format!("{finish}\nlet hidden_between_recovery_and_share = ();"),
        1,
    );
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &mutated,
        &scheduler,
        Some(&startup),
        Some(&runtime),
        None,
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("immediately before")),
        "finish_recovery adjacency mutation must be rejected: {violations:?}"
    );
}

#[test]
fn reviewer_mutations_reject_scheduler_fence_snapshot_and_hidden_tasks() {
    let root = repo_root();
    let main = read_source(&root, MAIN_ROOT);
    let startup = read_source(&root, MAIN_STARTUP_CHILD);
    let runtime = read_source(&root, MAIN_RUNTIME_CHILD);
    let scheduler = read_source(&root, SCHEDULER_ROOT);
    let without_snapshot = scheduler.replacen("let snapshot = {", "let omitted_snapshot = {", 1);
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &main,
        &without_snapshot,
        Some(&startup),
        Some(&runtime),
        None,
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("let snapshot = {")),
        "scheduler snapshot mutation must be rejected: {violations:?}"
    );
    let without_fence =
        scheduler.replacen("try_begin_background()", "removed_maintenance_fence()", 1);
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &main,
        &without_fence,
        Some(&startup),
        Some(&runtime),
        None,
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("try_begin_background")),
        "scheduler maintenance-fence mutation must be rejected: {violations:?}"
    );

    let synthetic_scheduler = r#"
pub fn spawn_scheduler() {
    tokio::spawn(async move {});
    loop { break; }
}
async fn wait_for_startup_model_admission() {
    loop { break; }
}
#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
struct HiddenScheduler;
impl HiddenScheduler {
    fn hidden_scheduler_task() {
        tokio::task::spawn(async move {});
        loop { break; }
    }
}
"#;
    let synthetic_ambient = r#"
fn hidden_ambient_task() {
    tokio::spawn(async move {});
    loop { break; }
    while false {}
}
"#;
    let violations =
        scheduler_task_locality_violations(synthetic_scheduler, Some(synthetic_ambient));
    assert!(
        violations.iter().any(
            |violation| violation.contains("scheduler production remainder")
                && (violation.contains("tokio::task::spawn")
                    || violation.contains("keyword \"loop\""))
        ),
        "a scheduler task/loop hidden in an impl method must be rejected: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains(SCHEDULER_AMBIENT_CHILD)),
        "an ambient child task/loop must be rejected: {violations:?}"
    );
}

#[test]
fn reviewer_mutations_reject_value_ownership_and_shared_read_drift() {
    let root = repo_root();
    let main = read_source(&root, MAIN_ROOT);
    let startup = read_source(&root, MAIN_STARTUP_CHILD);
    let runtime = read_source(&root, MAIN_RUNTIME_CHILD);
    let scheduler = read_source(&root, SCHEDULER_ROOT);

    for anchor in [
        "let config = wenlan_core::config::load_config(",
        "let reranker_cache_dir = wenlan_core::db::resolve_fastembed_cache_dir(",
        "let mut deep_bgebase_pending = false",
    ] {
        let duplicated = format!("{main}\nfn hidden_startup_value() {{ {anchor}; }}\n");
        let violations = structure_violations_with_children(
            &root,
            Phase::Runtime,
            &duplicated,
            &scheduler,
            Some(&startup),
            Some(&runtime),
            None,
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(anchor)),
            "a second startup-value owner for {anchor:?} must be rejected: {violations:?}"
        );
    }

    let changed_read = main.replacen("shared.read().await", "shared.write().await", 1);
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &changed_read,
        &scheduler,
        Some(&startup),
        Some(&runtime),
        None,
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("exactly 2 times")),
        "runtime facade shared-state read-count drift must be rejected: {violations:?}"
    );

    assert!(
        exact_token_count_violations(
            "shared.read().await; shared.read().await;",
            "shared.read().await",
            3,
            "register_optional_runtime_workers"
        )
        .iter()
        .any(|violation| violation.contains("exactly 3 times")),
        "runtime registration must reject fewer than three shared-state reads"
    );
}

#[test]
fn second_review_mutation_runtime_config_reload_is_rejected() {
    let root = repo_root();
    let main = read_source(&root, MAIN_ROOT);
    let startup = read_source(&root, MAIN_STARTUP_CHILD);
    let runtime = read_source(&root, MAIN_RUNTIME_CHILD);
    let scheduler = read_source(&root, SCHEDULER_ROOT);
    for (injected, expected) in [
        (
            "let runtime_config = wenlan_core::config::load_config();",
            "load_config",
        ),
        (
            "let alternate_cache = wenlan_core::db::resolve_fastembed_cache_dir(&data_dir);",
            "resolve_fastembed_cache_dir",
        ),
        (
            "let alternate_deep = wenlan_core::reranker::reranker_mode_resolved(&config);",
            "reranker_mode_resolved",
        ),
        (
            "use wenlan_core::config::load_config as reload;\n    \
             let runtime_config = reload();",
            "load_config",
        ),
    ] {
        let mutated = main.replacen(
            "let shared: SharedState = Arc::new(RwLock::new(server_state));",
            &format!(
                "let shared: SharedState = Arc::new(RwLock::new(server_state));\n    {injected}"
            ),
            1,
        );
        let violations = structure_violations_with_children(
            &root,
            Phase::Runtime,
            &mutated,
            &scheduler,
            Some(&startup),
            Some(&runtime),
            None,
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "an alternate runtime producer that retains the startup owner must be rejected: {violations:?}"
        );
    }
}

#[test]
fn second_review_mutations_reject_snapshot_reorder_and_stdout_deletion() {
    let root = repo_root();
    let main = read_source(&root, MAIN_ROOT);
    let startup = read_source(&root, MAIN_STARTUP_CHILD);
    let runtime = read_source(&root, MAIN_RUNTIME_CHILD);
    let scheduler = read_source(&root, SCHEDULER_ROOT);

    let reservation = r#"                let reservation = {
                    let state = shared.read().await;
                    state
                        .startup_model_load_reserved
                        .store(true, std::sync::atomic::Ordering::Release);
                    state.startup_model_load_reserved.clone()
                };
"#;
    let load_shutdown = r#"                let mut load_shutdown = {
                    let state = shared.read().await;
                    state.shutdown.subscribe()
                };
"#;
    let reordered_runtime = runtime.replacen(
        &format!("{reservation}{load_shutdown}"),
        &format!("{load_shutdown}{reservation}"),
        1,
    );
    assert_ne!(
        reordered_runtime, runtime,
        "runtime snapshot blocks must exist"
    );
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &main,
        &scheduler,
        Some(&startup),
        Some(&reordered_runtime),
        None,
    );
    assert!(
        violations.iter().any(|violation| {
            violation.contains("register_optional_runtime_workers snapshots")
                && violation.contains("reorders stage")
        }),
        "runtime snapshot reorder with all five reads retained must be rejected: {violations:?}"
    );

    let shutdown_snapshot = "    let shutdown = { shared.read().await.shutdown.clone() };\n";
    let moved = main.replacen(shutdown_snapshot, "", 1).replacen(
        "        let write_signal = {",
        &format!("{shutdown_snapshot}        let write_signal = {{"),
        1,
    );
    assert_ne!(moved, main, "facade shutdown snapshot must move");
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &moved,
        &scheduler,
        Some(&startup),
        Some(&runtime),
        None,
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("run_daemon facade snapshots")),
        "facade snapshot movement with all five reads retained must be rejected: {violations:?}"
    );

    let without_stdout = runtime.replacen(
        "    println!(\"WENLAN_LISTENING_ON={}\", local_addr);\n",
        "",
        1,
    );
    assert_ne!(without_stdout, runtime, "stdout announcement must exist");
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &main,
        &scheduler,
        Some(&startup),
        Some(&without_stdout),
        None,
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("println!")),
        "stdout discovery-line deletion must be rejected: {violations:?}"
    );

    let changed_stdout = runtime.replacen(
        "println!(\"WENLAN_LISTENING_ON={}\", local_addr);",
        "println!(\"WENLAN_PORT={}\", local_addr);",
        1,
    );
    assert_ne!(
        changed_stdout, runtime,
        "stdout announcement format must exist"
    );
    let violations = structure_violations_with_children(
        &root,
        Phase::Runtime,
        &main,
        &scheduler,
        Some(&startup),
        Some(&changed_stdout),
        None,
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("must remain exactly")),
        "stdout discovery-line format drift must be rejected: {violations:?}"
    );
}

#[test]
fn reviewer_mutations_reject_module_spoof_and_hidden_test_files() {
    let spoofed = r#"
// #[cfg(test)]
// #[path = "bind_addr_tests.rs"]
// mod bind_addr_tests;
const SPOOF: &str = "mod bind_addr_tests;";
#[cfg(test)]
#[path = "wrong.rs"]
#[path = "also-wrong.rs"]
pub(crate) mod bind_addr_tests;
"#;
    let violations = test_module_contract_violations(
        spoofed,
        "bind_addr_tests",
        false,
        Some("bind_addr_tests.rs"),
        "synthetic main",
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("must stay private")),
        "comment/string declarations must not spoof visibility: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("wrong.rs")),
        "comment/string declarations must not spoof path ownership: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("2 adjacent path attributes")),
        "duplicate path attributes must be rejected: {violations:?}"
    );

    let expected = BTreeSet::from(["scheduler_tests.rs".to_string()]);
    let actual = BTreeSet::from([
        "scheduler_tests.rs".to_string(),
        "hidden_tests.rs".to_string(),
    ]);
    let violations = exact_child_files_violations(&actual, &expected, "scheduler");
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("hidden_tests.rs")),
        "suffix-based filtering must not hide an extra test child: {violations:?}"
    );
}
