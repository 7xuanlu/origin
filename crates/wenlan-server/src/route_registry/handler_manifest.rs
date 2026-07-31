use super::{Builder, HandlerBinding, ReaderMethod};
use std::collections::BTreeSet;

const MANIFEST: &str = include_str!("handler_manifest.txt");

pub(super) fn bindings() -> Vec<HandlerBinding> {
    parse(MANIFEST)
}

fn parse(input: &'static str) -> Vec<HandlerBinding> {
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();

    for (index, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields = line.split('|').collect::<Vec<_>>();
        assert_eq!(
            fields.len(),
            4,
            "malformed handler manifest row {}: expected four fields",
            index + 1
        );
        let builder = match fields[0] {
            "main" => Builder::Main,
            "repair" => Builder::Repair,
            other => panic!(
                "malformed handler manifest row {}: unknown builder {other}",
                index + 1
            ),
        };
        let method = match fields[1] {
            "get" => ReaderMethod::Get,
            "post" => ReaderMethod::Post,
            "put" => ReaderMethod::Put,
            "delete" => ReaderMethod::Delete,
            "patch" => ReaderMethod::Patch,
            other => panic!(
                "malformed handler manifest row {}: unknown method {other}",
                index + 1
            ),
        };
        let path = fields[2];
        let handler = fields[3];
        assert!(
            path.starts_with('/'),
            "malformed handler manifest row {}: path must start with /",
            index + 1
        );
        assert!(
            handler.starts_with("wenlan_server::"),
            "malformed handler manifest row {}: handler must be an exact wenlan_server type path",
            index + 1
        );

        let row = HandlerBinding {
            builder,
            method,
            path,
            handler,
        };
        assert!(
            seen.insert(row),
            "duplicate handler manifest row {}: {row:?}",
            index + 1
        );
        rows.push(row);
    }

    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "expected four fields")]
    fn malformed_rows_fail_loud() {
        let _ = parse("main|get|/api/health");
    }

    #[test]
    #[should_panic(expected = "duplicate handler manifest row")]
    fn duplicate_rows_fail_loud() {
        let _ = parse(
            "main|get|/api/health|wenlan_server::routes::handle_health\n\
             main|get|/api/health|wenlan_server::routes::handle_health",
        );
    }
}
