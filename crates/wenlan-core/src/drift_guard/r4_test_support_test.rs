// SPDX-License-Identifier: Apache-2.0

use proc_macro2::{Delimiter, Span, TokenStream, TokenTree};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{
    Attribute, Expr, ExprCall, ExprField, ExprMethodCall, ImplItem, Item, Member, Meta, Pat,
    Path as SynPath, Type, Visibility,
};

const RAW_MANIFEST_PATH: &str =
    "crates/wenlan-core/src/drift_guard/r4_test_support_raw_manifest.txt";
const SUPPORT_MANIFEST_PATH: &str =
    "crates/wenlan-core/src/drift_guard/r4_test_support_api_manifest.txt";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum RawShape {
    PrimaryConnLock,
    PrimaryConnTryLock,
    AlternateDbField,
    ConnFieldEscape,
    StandaloneLibsqlOrigin,
}

impl RawShape {
    fn as_str(self) -> &'static str {
        match self {
            Self::PrimaryConnLock => "PrimaryConnLock",
            Self::PrimaryConnTryLock => "PrimaryConnTryLock",
            Self::AlternateDbField => "AlternateDbField",
            Self::ConnFieldEscape => "ConnFieldEscape",
            Self::StandaloneLibsqlOrigin => "StandaloneLibsqlOrigin",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "PrimaryConnLock" => Some(Self::PrimaryConnLock),
            "PrimaryConnTryLock" => Some(Self::PrimaryConnTryLock),
            "AlternateDbField" => Some(Self::AlternateDbField),
            "ConnFieldEscape" => Some(Self::ConnFieldEscape),
            "StandaloneLibsqlOrigin" => Some(Self::StandaloneLibsqlOrigin),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RawIdentity {
    path: String,
    owner: String,
    shape: RawShape,
    ordinal: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RawUse {
    identity: RawIdentity,
    line: usize,
    test_only: bool,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SupportIdentity {
    path: String,
    owner: String,
    callee: String,
    ordinal: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SupportCall {
    identity: SupportIdentity,
    line: usize,
    test_only: bool,
}

#[derive(Debug, Default)]
struct Analysis {
    raw_uses: Vec<RawUse>,
    support_calls: Vec<SupportCall>,
    errors: Vec<String>,
    visited_files: BTreeSet<String>,
}

#[derive(Debug, Eq, PartialEq)]
struct ManifestMismatch(String);

#[derive(Clone, Debug)]
struct PendingRawUse {
    owner: String,
    shape: RawShape,
    line: usize,
    test_only: bool,
}

struct RawVisitor<'analysis> {
    owner: &'analysis str,
    test_only: bool,
    builder_aliases: BTreeSet<String>,
    pending: &'analysis mut Vec<PendingRawUse>,
    errors: &'analysis mut Vec<String>,
}

impl RawVisitor<'_> {
    fn push(&mut self, shape: RawShape, span: Span) {
        self.pending.push(PendingRawUse {
            owner: self.owner.to_string(),
            shape,
            line: span.start().line,
            test_only: self.test_only,
        });
    }
}

impl<'ast> Visit<'ast> for RawVisitor<'_> {
    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        let method = node.method.to_string();
        if method == "lock" && expression_is_conn_field(&node.receiver) {
            self.push(RawShape::PrimaryConnLock, node.span());
            for argument in &node.args {
                self.visit_expr(argument);
            }
            return;
        }
        if method == "try_lock" && expression_is_conn_field(&node.receiver) {
            self.push(RawShape::PrimaryConnTryLock, node.span());
            for argument in &node.args {
                self.visit_expr(argument);
            }
            return;
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(path) = node.func.as_ref() {
            if is_libsql_builder_constructor(&path.path, &self.builder_aliases) {
                self.push(RawShape::StandaloneLibsqlOrigin, node.span());
            }
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_field(&mut self, node: &'ast ExprField) {
        match &node.member {
            Member::Named(member) if member == "_db" => {
                self.push(RawShape::AlternateDbField, node.span());
            }
            Member::Named(member) if member == "conn" => {
                self.push(RawShape::ConnFieldEscape, node.span());
            }
            _ => {}
        }
        visit::visit_expr_field(self, node);
    }

    fn visit_pat(&mut self, node: &'ast Pat) {
        if let Pat::Struct(pattern) = node {
            for field in &pattern.fields {
                let shape = match &field.member {
                    Member::Named(member) if member == "_db" => Some(RawShape::AlternateDbField),
                    Member::Named(member) if member == "conn" => Some(RawShape::ConnFieldEscape),
                    _ => None,
                };
                if let Some(shape) = shape {
                    self.push(shape, field.span());
                }
            }
        }
        visit::visit_pat(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        scan_macro_tokens(
            &node.tokens,
            self.owner,
            self.test_only,
            &self.builder_aliases,
            self.pending,
            self.errors,
        );
    }

    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        collect_libsql_builder_aliases_from_tree(
            &node.tree,
            &mut Vec::new(),
            &mut self.builder_aliases,
        );
    }
}

fn expression_is_conn_field(expression: &Expr) -> bool {
    match expression {
        Expr::Field(field) => matches!(&field.member, Member::Named(member) if member == "conn"),
        Expr::Group(group) => expression_is_conn_field(&group.expr),
        Expr::Paren(paren) => expression_is_conn_field(&paren.expr),
        Expr::Reference(reference) => expression_is_conn_field(&reference.expr),
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
            expression_is_conn_field(&unary.expr)
        }
        _ => false,
    }
}

fn path_ends_with(path: &SynPath, expected: &[&str]) -> bool {
    let segments: Vec<String> = path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    segments.len() >= expected.len()
        && segments[segments.len() - expected.len()..]
            .iter()
            .map(String::as_str)
            .eq(expected.iter().copied())
}

fn is_libsql_builder_constructor(path: &SynPath, builder_aliases: &BTreeSet<String>) -> bool {
    let segments: Vec<String> = path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    (segments.len() >= 3
        && segments[segments.len() - 3] == "libsql"
        && segments[segments.len() - 2] == "Builder"
        && segments[segments.len() - 1].starts_with("new_"))
        || (segments.len() == 2
            && builder_aliases.contains(&segments[0])
            && segments[1].starts_with("new_"))
}

fn collect_libsql_builder_aliases_from_tree(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    output: &mut BTreeSet<String>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            collect_libsql_builder_aliases_from_tree(&path.tree, prefix, output);
            prefix.pop();
        }
        syn::UseTree::Name(name) => {
            let mut full = prefix.clone();
            full.push(name.ident.to_string());
            if full == ["libsql", "Builder"] {
                output.insert(name.ident.to_string());
            }
        }
        syn::UseTree::Rename(rename) => {
            let mut full = prefix.clone();
            full.push(rename.ident.to_string());
            if full == ["libsql", "Builder"] {
                output.insert(rename.rename.to_string());
            }
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_libsql_builder_aliases_from_tree(item, prefix, output);
            }
        }
        syn::UseTree::Glob(_) => {}
    }
}

fn collect_libsql_builder_aliases(items: &[Item]) -> BTreeSet<String> {
    let mut aliases = BTreeSet::new();
    for item in items {
        if let Item::Use(item_use) = item {
            collect_libsql_builder_aliases_from_tree(&item_use.tree, &mut Vec::new(), &mut aliases);
        }
    }
    aliases
}

#[derive(Clone)]
struct FlatToken {
    text: String,
    span: Span,
}

fn flatten_tokens(stream: &TokenStream, output: &mut Vec<FlatToken>) {
    for token in stream.clone() {
        match token {
            TokenTree::Group(group) => {
                let (open, close) = match group.delimiter() {
                    Delimiter::Parenthesis => ("(", ")"),
                    Delimiter::Brace => ("{", "}"),
                    Delimiter::Bracket => ("[", "]"),
                    Delimiter::None => ("", ""),
                };
                output.push(FlatToken {
                    text: open.to_string(),
                    span: group.span_open(),
                });
                flatten_tokens(&group.stream(), output);
                output.push(FlatToken {
                    text: close.to_string(),
                    span: group.span_close(),
                });
            }
            TokenTree::Ident(ident) => output.push(FlatToken {
                text: ident.to_string(),
                span: ident.span(),
            }),
            TokenTree::Punct(punctuation) => output.push(FlatToken {
                text: punctuation.as_char().to_string(),
                span: punctuation.span(),
            }),
            TokenTree::Literal(literal) => output.push(FlatToken {
                text: literal.to_string(),
                span: literal.span(),
            }),
        }
    }
}

fn scan_macro_tokens(
    stream: &TokenStream,
    owner: &str,
    test_only: bool,
    builder_aliases: &BTreeSet<String>,
    pending: &mut Vec<PendingRawUse>,
    errors: &mut Vec<String>,
) {
    let mut tokens = Vec::new();
    flatten_tokens(stream, &mut tokens);
    let texts: Vec<&str> = tokens.iter().map(|token| token.text.as_str()).collect();
    let mut consumed_conn = BTreeSet::new();

    for index in 0..texts.len() {
        let rest = &texts[index..];
        let shape = if rest.starts_with(&[".", "conn", ".", "lock", "(", ")"]) {
            consumed_conn.insert(index);
            Some(RawShape::PrimaryConnLock)
        } else if rest.starts_with(&[".", "conn", ".", "try_lock", "(", ")"]) {
            consumed_conn.insert(index);
            Some(RawShape::PrimaryConnTryLock)
        } else if rest.starts_with(&[".", "_db"]) {
            Some(RawShape::AlternateDbField)
        } else if (rest.len() >= 7
            && rest[..6] == ["libsql", ":", ":", "Builder", ":", ":"]
            && rest[6].starts_with("new_"))
            || (rest.len() >= 4
                && builder_aliases.contains(rest[0])
                && rest[1..3] == [":", ":"]
                && rest[3].starts_with("new_"))
        {
            Some(RawShape::StandaloneLibsqlOrigin)
        } else {
            None
        };
        if let Some(shape) = shape {
            pending.push(PendingRawUse {
                owner: owner.to_string(),
                shape,
                line: tokens[index].span.start().line,
                test_only,
            });
        }
    }

    for index in 0..texts.len() {
        if texts[index..].starts_with(&[".", "conn"]) && !consumed_conn.contains(&index) {
            pending.push(PendingRawUse {
                owner: owner.to_string(),
                shape: RawShape::ConnFieldEscape,
                line: tokens[index].span.start().line,
                test_only,
            });
        }
    }

    for index in 0..texts.len().saturating_sub(1) {
        if texts[index] != "MemoryDB" || texts[index + 1] != "{" {
            continue;
        }
        let mut depth = 1usize;
        for field_index in index + 2..texts.len() {
            match texts[field_index] {
                "{" => depth += 1,
                "}" => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                "conn" if depth == 1 => pending.push(PendingRawUse {
                    owner: owner.to_string(),
                    shape: RawShape::ConnFieldEscape,
                    line: tokens[field_index].span.start().line,
                    test_only,
                }),
                "_db" if depth == 1 => pending.push(PendingRawUse {
                    owner: owner.to_string(),
                    shape: RawShape::AlternateDbField,
                    line: tokens[field_index].span.start().line,
                    test_only,
                }),
                _ => {}
            }
        }
    }

    for index in 0..texts.len() {
        let (alias, method_index) =
            if texts[index..].starts_with(&["libsql", ":", ":", "Builder", ":", ":"]) {
                ("libsql::Builder".to_string(), index + 6)
            } else if index + 3 < texts.len()
                && builder_aliases.contains(texts[index])
                && texts[index + 1..index + 3] == [":", ":"]
            {
                (texts[index].to_string(), index + 3)
            } else {
                continue;
            };
        let Some(method) = texts.get(method_index) else {
            errors.push(format!(
                "{owner}: unclassified macro Builder target {alias}::<missing>"
            ));
            continue;
        };
        if !method.starts_with("new_") {
            errors.push(format!(
                "{owner}: unclassified macro Builder target {alias}::{method}"
            ));
        }
    }

    let known_session_method = |method: &str| {
        matches!(
            method,
            "execute"
                | "execute_batch"
                | "query"
                | "begin_immediate"
                | "begin_read_only"
                | "commit"
                | "rollback"
                | "structural_digest"
                | "repair_database_content_digest"
                | "check_seed_contract"
        )
    };
    let metavariables: BTreeSet<&str> = texts
        .windows(4)
        .filter_map(|window| (window[0] == "$" && window[2] == ":").then_some(window[1]))
        .collect();
    let mut session_candidate_metavariables = BTreeSet::new();
    if metavariables.contains("session") {
        session_candidate_metavariables.insert("session");
    }
    for index in 1..texts.len().saturating_sub(1) {
        if texts[index] != "." || !known_session_method(texts[index + 1]) {
            continue;
        }
        let receiver_start = texts[..index]
            .iter()
            .rposition(|text| matches!(*text, ";" | ","))
            .map_or(0, |separator| separator + 1);
        if let Some(variable) = texts[receiver_start..index]
            .windows(2)
            .find_map(|pair| (pair[0] == "$" && metavariables.contains(pair[1])).then_some(pair[1]))
        {
            session_candidate_metavariables.insert(variable);
        }
    }
    for index in 0..texts.len().saturating_sub(3) {
        if texts[index] == "TestDbSession"
            && texts[index + 1..index + 3] == [":", ":"]
            && !known_session_method(texts[index + 3])
        {
            errors.push(format!(
                "{owner}: unclassified macro support target TestDbSession::{}",
                texts[index + 3]
            ));
        }
    }
    for index in 1..texts.len().saturating_sub(1) {
        if texts[index] != "." {
            continue;
        }
        let method = texts[index + 1];
        let receiver_start = texts[..index]
            .iter()
            .rposition(|text| matches!(*text, ";" | ","))
            .map_or(0, |separator| separator + 1);
        let receiver = &texts[receiver_start..index];
        let metavariable_receiver = receiver.windows(2).find_map(|pair| {
            (pair[0] == "$" && metavariables.contains(pair[1])).then_some(pair[1])
        });
        if let Some(variable) = metavariable_receiver {
            if session_candidate_metavariables.contains(variable) {
                errors.push(format!(
                    "{owner}: unproven macro support target ${variable}.{method}"
                ));
                continue;
            }
        }
        let target_bearing = receiver.contains(&"TestDbSession");
        let wrapper = matches!(method, "expect" | "unwrap" | "clone" | "as_ref" | "as_mut");
        if target_bearing && !known_session_method(method) && !wrapper {
            errors.push(format!(
                "{owner}: unclassified macro support target {method}"
            ));
        }
    }
}

fn finalize_raw_uses(path: &str, pending: Vec<PendingRawUse>) -> Vec<RawUse> {
    let mut ordinals = BTreeMap::<(String, RawShape), usize>::new();
    pending
        .into_iter()
        .map(|raw_use| {
            let key = (raw_use.owner.clone(), raw_use.shape);
            let ordinal = ordinals.entry(key).or_default();
            *ordinal += 1;
            let identity = RawIdentity {
                path: path.to_string(),
                owner: raw_use.owner,
                shape: raw_use.shape,
                ordinal: *ordinal,
            };
            RawUse {
                identity,
                line: raw_use.line,
                test_only: raw_use.test_only,
            }
        })
        .collect()
}

fn analyze_source_for_test(source: &str, owner: &str) -> Analysis {
    let syntax = match syn::parse_file(source) {
        Ok(syntax) => syntax,
        Err(error) => {
            return Analysis {
                errors: vec![format!(
                    "synthetic.rs:{}: {error}",
                    error.span().start().line
                )],
                ..Analysis::default()
            };
        }
    };
    let mut pending = Vec::new();
    let mut errors = Vec::new();
    let builder_aliases = collect_libsql_builder_aliases(&syntax.items);
    let mut visitor = RawVisitor {
        owner,
        test_only: true,
        builder_aliases,
        pending: &mut pending,
        errors: &mut errors,
    };
    visitor.visit_file(&syntax);
    Analysis {
        raw_uses: finalize_raw_uses("synthetic.rs", pending),
        support_calls: Vec::new(),
        errors,
        visited_files: BTreeSet::new(),
    }
}

fn manifest_row(identity: &RawIdentity) -> String {
    format!(
        "{}|{}|{}|{}",
        identity.path,
        identity.owner,
        identity.shape.as_str(),
        identity.ordinal
    )
}

fn parse_raw_manifest(manifest: &str) -> Result<Vec<RawIdentity>, ManifestMismatch> {
    let mut identities = Vec::new();
    let mut seen = BTreeSet::new();
    for (line_index, line) in manifest.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('|').collect();
        if fields.len() != 4 {
            return Err(ManifestMismatch(format!(
                "raw manifest line {} must have four fields",
                line_index + 1
            )));
        }
        let shape = RawShape::parse(fields[2]).ok_or_else(|| {
            ManifestMismatch(format!(
                "raw manifest line {} has unknown shape {:?}",
                line_index + 1,
                fields[2]
            ))
        })?;
        let ordinal = fields[3].parse::<usize>().map_err(|_| {
            ManifestMismatch(format!(
                "raw manifest line {} has invalid ordinal {:?}",
                line_index + 1,
                fields[3]
            ))
        })?;
        let identity = RawIdentity {
            path: fields[0].to_string(),
            owner: fields[1].to_string(),
            shape,
            ordinal,
        };
        if !seen.insert(identity.clone()) {
            return Err(ManifestMismatch(format!(
                "raw manifest contains duplicate row {}",
                manifest_row(&identity)
            )));
        }
        identities.push(identity);
    }
    Ok(identities)
}

fn compare_raw_manifest(actual: &[RawUse], manifest: &str) -> Result<(), ManifestMismatch> {
    let mut expected = parse_raw_manifest(manifest)?;
    let mut actual_identities: Vec<RawIdentity> = actual
        .iter()
        .map(|raw_use| raw_use.identity.clone())
        .collect();
    expected.sort();
    actual_identities.sort();
    if actual_identities == expected {
        Ok(())
    } else {
        let expected_set: BTreeSet<_> = expected.iter().cloned().collect();
        let actual_set: BTreeSet<_> = actual_identities.iter().cloned().collect();
        let removed: Vec<String> = expected_set
            .difference(&actual_set)
            .map(manifest_row)
            .collect();
        let added: Vec<String> = actual_set
            .difference(&expected_set)
            .map(manifest_row)
            .collect();
        Err(ManifestMismatch(format!(
            "raw manifest mismatch: removed={removed:?}, added={added:?}"
        )))
    }
}

fn support_manifest_row(identity: &SupportIdentity) -> String {
    format!(
        "{}|{}|{}|{}",
        identity.path, identity.owner, identity.callee, identity.ordinal
    )
}

fn parse_support_manifest(manifest: &str) -> Result<Vec<SupportIdentity>, ManifestMismatch> {
    let mut identities = Vec::new();
    let mut seen = BTreeSet::new();
    for (line_index, line) in manifest.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('|').collect();
        if fields.len() != 4 {
            return Err(ManifestMismatch(format!(
                "support manifest line {} must have four fields",
                line_index + 1
            )));
        }
        let ordinal = fields[3].parse::<usize>().map_err(|_| {
            ManifestMismatch(format!(
                "support manifest line {} has invalid ordinal {:?}",
                line_index + 1,
                fields[3]
            ))
        })?;
        let identity = SupportIdentity {
            path: fields[0].to_string(),
            owner: fields[1].to_string(),
            callee: fields[2].to_string(),
            ordinal,
        };
        if !seen.insert(identity.clone()) {
            return Err(ManifestMismatch(format!(
                "support manifest contains duplicate row {}",
                support_manifest_row(&identity)
            )));
        }
        identities.push(identity);
    }
    Ok(identities)
}

fn compare_support_manifest(
    actual: &[SupportCall],
    manifest: &str,
) -> Result<(), ManifestMismatch> {
    let mut expected = parse_support_manifest(manifest)?;
    let mut actual_identities: Vec<SupportIdentity> =
        actual.iter().map(|call| call.identity.clone()).collect();
    expected.sort();
    actual_identities.sort();
    if actual_identities == expected {
        Ok(())
    } else {
        let expected_set: BTreeSet<_> = expected.iter().cloned().collect();
        let actual_set: BTreeSet<_> = actual_identities.iter().cloned().collect();
        let removed: Vec<String> = expected_set
            .difference(&actual_set)
            .map(support_manifest_row)
            .collect();
        let added: Vec<String> = actual_set
            .difference(&expected_set)
            .map(support_manifest_row)
            .collect();
        Err(ManifestMismatch(format!(
            "support manifest mismatch: removed={removed:?}, added={added:?}"
        )))
    }
}

fn attribute_is_exact_cfg_test(attribute: &Attribute) -> bool {
    if !attribute.path().is_ident("cfg") {
        return false;
    }
    let Meta::List(meta) = &attribute.meta else {
        return false;
    };
    let mut tokens = meta.tokens.clone().into_iter();
    matches!(tokens.next(), Some(TokenTree::Ident(ident)) if ident == "test")
        && tokens.next().is_none()
}

fn attribute_is_test(attribute: &Attribute) -> bool {
    let segments: Vec<String> = attribute
        .path()
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    matches!(segments.as_slice(), [test] if test == "test")
        || matches!(segments.as_slice(), [runtime, test] if runtime == "tokio" && test == "test")
}

fn attrs_are_test_only(attributes: &[Attribute]) -> bool {
    attributes.iter().any(attribute_is_exact_cfg_test) || attributes.iter().any(attribute_is_test)
}

fn path_attribute(attributes: &[Attribute]) -> Option<String> {
    attributes.iter().find_map(|attribute| {
        if !attribute.path().is_ident("path") {
            return None;
        }
        let Meta::NameValue(name_value) = &attribute.meta else {
            return None;
        };
        let Expr::Lit(literal) = &name_value.value else {
            return None;
        };
        let syn::Lit::Str(path) = &literal.lit else {
            return None;
        };
        Some(path.value())
    })
}

fn canonical_type_name(ty: &Type) -> String {
    if let Type::Path(path) = ty {
        return path
            .path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect::<Vec<_>>()
            .join("::");
    }
    "<impl>".to_string()
}

fn external_module_path(
    source_file: &Path,
    module_dir: &Path,
    inside_inline_module: bool,
    item: &syn::ItemMod,
) -> Result<PathBuf, String> {
    if let Some(path) = path_attribute(&item.attrs) {
        let base = if inside_inline_module {
            module_dir
        } else {
            source_file.parent().unwrap_or_else(|| Path::new(""))
        };
        let candidate = base.join(path);
        return candidate.canonicalize().map_err(|error| {
            format!(
                "{}: unresolved #[path] module {}: {error}",
                source_file.display(),
                candidate.display()
            )
        });
    }

    let module = item.ident.to_string();
    let candidates = [
        module_dir.join(format!("{module}.rs")),
        module_dir.join(&module).join("mod.rs"),
    ];
    let existing: Vec<PathBuf> = candidates
        .iter()
        .filter(|candidate| candidate.is_file())
        .cloned()
        .collect();
    match existing.as_slice() {
        [candidate] => candidate.canonicalize().map_err(|error| {
            format!(
                "{}: cannot canonicalize module {}: {error}",
                source_file.display(),
                candidate.display()
            )
        }),
        [] => Err(format!(
            "{}: unresolved module {} (tried {:?})",
            source_file.display(),
            module,
            candidates
        )),
        _ => Err(format!(
            "{}: ambiguous module {} resolves to {:?}",
            source_file.display(),
            module,
            existing
        )),
    }
}

fn child_module_dir(source_file: &Path) -> PathBuf {
    let parent = source_file
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .to_path_buf();
    if source_file.file_name().and_then(|name| name.to_str()) == Some("mod.rs") {
        parent
    } else {
        let stem = source_file
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default();
        parent.join(stem)
    }
}

fn scan_with_owner(
    node: &impl for<'ast> VisitTarget<'ast>,
    owner: &str,
    test_only: bool,
    builder_aliases: &BTreeSet<String>,
    pending: &mut Vec<PendingRawUse>,
    errors: &mut Vec<String>,
) {
    let mut visitor = RawVisitor {
        owner,
        test_only,
        builder_aliases: builder_aliases.clone(),
        pending,
        errors,
    };
    node.visit_with(&mut visitor);
}

trait VisitTarget<'ast> {
    fn visit_with(&'ast self, visitor: &mut RawVisitor<'_>);
}

impl<'ast> VisitTarget<'ast> for syn::Block {
    fn visit_with(&'ast self, visitor: &mut RawVisitor<'_>) {
        visitor.visit_block(self);
    }
}

impl<'ast> VisitTarget<'ast> for syn::Expr {
    fn visit_with(&'ast self, visitor: &mut RawVisitor<'_>) {
        visitor.visit_expr(self);
    }
}

impl<'ast> VisitTarget<'ast> for syn::ItemMacro {
    fn visit_with(&'ast self, visitor: &mut RawVisitor<'_>) {
        visitor.visit_item_macro(self);
    }
}

struct NestedFunctionCandidateVisitor<'analysis> {
    outer_owner: &'analysis str,
    test_only: bool,
    builder_aliases: &'analysis BTreeSet<String>,
    errors: &'analysis mut Vec<String>,
}

impl<'ast> Visit<'ast> for NestedFunctionCandidateVisitor<'_> {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let owner = format!("{}::{}", self.outer_owner, node.sig.ident);
        let test_only = self.test_only || attrs_are_test_only(&node.attrs);
        let mut raw = Vec::new();
        scan_with_owner(
            node.block.as_ref(),
            &owner,
            test_only,
            self.builder_aliases,
            &mut raw,
            self.errors,
        );
        let mut support = Vec::new();
        scan_support_block(
            &node.sig,
            node.block.as_ref(),
            &owner,
            test_only,
            &mut support,
            self.errors,
        );
        if !raw.is_empty() || !support.is_empty() {
            self.errors.push(format!(
                "nested function {owner} contains a raw or test-support candidate"
            ));
        }
        visit::visit_block(self, node.block.as_ref());
    }
}

fn reject_nested_function_candidates(
    block: &syn::Block,
    owner: &str,
    test_only: bool,
    builder_aliases: &BTreeSet<String>,
    errors: &mut Vec<String>,
) {
    let mut visitor = NestedFunctionCandidateVisitor {
        outer_owner: owner,
        test_only,
        builder_aliases,
        errors,
    };
    visitor.visit_block(block);
}

struct ModuleScanContext<'path> {
    repo_root: &'path Path,
    source_file: &'path Path,
    module_dir: &'path Path,
    inside_inline_module: bool,
    module_owner: &'path str,
    inherited_test: bool,
}

fn scan_items(
    context: ModuleScanContext<'_>,
    items: &[Item],
    output: &mut Analysis,
    visited: &mut BTreeSet<(PathBuf, String, bool)>,
) {
    let relative_path = context
        .source_file
        .strip_prefix(context.repo_root)
        .unwrap_or(context.source_file)
        .to_string_lossy()
        .replace('\\', "/");
    let raw_exempt = relative_path == "crates/wenlan-core/src/db.rs"
        || relative_path.starts_with("crates/wenlan-core/src/db/");
    let builder_aliases = collect_libsql_builder_aliases(items);
    let mut file_pending = Vec::new();
    let mut file_support_pending = Vec::new();

    for item in items {
        match item {
            Item::Fn(function) => {
                if raw_exempt {
                    continue;
                }
                let owner = format!("{}::{}", context.module_owner, function.sig.ident);
                let test_only = context.inherited_test || attrs_are_test_only(&function.attrs);
                scan_with_owner(
                    function.block.as_ref(),
                    &owner,
                    test_only,
                    &builder_aliases,
                    &mut file_pending,
                    &mut output.errors,
                );
                scan_support_block(
                    &function.sig,
                    function.block.as_ref(),
                    &owner,
                    test_only,
                    &mut file_support_pending,
                    &mut output.errors,
                );
                reject_nested_function_candidates(
                    function.block.as_ref(),
                    &owner,
                    test_only,
                    &builder_aliases,
                    &mut output.errors,
                );
            }
            Item::Impl(implementation) => {
                if raw_exempt {
                    continue;
                }
                let type_name = canonical_type_name(&implementation.self_ty);
                let impl_test =
                    context.inherited_test || attrs_are_test_only(&implementation.attrs);
                for impl_item in &implementation.items {
                    if let ImplItem::Fn(function) = impl_item {
                        let owner = format!(
                            "{}::{type_name}::{}",
                            context.module_owner, function.sig.ident
                        );
                        let test_only = impl_test || attrs_are_test_only(&function.attrs);
                        scan_with_owner(
                            &function.block,
                            &owner,
                            test_only,
                            &builder_aliases,
                            &mut file_pending,
                            &mut output.errors,
                        );
                        scan_support_block(
                            &function.sig,
                            &function.block,
                            &owner,
                            test_only,
                            &mut file_support_pending,
                            &mut output.errors,
                        );
                        reject_nested_function_candidates(
                            &function.block,
                            &owner,
                            test_only,
                            &builder_aliases,
                            &mut output.errors,
                        );
                    }
                }
            }
            Item::Trait(item_trait) => {
                if raw_exempt {
                    continue;
                }
                let trait_test = context.inherited_test || attrs_are_test_only(&item_trait.attrs);
                for trait_item in &item_trait.items {
                    let syn::TraitItem::Fn(function) = trait_item else {
                        continue;
                    };
                    let Some(block) = &function.default else {
                        continue;
                    };
                    let owner = format!(
                        "{}::{}::{}",
                        context.module_owner, item_trait.ident, function.sig.ident
                    );
                    let test_only = trait_test || attrs_are_test_only(&function.attrs);
                    scan_with_owner(
                        block,
                        &owner,
                        test_only,
                        &builder_aliases,
                        &mut file_pending,
                        &mut output.errors,
                    );
                    scan_support_block(
                        &function.sig,
                        block,
                        &owner,
                        test_only,
                        &mut file_support_pending,
                        &mut output.errors,
                    );
                    reject_nested_function_candidates(
                        block,
                        &owner,
                        test_only,
                        &builder_aliases,
                        &mut output.errors,
                    );
                }
            }
            Item::Mod(module) => {
                let child_owner = format!("{}::{}", context.module_owner, module.ident);
                let child_test = context.inherited_test || attrs_are_test_only(&module.attrs);
                if let Some((_, inline_items)) = &module.content {
                    let inline_module_dir = context.module_dir.join(module.ident.to_string());
                    scan_items(
                        ModuleScanContext {
                            repo_root: context.repo_root,
                            source_file: context.source_file,
                            module_dir: &inline_module_dir,
                            inside_inline_module: true,
                            module_owner: &child_owner,
                            inherited_test: child_test,
                        },
                        inline_items,
                        output,
                        visited,
                    );
                } else {
                    match external_module_path(
                        context.source_file,
                        context.module_dir,
                        context.inside_inline_module,
                        module,
                    ) {
                        Ok(child_file) => scan_module_file(
                            context.repo_root,
                            &child_file,
                            &child_module_dir(&child_file),
                            &child_owner,
                            child_test,
                            output,
                            visited,
                        ),
                        Err(error) => output.errors.push(error),
                    }
                }
            }
            Item::Macro(item_macro) if !raw_exempt => {
                let owner = format!("{}::<module>", context.module_owner);
                let test_only = context.inherited_test || attrs_are_test_only(&item_macro.attrs);
                scan_with_owner(
                    item_macro,
                    &owner,
                    test_only,
                    &builder_aliases,
                    &mut file_pending,
                    &mut output.errors,
                );
                let locals = BTreeMap::new();
                let mut visitor = SupportExprVisitor {
                    owner: &owner,
                    test_only,
                    locals: &locals,
                    pending: &mut file_support_pending,
                    errors: &mut output.errors,
                };
                visitor.visit_item_macro(item_macro);
            }
            Item::Const(item_const) if !raw_exempt => {
                let owner = format!("{}::{}", context.module_owner, item_const.ident);
                let test_only = context.inherited_test || attrs_are_test_only(&item_const.attrs);
                scan_with_owner(
                    item_const.expr.as_ref(),
                    &owner,
                    test_only,
                    &builder_aliases,
                    &mut file_pending,
                    &mut output.errors,
                );
            }
            Item::Static(item_static) if !raw_exempt => {
                let owner = format!("{}::{}", context.module_owner, item_static.ident);
                let test_only = context.inherited_test || attrs_are_test_only(&item_static.attrs);
                scan_with_owner(
                    item_static.expr.as_ref(),
                    &owner,
                    test_only,
                    &builder_aliases,
                    &mut file_pending,
                    &mut output.errors,
                );
            }
            _ => {}
        }
    }

    output
        .raw_uses
        .extend(finalize_raw_uses(&relative_path, file_pending));
    output
        .support_calls
        .extend(finalize_support_calls(&relative_path, file_support_pending));
}

fn scan_module_file(
    repo_root: &Path,
    source_file: &Path,
    module_dir: &Path,
    module_owner: &str,
    inherited_test: bool,
    output: &mut Analysis,
    visited: &mut BTreeSet<(PathBuf, String, bool)>,
) {
    let canonical = match source_file.canonicalize() {
        Ok(path) => path,
        Err(error) => {
            output.errors.push(format!(
                "{}: cannot canonicalize module: {error}",
                source_file.display()
            ));
            return;
        }
    };
    if !visited.insert((canonical.clone(), module_owner.to_string(), inherited_test)) {
        return;
    }
    output.visited_files.insert(
        canonical
            .strip_prefix(repo_root)
            .unwrap_or(&canonical)
            .to_string_lossy()
            .replace('\\', "/"),
    );
    let source = match std::fs::read_to_string(&canonical) {
        Ok(source) => source,
        Err(error) => {
            output.errors.push(format!(
                "{}: cannot read module: {error}",
                canonical.display()
            ));
            return;
        }
    };
    let syntax = match syn::parse_file(&source) {
        Ok(syntax) => syntax,
        Err(error) => {
            output.errors.push(format!(
                "{}:{}: cannot parse module: {error}",
                canonical.display(),
                error.span().start().line
            ));
            return;
        }
    };
    scan_items(
        ModuleScanContext {
            repo_root,
            source_file: &canonical,
            module_dir,
            inside_inline_module: false,
            module_owner,
            inherited_test,
        },
        &syntax.items,
        output,
        visited,
    );
}

fn analyze_repository(repo_root: &Path) -> Analysis {
    let source_root = repo_root.join("crates/wenlan-core/src");
    let mut roots = vec![(source_root.join("lib.rs"), "crate".to_string())];
    let bin_root = source_root.join("bin");
    if let Ok(entries) = std::fs::read_dir(&bin_root) {
        let mut binaries: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|extension| extension.to_str()) == Some("rs"))
            .collect();
        binaries.sort();
        roots.extend(binaries.into_iter().map(|path| {
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("unknown")
                .to_string();
            (path, format!("bin::{stem}"))
        }));
    }

    let mut analysis = Analysis::default();
    let mut visited = BTreeSet::new();
    for (root_file, owner) in roots {
        scan_module_file(
            repo_root,
            &root_file,
            root_file.parent().unwrap_or(&source_root),
            &owner,
            false,
            &mut analysis,
            &mut visited,
        );
    }
    if repo_root.join(".git").exists() {
        let tracked_rust: Vec<String> = super::git_ls_files(repo_root, "crates/wenlan-core/src")
            .into_iter()
            .filter(|path| path.ends_with(".rs"))
            .collect();
        analysis.errors.extend(unclassified_rust_file_errors(
            repo_root,
            &tracked_rust,
            &analysis.visited_files,
        ));
    }
    analysis
}

fn unclassified_rust_file_errors(
    _repo_root: &Path,
    candidates: &[String],
    visited: &BTreeSet<String>,
) -> Vec<String> {
    candidates
        .iter()
        .filter(|path| !visited.contains(path.as_str()))
        .map(|path| {
            format!("{path}: tracked Rust candidate is outside the classified module graph")
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LocalKind {
    Session,
    Rows,
    Row,
    RepairVerificationContender,
}

#[derive(Clone, Debug)]
struct PendingSupportCall {
    owner: String,
    callee: String,
    line: usize,
    column: usize,
    test_only: bool,
}

const UNIQUE_SUPPORT_FUNCTIONS: &[&str] = &[
    "with_repair_verification_test_control",
    "assert_repair_verification_transaction_reusable",
    "rollback_repair_verification_test_transaction",
];

const UNIQUE_SUPPORT_METHODS: &[&str] = &[
    "test_primary_session",
    "test_secondary_session",
    "primary_mutex_available",
    "open_isolated_lint_snapshot_for_test",
];

fn support_callee_for_method(method: &str, receiver_kind: Option<LocalKind>) -> Option<String> {
    if UNIQUE_SUPPORT_METHODS.contains(&method) {
        return Some(method.to_string());
    }
    match (receiver_kind, method) {
        (Some(LocalKind::Session), "execute")
        | (Some(LocalKind::Session), "execute_batch")
        | (Some(LocalKind::Session), "query")
        | (Some(LocalKind::Session), "begin_immediate")
        | (Some(LocalKind::Session), "begin_read_only")
        | (Some(LocalKind::Session), "commit")
        | (Some(LocalKind::Session), "rollback")
        | (Some(LocalKind::Session), "structural_digest")
        | (Some(LocalKind::Session), "repair_database_content_digest")
        | (Some(LocalKind::Session), "check_seed_contract") => {
            Some(format!("TestDbSession::{method}"))
        }
        (Some(LocalKind::Rows), "next") => Some("TestDbRows::next".to_string()),
        (Some(LocalKind::Row), "get") => Some("TestDbRow::get".to_string()),
        (Some(LocalKind::RepairVerificationContender), "start_and_observe_pending")
        | (Some(LocalKind::RepairVerificationContender), "assert_pending")
        | (Some(LocalKind::RepairVerificationContender), "assert_entered_after_verification") => {
            Some(format!("RepairVerificationDbContender::{method}"))
        }
        _ => None,
    }
}

fn simple_path_ident(expression: &Expr) -> Option<String> {
    let Expr::Path(path) = expression else {
        return None;
    };
    if path.qself.is_none() && path.path.segments.len() == 1 {
        path.path
            .segments
            .first()
            .map(|segment| segment.ident.to_string())
    } else {
        None
    }
}

fn expression_local_kind(
    expression: &Expr,
    locals: &BTreeMap<String, LocalKind>,
) -> Option<LocalKind> {
    match expression {
        Expr::Path(_) => simple_path_ident(expression).and_then(|name| locals.get(&name).copied()),
        Expr::Await(awaited) => expression_local_kind(&awaited.base, locals),
        Expr::Group(group) => expression_local_kind(&group.expr, locals),
        Expr::Paren(paren) => expression_local_kind(&paren.expr, locals),
        Expr::Reference(reference) => expression_local_kind(&reference.expr, locals),
        Expr::Try(tried) => expression_local_kind(&tried.expr, locals),
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
            expression_local_kind(&unary.expr, locals)
        }
        Expr::MethodCall(call) => {
            let method = call.method.to_string();
            if matches!(
                method.as_str(),
                "test_primary_session" | "test_secondary_session"
            ) {
                return Some(LocalKind::Session);
            }
            let receiver = expression_local_kind(&call.receiver, locals);
            match (receiver, method.as_str()) {
                (kind, "expect" | "unwrap" | "clone" | "as_ref" | "as_mut") => kind,
                (
                    Some(LocalKind::Session),
                    "begin_immediate" | "begin_read_only" | "commit" | "rollback",
                ) => Some(LocalKind::Session),
                (Some(LocalKind::Session), "query") => Some(LocalKind::Rows),
                (Some(LocalKind::Rows), "next") => Some(LocalKind::Row),
                _ => None,
            }
        }
        Expr::Call(call) => {
            let Expr::Path(path) = call.func.as_ref() else {
                return None;
            };
            if path_ends_with(&path.path, &["RepairVerificationDbContender", "new"]) {
                Some(LocalKind::RepairVerificationContender)
            } else if path_ends_with(&path.path, &["TestDbSession", "query"]) {
                Some(LocalKind::Rows)
            } else if path_ends_with(&path.path, &["TestDbSession", "commit"])
                || path_ends_with(&path.path, &["TestDbSession", "rollback"])
                || path_ends_with(&path.path, &["TestDbSession", "begin_immediate"])
                || path_ends_with(&path.path, &["TestDbSession", "begin_read_only"])
            {
                Some(LocalKind::Session)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn pattern_bindings(pattern: &Pat, output: &mut Vec<String>) {
    match pattern {
        Pat::Ident(ident) => output.push(ident.ident.to_string()),
        Pat::Type(typed) => pattern_bindings(&typed.pat, output),
        Pat::Reference(reference) => pattern_bindings(&reference.pat, output),
        Pat::Tuple(tuple) => {
            for element in &tuple.elems {
                pattern_bindings(element, output);
            }
        }
        Pat::TupleStruct(tuple) => {
            for element in &tuple.elems {
                pattern_bindings(element, output);
            }
        }
        Pat::Struct(structure) => {
            for field in &structure.fields {
                pattern_bindings(&field.pat, output);
            }
        }
        Pat::Slice(slice) => {
            for element in &slice.elems {
                pattern_bindings(element, output);
            }
        }
        Pat::Paren(paren) => pattern_bindings(&paren.pat, output),
        _ => {}
    }
}

fn type_local_kind(ty: &Type) -> Option<LocalKind> {
    let type_name = match ty {
        Type::Path(path) => path.path.segments.last()?.ident.to_string(),
        Type::Reference(reference) => return type_local_kind(&reference.elem),
        Type::Paren(paren) => return type_local_kind(&paren.elem),
        Type::Group(group) => return type_local_kind(&group.elem),
        _ => return None,
    };
    match type_name.as_str() {
        "TestDbSession" => Some(LocalKind::Session),
        "TestDbRows" => Some(LocalKind::Rows),
        "TestDbRow" => Some(LocalKind::Row),
        _ => None,
    }
}

fn typed_pattern_kind(pattern: &Pat) -> Option<LocalKind> {
    let Pat::Type(typed) = pattern else {
        return None;
    };
    type_local_kind(&typed.ty)
}

fn support_callee_for_ufcs(path: &SynPath) -> Option<String> {
    let segments: Vec<String> = path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect();
    let [.., owner, method] = segments.as_slice() else {
        return None;
    };
    match owner.as_str() {
        "TestDbSession"
            if matches!(
                method.as_str(),
                "execute"
                    | "execute_batch"
                    | "query"
                    | "begin_immediate"
                    | "begin_read_only"
                    | "commit"
                    | "rollback"
                    | "structural_digest"
                    | "repair_database_content_digest"
                    | "check_seed_contract"
            ) =>
        {
            Some(format!("TestDbSession::{method}"))
        }
        "TestDbRows" if method == "next" => Some("TestDbRows::next".to_string()),
        "TestDbRow" if method == "get" => Some("TestDbRow::get".to_string()),
        _ => None,
    }
}

struct SupportExprVisitor<'analysis> {
    owner: &'analysis str,
    test_only: bool,
    locals: &'analysis BTreeMap<String, LocalKind>,
    pending: &'analysis mut Vec<PendingSupportCall>,
    errors: &'analysis mut Vec<String>,
}

impl SupportExprVisitor<'_> {
    fn push(&mut self, callee: impl Into<String>, span: Span) {
        self.pending.push(PendingSupportCall {
            owner: self.owner.to_string(),
            callee: callee.into(),
            line: span.start().line,
            column: span.start().column,
            test_only: self.test_only,
        });
    }
}

impl<'ast> Visit<'ast> for SupportExprVisitor<'_> {
    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        self.visit_expr(&node.receiver);
        let method = node.method.to_string();
        let receiver_kind = expression_local_kind(&node.receiver, self.locals);
        if let Some(callee) = support_callee_for_method(&method, receiver_kind) {
            self.push(callee, node.span());
        }
        for argument in &node.args {
            self.visit_expr(argument);
        }
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(path) = node.func.as_ref() {
            if let Some(callee) = support_callee_for_ufcs(&path.path) {
                self.push(callee, node.span());
            }
            if let Some(last) = path.path.segments.last() {
                let function = last.ident.to_string();
                if UNIQUE_SUPPORT_FUNCTIONS.contains(&function.as_str()) {
                    self.push(function, node.span());
                } else if function == "new"
                    && path_ends_with(&path.path, &["RepairVerificationDbContender", "new"])
                {
                    self.push("RepairVerificationDbContender::new", node.span());
                }
            }
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        let mut segments = Vec::<TokenStream>::new();
        let mut current = TokenStream::new();
        for token in node.tokens.clone() {
            if matches!(&token, TokenTree::Punct(punctuation) if matches!(punctuation.as_char(), ';' | ','))
            {
                if !current.is_empty() {
                    segments.push(current);
                    current = TokenStream::new();
                }
            } else {
                current.extend(std::iter::once(token));
            }
        }
        if !current.is_empty() {
            segments.push(current);
        }
        let mut parsed_any = false;
        let mut all_parsed = !segments.is_empty();
        let mut parsed_pending = Vec::new();
        for segment in segments {
            match syn::parse2::<Expr>(segment) {
                Ok(expression) => {
                    parsed_any = true;
                    let mut target_guard = MacroSupportTargetVisitor {
                        owner: self.owner,
                        locals: self.locals,
                        errors: self.errors,
                    };
                    target_guard.visit_expr(&expression);
                    let mut visitor = SupportExprVisitor {
                        owner: self.owner,
                        test_only: self.test_only,
                        locals: self.locals,
                        pending: &mut parsed_pending,
                        errors: self.errors,
                    };
                    visitor.visit_expr(&expression);
                }
                Err(_) => all_parsed = false,
            }
        }
        if parsed_any && all_parsed {
            parsed_pending.sort_by_key(|call| (call.line, call.column));
            self.pending.extend(parsed_pending);
            return;
        }

        let mut tokens = Vec::new();
        flatten_tokens(&node.tokens, &mut tokens);
        let texts: Vec<&str> = tokens.iter().map(|token| token.text.as_str()).collect();
        let mut depths = Vec::with_capacity(texts.len());
        let mut depth = 0usize;
        for text in &texts {
            if matches!(*text, ")" | "}" | "]") {
                depth = depth.saturating_sub(1);
            }
            depths.push(depth);
            if matches!(*text, "(" | "{" | "[") {
                depth += 1;
            }
        }
        for (index, text) in texts.iter().enumerate() {
            if index >= 3
                && texts[index - 3..index] == ["TestDbSession", ":", ":"]
                && matches!(
                    *text,
                    "execute"
                        | "execute_batch"
                        | "query"
                        | "begin_immediate"
                        | "begin_read_only"
                        | "commit"
                        | "rollback"
                        | "structural_digest"
                        | "repair_database_content_digest"
                        | "check_seed_contract"
                )
            {
                self.push(format!("TestDbSession::{text}"), tokens[index].span);
            }
            let unique =
                UNIQUE_SUPPORT_FUNCTIONS.contains(text) || UNIQUE_SUPPORT_METHODS.contains(text);
            if unique {
                self.push(*text, tokens[index].span);
                if matches!(*text, "test_primary_session" | "test_secondary_session") {
                    let base_depth = depths[index];
                    for chain_index in index + 1..texts.len().saturating_sub(1) {
                        if depths[chain_index] < base_depth
                            || (depths[chain_index] == base_depth
                                && matches!(texts[chain_index], "," | ";"))
                        {
                            break;
                        }
                        if depths[chain_index] == base_depth && texts[chain_index] == "." {
                            let method = texts[chain_index + 1];
                            if let Some(callee) =
                                support_callee_for_method(method, Some(LocalKind::Session))
                            {
                                self.push(callee, tokens[chain_index + 1].span);
                            }
                        }
                    }
                }
            }
            if *text == "new"
                && index >= 3
                && texts[index - 3..=index] == ["RepairVerificationDbContender", ":", ":", "new"]
            {
                self.push("RepairVerificationDbContender::new", tokens[index].span);
            }
            if !unique && index > 0 && texts[index - 1] == "." {
                let receiver = index
                    .checked_sub(2)
                    .and_then(|receiver_index| self.locals.get(texts[receiver_index]))
                    .copied();
                if let Some(callee) = support_callee_for_method(text, receiver) {
                    self.push(callee, tokens[index].span);
                }
            }
        }
    }

    fn visit_block(&mut self, _node: &'ast syn::Block) {
        // Blocks have their own lexical data-flow pass.
    }
}

struct MacroSupportTargetVisitor<'analysis> {
    owner: &'analysis str,
    locals: &'analysis BTreeMap<String, LocalKind>,
    errors: &'analysis mut Vec<String>,
}

impl<'ast> Visit<'ast> for MacroSupportTargetVisitor<'_> {
    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        let method = node.method.to_string();
        let receiver = expression_local_kind(&node.receiver, self.locals);
        let wrapper = matches!(
            method.as_str(),
            "expect" | "unwrap" | "clone" | "as_ref" | "as_mut"
        );
        if receiver == Some(LocalKind::Session)
            && support_callee_for_method(&method, receiver).is_none()
            && !wrapper
        {
            self.errors.push(format!(
                "{}: unclassified macro support target {method}",
                self.owner
            ));
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(path) = node.func.as_ref() {
            let segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect();
            if matches!(segments.as_slice(), [.., owner, _] if owner == "TestDbSession")
                && support_callee_for_ufcs(&path.path).is_none()
            {
                self.errors.push(format!(
                    "{}: unclassified macro support target {}",
                    self.owner,
                    segments.join("::")
                ));
            }
        }
        visit::visit_expr_call(self, node);
    }
}

struct SupportDataflow<'analysis> {
    owner: &'analysis str,
    test_only: bool,
    locals: BTreeMap<String, LocalKind>,
    pending: &'analysis mut Vec<PendingSupportCall>,
    errors: &'analysis mut Vec<String>,
}

impl SupportDataflow<'_> {
    fn scan_expression(&mut self, expression: &Expr) {
        let mut visitor = SupportExprVisitor {
            owner: self.owner,
            test_only: self.test_only,
            locals: &self.locals,
            pending: self.pending,
            errors: self.errors,
        };
        visitor.visit_expr(expression);

        let inherited = self.locals.clone();
        let mut nested = NestedSupportBlockVisitor {
            owner: self.owner,
            test_only: self.test_only,
            inherited: &inherited,
            pending: self.pending,
            errors: self.errors,
        };
        nested.visit_expr(expression);
    }

    fn scan_block(&mut self, block: &syn::Block) {
        for statement in &block.stmts {
            match statement {
                syn::Stmt::Local(local) => {
                    if let Some(initializer) = &local.init {
                        self.scan_expression(&initializer.expr);
                        let kind = typed_pattern_kind(&local.pat)
                            .or_else(|| expression_local_kind(&initializer.expr, &self.locals));
                        if let Some(kind) = kind {
                            let mut names = Vec::new();
                            pattern_bindings(&local.pat, &mut names);
                            for name in names {
                                self.locals.insert(name, kind);
                            }
                        }
                        if let Some((_, diverge)) = &initializer.diverge {
                            self.scan_expression(diverge);
                        }
                    }
                }
                syn::Stmt::Expr(expression, _) => self.scan_expression(expression),
                syn::Stmt::Macro(statement_macro) => {
                    let mut visitor = SupportExprVisitor {
                        owner: self.owner,
                        test_only: self.test_only,
                        locals: &self.locals,
                        pending: self.pending,
                        errors: self.errors,
                    };
                    visitor.visit_macro(&statement_macro.mac);
                }
                syn::Stmt::Item(_) => {}
            }
        }
    }
}

struct NestedSupportBlockVisitor<'analysis> {
    owner: &'analysis str,
    test_only: bool,
    inherited: &'analysis BTreeMap<String, LocalKind>,
    pending: &'analysis mut Vec<PendingSupportCall>,
    errors: &'analysis mut Vec<String>,
}

impl<'ast> Visit<'ast> for NestedSupportBlockVisitor<'_> {
    fn visit_block(&mut self, node: &'ast syn::Block) {
        let mut scanner = SupportDataflow {
            owner: self.owner,
            test_only: self.test_only,
            locals: self.inherited.clone(),
            pending: self.pending,
            errors: self.errors,
        };
        scanner.scan_block(node);
    }

    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        self.visit_expr(&node.cond);

        let mut inherited = self.inherited.clone();
        if let Expr::Let(let_expression) = node.cond.as_ref() {
            if let Some(kind) = expression_local_kind(&let_expression.expr, self.inherited) {
                let mut names = Vec::new();
                pattern_bindings(&let_expression.pat, &mut names);
                for name in names {
                    inherited.insert(name, kind);
                }
            }
        }
        let mut scanner = SupportDataflow {
            owner: self.owner,
            test_only: self.test_only,
            locals: inherited,
            pending: self.pending,
            errors: self.errors,
        };
        scanner.scan_block(&node.body);
    }
}

fn support_params(signature: &syn::Signature) -> BTreeMap<String, LocalKind> {
    let mut locals = BTreeMap::new();
    for input in &signature.inputs {
        let syn::FnArg::Typed(argument) = input else {
            continue;
        };
        let pattern = Pat::Type(syn::PatType {
            attrs: Vec::new(),
            pat: argument.pat.clone(),
            colon_token: Default::default(),
            ty: argument.ty.clone(),
        });
        if let Some(kind) = typed_pattern_kind(&pattern) {
            let mut names = Vec::new();
            pattern_bindings(&argument.pat, &mut names);
            for name in names {
                locals.insert(name, kind);
            }
        }
    }
    locals
}

fn scan_support_block(
    signature: &syn::Signature,
    block: &syn::Block,
    owner: &str,
    test_only: bool,
    pending: &mut Vec<PendingSupportCall>,
    errors: &mut Vec<String>,
) {
    let mut scanner = SupportDataflow {
        owner,
        test_only,
        locals: support_params(signature),
        pending,
        errors,
    };
    scanner.scan_block(block);
}

fn finalize_support_calls(path: &str, pending: Vec<PendingSupportCall>) -> Vec<SupportCall> {
    let mut ordinals = BTreeMap::<(String, String), usize>::new();
    pending
        .into_iter()
        .map(|call| {
            let ordinal = ordinals
                .entry((call.owner.clone(), call.callee.clone()))
                .or_default();
            *ordinal += 1;
            SupportCall {
                identity: SupportIdentity {
                    path: path.to_string(),
                    owner: call.owner,
                    callee: call.callee,
                    ordinal: *ordinal,
                },
                line: call.line,
                test_only: call.test_only,
            }
        })
        .collect()
}

fn visibility_is_exposed(visibility: &Visibility) -> bool {
    !matches!(visibility, Visibility::Inherited)
}

#[derive(Default)]
struct RawSignatureTypeVisitor {
    raw_types: BTreeSet<String>,
}

impl<'ast> Visit<'ast> for RawSignatureTypeVisitor {
    fn visit_type_path(&mut self, node: &'ast syn::TypePath) {
        if let Some(segment) = node.path.segments.last() {
            let name = segment.ident.to_string();
            if matches!(
                name.as_str(),
                "Connection"
                    | "Database"
                    | "Transaction"
                    | "Rows"
                    | "Row"
                    | "MutexGuard"
                    | "OwnedMutexGuard"
            ) {
                self.raw_types.insert(name);
            }
        }
        visit::visit_type_path(self, node);
    }
}

fn raw_types_in_signature(signature: &syn::Signature) -> BTreeSet<String> {
    let mut visitor = RawSignatureTypeVisitor::default();
    visitor.visit_signature(signature);
    visitor.raw_types
}

fn visible_raw_signature_violations(source: &str) -> Vec<String> {
    let syntax = match syn::parse_file(source) {
        Ok(syntax) => syntax,
        Err(error) => {
            return vec![format!(
                "test-support source is unparseable at line {}: {error}",
                error.span().start().line
            )];
        }
    };
    let mut violations = Vec::new();
    for item in syntax.items {
        match item {
            Item::Fn(function) if visibility_is_exposed(&function.vis) => {
                let raw = raw_types_in_signature(&function.sig);
                if !raw.is_empty() {
                    violations.push(format!(
                        "visible function {} exposes raw types {raw:?}",
                        function.sig.ident
                    ));
                }
            }
            Item::Impl(implementation) => {
                let self_name = canonical_type_name(&implementation.self_ty);
                if let Some((_, trait_path, _)) = &implementation.trait_ {
                    let trait_name = trait_path
                        .segments
                        .last()
                        .map(|segment| segment.ident.to_string())
                        .unwrap_or_default();
                    if self_name.starts_with("TestDb")
                        && matches!(trait_name.as_str(), "Deref" | "AsRef" | "Borrow")
                    {
                        violations.push(format!(
                            "{self_name} must not implement raw escape trait {trait_name}"
                        ));
                    }
                }
                for impl_item in implementation.items {
                    if let ImplItem::Fn(function) = impl_item {
                        if visibility_is_exposed(&function.vis) {
                            let raw = raw_types_in_signature(&function.sig);
                            if !raw.is_empty() {
                                violations.push(format!(
                                    "visible method {self_name}::{} exposes raw types {raw:?}",
                                    function.sig.ident
                                ));
                            }
                        }
                    }
                }
            }
            Item::Struct(structure) if visibility_is_exposed(&structure.vis) => {
                for field in structure.fields {
                    if visibility_is_exposed(&field.vis) {
                        let mut visitor = RawSignatureTypeVisitor::default();
                        visitor.visit_type(&field.ty);
                        if !visitor.raw_types.is_empty() {
                            violations.push(format!(
                                "visible field on {} exposes raw types {:?}",
                                structure.ident, visitor.raw_types
                            ));
                        }
                    }
                }
            }
            Item::Trait(item_trait) if visibility_is_exposed(&item_trait.vis) => {
                for trait_item in item_trait.items {
                    if let syn::TraitItem::Fn(function) = trait_item {
                        let raw = raw_types_in_signature(&function.sig);
                        if !raw.is_empty() {
                            violations.push(format!(
                                "visible trait {}::{} exposes raw types {raw:?}",
                                item_trait.ident, function.sig.ident
                            ));
                        }
                    }
                }
            }
            Item::Type(item_type) if visibility_is_exposed(&item_type.vis) => {
                let mut visitor = RawSignatureTypeVisitor::default();
                visitor.visit_type(&item_type.ty);
                if !visitor.raw_types.is_empty() {
                    violations.push(format!(
                        "visible type alias {} exposes raw types {:?}",
                        item_type.ident, visitor.raw_types
                    ));
                }
            }
            _ => {}
        }
    }
    violations
}

fn rows_lifetime_shape_violations(source: &str) -> Vec<String> {
    let syntax = match syn::parse_file(source) {
        Ok(syntax) => syntax,
        Err(error) => return vec![format!("test-support source is unparseable: {error}")],
    };
    let Some(rows) = syntax.items.iter().find_map(|item| match item {
        Item::Struct(structure) if structure.ident == "TestDbRows" => Some(structure),
        _ => None,
    }) else {
        return vec!["TestDbRows is missing".to_string()];
    };
    let has_session_lifetime = rows
        .generics
        .lifetimes()
        .any(|lifetime| lifetime.lifetime.ident == "session");
    let has_phantom_session_borrow = rows.fields.iter().any(|field| {
        let Type::Path(phantom) = &field.ty else {
            return false;
        };
        let Some(segment) = phantom.path.segments.last() else {
            return false;
        };
        if segment.ident != "PhantomData" {
            return false;
        }
        let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
            return false;
        };
        arguments.args.iter().any(|argument| {
            let syn::GenericArgument::Type(Type::Reference(reference)) = argument else {
                return false;
            };
            let Some(lifetime) = &reference.lifetime else {
                return false;
            };
            let Type::Path(target) = reference.elem.as_ref() else {
                return false;
            };
            lifetime.ident == "session"
                && target
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "TestDbSession")
        })
    });
    let mut violations = Vec::new();
    if !has_session_lifetime {
        violations.push("TestDbRows must declare the 'session lifetime".to_string());
    }
    if !has_phantom_session_borrow {
        violations.push("TestDbRows must carry PhantomData<&'session TestDbSession>".to_string());
    }
    violations
}

#[derive(Clone, Debug, Default)]
struct StrongCapabilityAliases {
    names: BTreeMap<String, String>,
    modules: BTreeMap<String, String>,
    libsql_glob: bool,
}

impl StrongCapabilityAliases {
    fn from_items(items: &[Item]) -> Self {
        fn bindings(
            tree: &syn::UseTree,
            prefix: &mut Vec<String>,
            output: &mut Vec<(Vec<String>, String)>,
            globs: &mut Vec<Vec<String>>,
        ) {
            match tree {
                syn::UseTree::Path(path) => {
                    prefix.push(path.ident.to_string());
                    bindings(&path.tree, prefix, output, globs);
                    prefix.pop();
                }
                syn::UseTree::Name(name) => {
                    let mut full = prefix.clone();
                    full.push(name.ident.to_string());
                    output.push((full, name.ident.to_string()));
                }
                syn::UseTree::Rename(rename) => {
                    let mut full = prefix.clone();
                    full.push(rename.ident.to_string());
                    output.push((full, rename.rename.to_string()));
                }
                syn::UseTree::Group(group) => {
                    for item in &group.items {
                        bindings(item, prefix, output, globs);
                    }
                }
                syn::UseTree::Glob(_) => globs.push(prefix.clone()),
            }
        }

        let mut aliases = Self::default();
        for item in items {
            let Item::Use(item_use) = item else {
                continue;
            };
            let mut found = Vec::new();
            let mut globs = Vec::new();
            bindings(&item_use.tree, &mut Vec::new(), &mut found, &mut globs);
            for (path, alias) in found {
                let canonical = path.join("::");
                if matches!(
                    canonical.as_str(),
                    "libsql::Database"
                        | "libsql::Connection"
                        | "libsql::Transaction"
                        | "tokio::sync::MutexGuard"
                        | "tokio::sync::OwnedMutexGuard"
                ) {
                    aliases.names.insert(alias, canonical);
                } else if canonical == "libsql" {
                    aliases.modules.insert(alias, canonical);
                }
            }
            aliases.libsql_glob |= globs.iter().any(|path| path.as_slice() == ["libsql"]);
        }
        aliases
    }
}

struct StrongCapabilityVisitor<'aliases> {
    aliases: &'aliases StrongCapabilityAliases,
    capabilities: BTreeSet<String>,
}

impl Visit<'_> for StrongCapabilityVisitor<'_> {
    fn visit_type_path(&mut self, node: &syn::TypePath) {
        let names: Vec<String> = node
            .path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        let canonical = if names.len() == 1 {
            self.aliases.names.get(&names[0]).cloned().or_else(|| {
                (self.aliases.libsql_glob
                    && matches!(names[0].as_str(), "Database" | "Connection" | "Transaction"))
                .then(|| format!("libsql::{}", names[0]))
            })
        } else if names.len() == 2
            && self.aliases.modules.get(&names[0]).map(String::as_str) == Some("libsql")
            && matches!(names[1].as_str(), "Database" | "Connection" | "Transaction")
        {
            Some(format!("libsql::{}", names[1]))
        } else {
            let joined = names.join("::");
            matches!(
                joined.as_str(),
                "libsql::Database"
                    | "libsql::Connection"
                    | "libsql::Transaction"
                    | "tokio::sync::MutexGuard"
                    | "tokio::sync::OwnedMutexGuard"
            )
            .then_some(joined)
        };
        if let Some(capability) = canonical {
            if matches!(
                capability.as_str(),
                "tokio::sync::MutexGuard" | "tokio::sync::OwnedMutexGuard"
            ) {
                let mut target = StrongCapabilityVisitor {
                    aliases: self.aliases,
                    capabilities: BTreeSet::new(),
                };
                for segment in &node.path.segments {
                    target.visit_path_arguments(&segment.arguments);
                }
                if target.capabilities.contains("libsql::Connection") {
                    self.capabilities.insert(capability);
                }
            } else {
                self.capabilities.insert(capability);
            }
        }
        visit::visit_type_path(self, node);
    }
}

fn strong_capabilities_in_type(ty: &Type, aliases: &StrongCapabilityAliases) -> BTreeSet<String> {
    let mut visitor = StrongCapabilityVisitor {
        aliases,
        capabilities: BTreeSet::new(),
    };
    visitor.visit_type(ty);
    visitor.capabilities
}

fn strong_capabilities_in_signature(
    signature: &syn::Signature,
    aliases: &StrongCapabilityAliases,
) -> BTreeSet<String> {
    let mut visitor = StrongCapabilityVisitor {
        aliases,
        capabilities: BTreeSet::new(),
    };
    visitor.visit_signature(signature);
    visitor.capabilities
}

fn visibility_is_crate_or_public(path: &str, visibility: &Visibility) -> bool {
    match visibility {
        Visibility::Public(_) => true,
        Visibility::Restricted(restricted) => {
            restricted.path.is_ident("crate")
                || (path == "crates/wenlan-core/src/db.rs" && restricted.path.is_ident("super"))
        }
        Visibility::Inherited => false,
    }
}

fn db_owned_test_api_violations(
    db_sources: &[(&str, &str, bool)],
    external_sources: &[(&str, &str)],
) -> Vec<String> {
    fn scan_items(
        path: &str,
        items: &[Item],
        _inherited_test: bool,
        violations: &mut Vec<String>,
        exported_functions: &mut BTreeSet<String>,
    ) {
        let aliases = StrongCapabilityAliases::from_items(items);
        for item in items {
            match item {
                Item::Mod(module) => {
                    if let Some((_, items)) = &module.content {
                        scan_items(path, items, false, violations, exported_functions);
                    }
                }
                Item::Fn(function) if visibility_is_crate_or_public(path, &function.vis) => {
                    let raw = strong_capabilities_in_signature(&function.sig, &aliases);
                    if !raw.is_empty() {
                        exported_functions.insert(function.sig.ident.to_string());
                        violations.push(format!(
                            "{path}: visible DB function {} exposes strong capabilities {raw:?}",
                            function.sig.ident
                        ));
                    }
                }
                Item::Struct(structure) => {
                    if !visibility_is_crate_or_public(path, &structure.vis) {
                        continue;
                    }
                    for field in &structure.fields {
                        if !visibility_is_crate_or_public(path, &field.vis) {
                            continue;
                        }
                        let raw = strong_capabilities_in_type(&field.ty, &aliases);
                        if raw.is_empty() {
                            continue;
                        }
                        let field_name = field
                            .ident
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| "<tuple>".to_string());
                        violations.push(format!(
                            "{path}: visible DB field {}::{field_name} exposes strong capabilities {raw:?}",
                            structure.ident
                        ));
                    }
                }
                Item::Enum(enumeration) => {
                    if !visibility_is_crate_or_public(path, &enumeration.vis) {
                        continue;
                    }
                    for variant in &enumeration.variants {
                        for field in &variant.fields {
                            let raw = strong_capabilities_in_type(&field.ty, &aliases);
                            if !raw.is_empty() {
                                violations.push(format!(
                                    "{path}: visible DB enum {}::{} exposes strong capabilities {raw:?}",
                                    enumeration.ident, variant.ident
                                ));
                            }
                        }
                    }
                }
                Item::Trait(item_trait) => {
                    if !visibility_is_crate_or_public(path, &item_trait.vis) {
                        continue;
                    }
                    for trait_item in &item_trait.items {
                        if let syn::TraitItem::Fn(function) = trait_item {
                            let raw = strong_capabilities_in_signature(&function.sig, &aliases);
                            if !raw.is_empty() {
                                violations.push(format!(
                                    "{path}: visible DB trait {}::{} exposes strong capabilities {raw:?}",
                                    item_trait.ident, function.sig.ident
                                ));
                            }
                        }
                    }
                }
                Item::Type(item_type) if visibility_is_crate_or_public(path, &item_type.vis) => {
                    let raw = strong_capabilities_in_type(&item_type.ty, &aliases);
                    if !raw.is_empty() {
                        violations.push(format!(
                            "{path}: visible DB type {} exposes strong capabilities {raw:?}",
                            item_type.ident
                        ));
                    }
                }
                Item::Impl(implementation) => {
                    let self_name = canonical_type_name(&implementation.self_ty);
                    if let Some((_, trait_path, _)) = &implementation.trait_ {
                        let trait_name = trait_path
                            .segments
                            .last()
                            .map(|segment| segment.ident.to_string())
                            .unwrap_or_default();
                        if matches!(
                            trait_name.as_str(),
                            "Into" | "From" | "Deref" | "AsRef" | "Borrow"
                        ) {
                            let mut raw =
                                strong_capabilities_in_type(&implementation.self_ty, &aliases);
                            let mut trait_visitor = StrongCapabilityVisitor {
                                aliases: &aliases,
                                capabilities: BTreeSet::new(),
                            };
                            trait_visitor.visit_path(trait_path);
                            raw.extend(trait_visitor.capabilities);
                            for impl_item in &implementation.items {
                                match impl_item {
                                    ImplItem::Type(item_type) => {
                                        raw.extend(strong_capabilities_in_type(
                                            &item_type.ty,
                                            &aliases,
                                        ));
                                    }
                                    ImplItem::Fn(function) => {
                                        raw.extend(strong_capabilities_in_signature(
                                            &function.sig,
                                            &aliases,
                                        ));
                                    }
                                    _ => {}
                                }
                            }
                            if !raw.is_empty() {
                                violations.push(format!(
                                    "{path}: DB type {self_name} implements strong-capability escape trait {trait_name}"
                                ));
                            }
                        }
                    }
                    for impl_item in &implementation.items {
                        if let ImplItem::Fn(function) = impl_item {
                            if visibility_is_crate_or_public(path, &function.vis) {
                                let raw = strong_capabilities_in_signature(&function.sig, &aliases);
                                if !raw.is_empty() {
                                    violations.push(format!(
                                        "{path}: visible DB method {self_name}::{} exposes strong capabilities {raw:?}",
                                        function.sig.ident
                                    ));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let mut violations = Vec::new();
    let mut exported_functions = BTreeSet::new();
    for (path, source, inherited_test) in db_sources {
        match syn::parse_file(source) {
            Ok(syntax) => scan_items(
                path,
                &syntax.items,
                *inherited_test,
                &mut violations,
                &mut exported_functions,
            ),
            Err(error) => violations.push(format!(
                "{path}: cannot parse DB-owned source at line {}: {error}",
                error.span().start().line
            )),
        }
    }

    struct ExternalCallVisitor<'names> {
        names: &'names BTreeSet<String>,
        calls: BTreeSet<String>,
    }
    impl<'ast> Visit<'ast> for ExternalCallVisitor<'_> {
        fn visit_expr_call(&mut self, node: &'ast ExprCall) {
            if let Expr::Path(path) = node.func.as_ref() {
                if let Some(last) = path.path.segments.last() {
                    let name = last.ident.to_string();
                    if self.names.contains(&name) {
                        self.calls.insert(name);
                    }
                }
            }
            visit::visit_expr_call(self, node);
        }
    }
    for (path, source) in external_sources {
        let syntax = match syn::parse_file(source) {
            Ok(syntax) => syntax,
            Err(error) => {
                violations.push(format!(
                    "{path}: cannot parse external source at line {}: {error}",
                    error.span().start().line
                ));
                continue;
            }
        };
        let mut visitor = ExternalCallVisitor {
            names: &exported_functions,
            calls: BTreeSet::new(),
        };
        visitor.visit_file(&syntax);
        for call in visitor.calls {
            violations.push(format!(
                "{path}: external caller reaches raw DB test API {call}"
            ));
        }
    }
    violations
}

#[test]
fn synthetic_parser_classifies_ordinary_multiline_and_macro_raw_shapes() {
    let source = r#"
async fn ordinary(db: &MemoryDB) {
    let _guard = db.conn.lock().await;
    let _secondary = db._db.connect();
}

fn multiline(db: &MemoryDB) {
    let _guard = db
        .conn
        .try_lock();
    retain(db.conn.clone());
}

macro_rules! raw {
    ($db:expr) => {
        $db.conn.lock().await;
        libsql::Builder::new_local("isolated.db");
    };
}
"#;

    let analysis = analyze_source_for_test(source, "crate::synthetic");
    assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    let shapes: Vec<RawShape> = analysis
        .raw_uses
        .iter()
        .map(|raw_use| raw_use.identity.shape)
        .collect();

    assert_eq!(
        shapes,
        vec![
            RawShape::PrimaryConnLock,
            RawShape::AlternateDbField,
            RawShape::PrimaryConnTryLock,
            RawShape::ConnFieldEscape,
            RawShape::PrimaryConnLock,
            RawShape::StandaloneLibsqlOrigin,
        ]
    );
}

#[test]
fn synthetic_parser_ignores_strings_and_comments() {
    let source = r#"
fn clean() {
    let _text = "db.conn.lock().await libsql::Builder::new_local";
    // db._db.connect();
    /* db.conn.try_lock(); */
}
"#;

    assert!(analyze_source_for_test(source, "crate::clean")
        .raw_uses
        .is_empty());
}

#[test]
fn synthetic_parser_fails_closed_on_unparseable_source() {
    let analysis = analyze_source_for_test("fn broken( {", "crate::broken");
    assert_eq!(analysis.errors.len(), 1);
    assert!(analysis.errors[0].contains("synthetic.rs"));
}

#[test]
fn synthetic_parser_handles_borrow_deref_clone_and_destructure_shapes() {
    let analysis = analyze_source_for_test(
        r#"
fn variations(db: &MemoryDB) {
    let _ = (&*db.conn).lock().await;
    retain(&db.conn);
    retain(db.conn.clone());
    let MemoryDB { conn, _db, .. } = db;
}
"#,
        "crate::variations",
    );
    assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    let shapes: Vec<RawShape> = analysis
        .raw_uses
        .iter()
        .map(|raw_use| raw_use.identity.shape)
        .collect();
    assert_eq!(
        shapes,
        vec![
            RawShape::PrimaryConnLock,
            RawShape::ConnFieldEscape,
            RawShape::ConnFieldEscape,
            RawShape::ConnFieldEscape,
            RawShape::AlternateDbField,
        ]
    );
}

#[test]
fn synthetic_parser_tracks_every_libsql_builder_new_constructor() {
    let analysis = analyze_source_for_test(
        r#"
fn remote() {
    let _ = libsql::Builder::new_remote(url, token);
}

macro_rules! replica {
    () => {
        libsql::Builder::new_remote_replica(path, url, token);
    };
}
"#,
        "crate::standalone",
    );
    assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    assert_eq!(
        analysis
            .raw_uses
            .iter()
            .filter(|raw_use| { raw_use.identity.shape == RawShape::StandaloneLibsqlOrigin })
            .count(),
        2
    );
}

#[test]
fn synthetic_parser_tracks_imported_and_aliased_libsql_builders() {
    let analysis = analyze_source_for_test(
        r#"
use libsql::Builder;
use libsql::Builder as LocalBuilder;

fn imported() {
    let _ = Builder::new_local("one.db");
    let _ = LocalBuilder::new_remote(url, token);
}
"#,
        "crate::standalone_imports",
    );
    assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    assert_eq!(
        analysis
            .raw_uses
            .iter()
            .filter(|raw_use| { raw_use.identity.shape == RawShape::StandaloneLibsqlOrigin })
            .count(),
        2
    );
}

#[test]
fn synthetic_macro_parser_tracks_memorydb_destructure() {
    let analysis = analyze_source_for_test(
        r#"
macro_rules! expose {
    ($db:expr) => {
        let MemoryDB { conn, .. } = $db;
    };
}
"#,
        "crate::macro_destructure",
    );
    assert_eq!(
        analysis
            .raw_uses
            .iter()
            .filter(|raw_use| raw_use.identity.shape == RawShape::ConnFieldEscape)
            .count(),
        1
    );
}

#[test]
fn raw_manifest_rejects_empty_wrong_and_same_count_moved_rows() {
    let actual = vec![RawUse {
        identity: RawIdentity {
            path: "src/one.rs".to_string(),
            owner: "crate::one::test_body".to_string(),
            shape: RawShape::PrimaryConnLock,
            ordinal: 1,
        },
        line: 10,
        test_only: true,
    }];
    let exact = "src/one.rs|crate::one::test_body|PrimaryConnLock|1\n";
    assert!(compare_raw_manifest(&actual, exact).is_ok());

    for wrong_manifest in [
        "",
        "src/one.rs|crate::one::test_body|PrimaryConnTryLock|1",
        "src/two.rs|crate::two::test_body|PrimaryConnLock|1",
        concat!(
            "src/one.rs|crate::one::test_body|PrimaryConnLock|1\n",
            "src/two.rs|crate::two::test_body|StandaloneLibsqlOrigin|1\n",
        ),
    ] {
        assert!(
            compare_raw_manifest(&actual, wrong_manifest).is_err(),
            "manifest drift must fail closed: {wrong_manifest:?}"
        );
    }
}

#[test]
fn raw_manifest_rejects_duplicate_rows() {
    let duplicate = concat!(
        "src/one.rs|crate::one::test_body|PrimaryConnLock|1\n",
        "src/one.rs|crate::one::test_body|PrimaryConnLock|1\n",
    );
    assert!(
        compare_raw_manifest(&[], duplicate).is_err(),
        "duplicate rows must not collapse into a set"
    );
}

fn support_analysis_in_source(source: &str, test_only: bool) -> Analysis {
    let syntax = syn::parse_file(source).expect("synthetic support source parses");
    let mut pending = Vec::new();
    let mut errors = Vec::new();
    for item in syntax.items {
        if let Item::Fn(function) = item {
            scan_support_block(
                &function.sig,
                &function.block,
                &format!("crate::synthetic::{}", function.sig.ident),
                test_only,
                &mut pending,
                &mut errors,
            );
        }
    }
    Analysis {
        support_calls: finalize_support_calls("synthetic.rs", pending),
        errors,
        ..Analysis::default()
    }
}

fn support_calls_in_source(source: &str, test_only: bool) -> Vec<SupportCall> {
    let analysis = support_analysis_in_source(source, test_only);
    assert!(
        analysis.errors.is_empty(),
        "synthetic support source must classify exactly: {:?}",
        analysis.errors
    );
    analysis.support_calls
}

#[test]
fn support_parser_tracks_resolved_local_flow_without_counting_unrelated_query_methods() {
    let source = r#"
async fn exercise(db: &MemoryDB, typed: TestDbSession) {
    unrelated.execute("not support");
    let session = db.test_primary_session().await;
    session.execute("SELECT 1", ()).await.unwrap();
    let mut rows = session.query("SELECT 1", ()).await.unwrap();
    let row = rows.next().await.unwrap().unwrap();
    let _: i64 = row.get(0).unwrap();
    typed.structural_digest().await.unwrap();
    support_macro!(db.test_secondary_session());
}
"#;
    let calls = support_calls_in_source(source, true);
    let callees: Vec<&str> = calls
        .iter()
        .map(|call| call.identity.callee.as_str())
        .collect();
    assert_eq!(
        callees,
        vec![
            "test_primary_session",
            "TestDbSession::execute",
            "TestDbSession::query",
            "TestDbRows::next",
            "TestDbRow::get",
            "TestDbSession::structural_digest",
            "test_secondary_session",
        ]
    );
}

#[test]
fn support_parser_tracks_while_let_row_flow_without_counting_unrelated_get() {
    let calls = support_calls_in_source(
        r#"
async fn chained(session: TestDbSession) {
    let mut rows = session.query("SELECT 1", ()).await.unwrap();
    while let Some(direct_row) = rows.next().await.unwrap() {
        let _ = direct_row.get::<i64>(0).unwrap();
    }
    while let Some(multiline_row) = rows
        .next()
        .await
        .expect("step")
    {
        let _ = multiline_row
            .get::<i64>(0)
            .expect("value");
    }
    while let Some(unrelated_row) = unrelated.next().await.unwrap() {
        let _ = unrelated_row.get::<i64>(0).unwrap();
    }
}
"#,
        true,
    );
    let callees: Vec<&str> = calls
        .iter()
        .map(|call| call.identity.callee.as_str())
        .collect();
    assert_eq!(
        callees,
        vec![
            "TestDbSession::query",
            "TestDbRows::next",
            "TestDbRow::get",
            "TestDbRows::next",
            "TestDbRow::get",
        ]
    );
}

#[test]
fn macro_chained_factory_call_is_exact_and_trips_manifest_drift() {
    let calls = support_calls_in_source(
        r#"
fn macro_owner() {
    chained!(
        db.test_secondary_session()
            .unwrap()
            .query("SELECT 1", ())
    );
}
"#,
        true,
    );
    let callees: Vec<&str> = calls
        .iter()
        .map(|call| call.identity.callee.as_str())
        .collect();
    assert_eq!(
        callees,
        vec!["test_secondary_session", "TestDbSession::query"]
    );
    assert!(
        compare_support_manifest(&calls, "").is_err(),
        "a macro-contained support call must change the exact manifest"
    );
}

#[test]
fn macro_support_parser_handles_ufcs_and_wrapped_local_receivers() {
    let calls = support_calls_in_source(
        r#"
async fn macro_owner(session: TestDbSession) {
    support_macro!(
        TestDbSession::execute(&session, "SELECT 1", ());
        (session).query("SELECT 1", ());
        (&*session).structural_digest();
    );
}
"#,
        true,
    );
    let callees: Vec<&str> = calls
        .iter()
        .map(|call| call.identity.callee.as_str())
        .collect();
    assert_eq!(
        callees,
        vec![
            "TestDbSession::execute",
            "TestDbSession::query",
            "TestDbSession::structural_digest",
        ]
    );
}

#[test]
fn macro_target_bearing_forms_are_exact_or_fail_closed() {
    let classified = analyze_source_for_test(
        r#"
use libsql::Builder as B;
macro_rules! build {
    () => { B::new_local("classified.db") };
}
"#,
        "crate::macro_classified",
    );
    assert!(classified.errors.is_empty(), "{:?}", classified.errors);
    assert_eq!(
        classified
            .raw_uses
            .iter()
            .filter(|raw_use| raw_use.identity.shape == RawShape::StandaloneLibsqlOrigin)
            .count(),
        1
    );

    let rejected = analyze_source_for_test(
        r#"
use libsql::Builder as B;
macro_rules! ambiguous {
    ($session:expr) => {
        TestDbSession::mystery($session);
        ($session).unknown_support_operation();
        B::from_custom_source("ambiguous.db");
    };
}
"#,
        "crate::macro_rejected",
    );
    for target in [
        "TestDbSession::mystery",
        "unknown_support_operation",
        "B::from_custom_source",
    ] {
        assert!(
            rejected.errors.iter().any(|error| error.contains(target)),
            "unclassified macro target {target} did not fail closed: {:?}",
            rejected.errors
        );
    }

    let typed_local = support_analysis_in_source(
        r#"
fn typed(session: TestDbSession) {
    ambiguous!((session).unknown_typed_operation());
}
"#,
        true,
    );
    assert!(
        typed_local
            .errors
            .iter()
            .any(|error| error.contains("unknown_typed_operation")),
        "typed macro receiver target did not fail closed: {:?}",
        typed_local.errors
    );
}

#[test]
fn module_macro_definitions_record_ufcs_or_reject_unproven_metavariables() {
    let temp = tempfile::tempdir().expect("module macro source graph");
    let source_root = temp.path().join("crates/wenlan-core/src");
    std::fs::create_dir_all(&source_root).expect("create module macro source root");
    std::fs::write(
        source_root.join("lib.rs"),
        r#"
macro_rules! exact {
    ($s:expr) => {
        TestDbSession::execute($s, "SELECT 1", ());
    };
}

macro_rules! ambiguous {
    ($s:expr) => {
        ($s).query("SELECT 1", ());
        $s.unknown_operation();
    };
}
"#,
    )
    .expect("write module macro root");
    let analysis = analyze_repository(temp.path());
    assert!(
        analysis.support_calls.iter().any(|call| {
            call.identity.owner == "crate::<module>"
                && call.identity.callee == "TestDbSession::execute"
        }),
        "module macro UFCS support target was not recorded: {:?}",
        analysis.support_calls
    );
    for target in ["query", "unknown_operation"] {
        assert!(
            analysis.errors.iter().any(|error| error.contains(target)),
            "unproven metavariable target {target} did not fail closed: {:?}",
            analysis.errors
        );
    }
}

#[test]
fn comma_separated_macro_invocation_resolves_wrapped_known_locals() {
    let analysis = support_analysis_in_source(
        r#"
fn comma_separated(session: TestDbSession) {
    support_macro!(
        (session).query("SELECT 1", ()),
        (&*session).structural_digest()
    );
}
"#,
        true,
    );
    assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    let callees: Vec<&str> = analysis
        .support_calls
        .iter()
        .map(|call| call.identity.callee.as_str())
        .collect();
    assert_eq!(
        callees,
        vec!["TestDbSession::query", "TestDbSession::structural_digest"]
    );
}

#[test]
fn function_local_builder_alias_is_a_standalone_origin() {
    let analysis = analyze_source_for_test(
        r#"
fn local_builder() {
    use libsql::Builder as B;
    let _ = B::new_local("local.db");
}
"#,
        "crate::local_builder",
    );
    assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    assert_eq!(
        analysis
            .raw_uses
            .iter()
            .filter(|raw_use| raw_use.identity.shape == RawShape::StandaloneLibsqlOrigin)
            .count(),
        1
    );
}

#[test]
fn support_parser_marks_resolved_production_calls() {
    let calls = support_calls_in_source(
        "async fn production(db: &MemoryDB) { let _ = db.test_primary_session().await; }",
        false,
    );
    assert_eq!(calls.len(), 1);
    assert!(!calls[0].test_only);
}

#[test]
fn support_parser_propagates_receiver_renames() {
    let calls = support_calls_in_source(
        r#"
async fn renamed(db: &MemoryDB) {
    let original = db.test_primary_session().await;
    let renamed = original;
    renamed.execute("SELECT 1", ()).await.unwrap();
}
"#,
        true,
    );
    let callees: Vec<&str> = calls
        .iter()
        .map(|call| call.identity.callee.as_str())
        .collect();
    assert_eq!(
        callees,
        vec!["test_primary_session", "TestDbSession::execute"]
    );
}

#[test]
fn support_parser_tracks_reference_parameters_and_ufcs_calls() {
    let calls = support_calls_in_source(
        r#"
async fn referenced(
    session: &TestDbSession,
    mutable: &mut TestDbSession,
) {
    session.execute("SELECT 1", ()).await.unwrap();
    mutable.query("SELECT 1", ()).await.unwrap();
    TestDbSession::execute(session, "SELECT 1", ()).await.unwrap();
    TestDbSession::query(mutable, "SELECT 1", ()).await.unwrap();
}
"#,
        true,
    );
    let callees: Vec<&str> = calls
        .iter()
        .map(|call| call.identity.callee.as_str())
        .collect();
    assert_eq!(
        callees,
        vec![
            "TestDbSession::execute",
            "TestDbSession::query",
            "TestDbSession::execute",
            "TestDbSession::query",
        ]
    );
}

#[test]
fn support_parser_propagates_consuming_transaction_results() {
    let calls = support_calls_in_source(
        r#"
async fn restored(session: TestDbSession) {
    let after_rollback = session.rollback().await.unwrap();
    after_rollback.execute("SELECT 1", ()).await.unwrap();
    let after_commit = TestDbSession::commit(after_rollback).await.unwrap();
    after_commit.query("SELECT 1", ()).await.unwrap();
}
"#,
        true,
    );
    let callees: Vec<&str> = calls
        .iter()
        .map(|call| call.identity.callee.as_str())
        .collect();
    assert_eq!(
        callees,
        vec![
            "TestDbSession::rollback",
            "TestDbSession::execute",
            "TestDbSession::commit",
            "TestDbSession::query",
        ]
    );
}

#[test]
fn trait_default_methods_receive_exact_owners_and_candidate_scanning() {
    let temp = tempfile::tempdir().expect("trait source graph");
    let source_root = temp.path().join("crates/wenlan-core/src");
    std::fs::create_dir_all(&source_root).expect("create trait source root");
    std::fs::write(
        source_root.join("lib.rs"),
        r#"
#[cfg(test)]
trait Probe {
    async fn default_probe(&self, db: &MemoryDB, session: &TestDbSession) {
        let _ = db.conn.lock().await;
        session.execute("SELECT 1", ()).await.unwrap();
    }
}
"#,
    )
    .expect("write trait root");
    let analysis = analyze_repository(temp.path());
    assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    assert!(analysis.raw_uses.iter().any(|raw_use| {
        raw_use.identity.owner == "crate::Probe::default_probe"
            && raw_use.identity.shape == RawShape::PrimaryConnLock
    }));
    assert!(analysis.support_calls.iter().any(|call| {
        call.identity.owner == "crate::Probe::default_probe"
            && call.identity.callee == "TestDbSession::execute"
    }));
}

#[test]
fn nested_function_candidates_fail_closed_instead_of_inheriting_outer_owner() {
    let temp = tempfile::tempdir().expect("nested item graph");
    let source_root = temp.path().join("crates/wenlan-core/src");
    std::fs::create_dir_all(&source_root).expect("create nested source root");
    std::fs::write(
        source_root.join("lib.rs"),
        r#"
#[test]
fn outer() {
    async fn nested(db: &MemoryDB, session: &TestDbSession) {
        let _ = db.conn.lock().await;
        session.execute("SELECT 1", ()).await.unwrap();
    }
}
"#,
    )
    .expect("write nested root");
    let analysis = analyze_repository(temp.path());
    assert!(
        analysis
            .errors
            .iter()
            .any(|error| error.contains("nested function") && error.contains("nested")),
        "nested candidate must fail closed with its own identity: {:?}",
        analysis.errors
    );
}

#[test]
fn repository_module_graph_matches_r4_25_group_6_census() {
    let analysis = analyze_repository(&super::repo_root());
    assert!(
        analysis.errors.is_empty(),
        "R4 source graph must resolve and parse completely: {:#?}",
        analysis.errors
    );

    let mut counts = BTreeMap::<RawShape, usize>::new();
    for raw_use in &analysis.raw_uses {
        *counts.entry(raw_use.identity.shape).or_default() += 1;
    }
    assert_eq!(
        counts.get(&RawShape::PrimaryConnLock).copied().unwrap_or(0),
        0
    );
    assert_eq!(
        counts
            .get(&RawShape::PrimaryConnTryLock)
            .copied()
            .unwrap_or(0),
        0
    );
    assert_eq!(
        counts
            .get(&RawShape::AlternateDbField)
            .copied()
            .unwrap_or(0),
        0
    );
    assert_eq!(
        counts.get(&RawShape::ConnFieldEscape).copied().unwrap_or(0),
        0
    );
    assert_eq!(
        counts
            .get(&RawShape::StandaloneLibsqlOrigin)
            .copied()
            .unwrap_or(0),
        17
    );
    assert!(
        analysis.raw_uses.iter().all(|raw_use| raw_use.test_only),
        "production raw DB access is forbidden: {:#?}",
        analysis
            .raw_uses
            .iter()
            .filter(|raw_use| !raw_use.test_only)
            .collect::<Vec<_>>()
    );
    assert!(
        analysis.support_calls.iter().all(|call| call.test_only),
        "resolved test-support calls are forbidden in production owners: {:#?}",
        analysis
            .support_calls
            .iter()
            .filter(|call| !call.test_only)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        analysis.support_calls.len(),
        1020,
        "PR-D integration must expose the frozen 967 support calls, the 6 PR-D test identities, \
         the 5 M5 derivation-marker fixture calls, the 10 M6 shadow-promoter fixture calls, \
         the 7 G6 BindPageLink repair-test calls (G6 Stage 2 PR 2b, item 3: \
         bind_page_link_repair_mints_the_links_edge dropped its second TestDbRow::get call when \
         the resolved-row assertion was replaced with an orphan-row-absence check), the 9 G6 \
         Stage 1.5a raw_seeded_entity_without_shadow_page_deletes_via_applier_shadow_page_guard \
         calls, the 1 G6 Stage 1.5b uncategorized_scope_selects_the_unfiled_sentinel_entity call, \
         the 1 G6 Stage 1.5b Part 3 second test_primary_session re-acquire in \
         summary_eligibility_requires_a_qualifying_community_and_candidate (drop/reseed/reacquire \
         around test_seed_entity_shadow_page's internal conn lock), the 3 G6 edges-parity \
         repair RED-controlled test calls in \
         write_document_source_page_replace_keeps_carried_over_retires_dropped, and the net +17 \
         G6 Stage 2 PR 2b sweep instance 4 calls (fold_relation_type discovery-scan port re-seeded \
         onto edges: fold_relation_type_merges_provenance_and_ledgers_the_loser and \
         fold_relation_type_rolls_back_when_ledger_insert_fails switched their raw-INSERT seeding \
         to create_relation, each dropping its TestDbSession::execute pair (rolls_back also gains \
         one TestDbRow::get from its edges-shaped readback); \
         heal_relation_vocabulary_folds_aliases_and_queues_semantics and \
         test_run_rethink_normalizes_relation_types re-seed a non-canonical type via raw \
         INSERT INTO edges, each gaining a query/next pair (test_run_rethink_normalizes_relation_types \
         also gains a TestDbRow::get); and the new acceptance-pin test \
         heal_relation_vocabulary_discovers_and_folds_edges_collision_keeps_stronger contributes \
         15 calls); minus the net -6 G6 Stage 2 PR 2c item 4 calls (entity alias storage moves \
         off the raw entity_aliases table onto resolve_entity_by_alias's shadow-page scan: \
         create_entity_minhash_disabled_is_noop and create_entity_minhash_short_name_skips_fuzzy \
         each drop their second TestDbRow::get, TestDbRows::next, and TestDbSession::query call, \
         losing the raw alias-row readback the old exact-match SQL needed)"
    );
}

#[test]
fn module_graph_resolves_path_and_inline_modules_with_exact_test_ancestry() {
    let temp = tempfile::tempdir().expect("temp module graph");
    let source_root = temp.path().join("crates/wenlan-core/src");
    std::fs::create_dir_all(&source_root).expect("create synthetic source root");
    std::fs::write(
        source_root.join("lib.rs"),
        r#"
#[cfg(test)]
mod external;

#[path = "alternate.rs"]
mod renamed;

mod outer;

mod inline {
    mod child;
    #[path = "renamed.rs"]
    mod alternate_child;

    #[tokio::test]
    async fn test_body(db: &MemoryDB) {
        let _ = db.conn.lock().await;
    }
}

#[cfg(any(test, feature = "synthetic"))]
mod not_exact {
    async fn production_owner(db: &MemoryDB) {
        let _ = db.conn.lock().await;
    }
}
"#,
    )
    .expect("write lib root");
    std::fs::write(
        source_root.join("external.rs"),
        "async fn inherited(db: &MemoryDB) { let _ = db._db.connect(); }",
    )
    .expect("write external module");
    std::fs::write(
        source_root.join("alternate.rs"),
        "#[test] fn direct(db: &MemoryDB) { let _ = db.conn.try_lock(); }",
    )
    .expect("write path module");
    std::fs::write(
        source_root.join("outer.rs"),
        "#[path = \"outer_test.rs\"] mod tests;",
    )
    .expect("write non-inline external module");
    std::fs::write(
        source_root.join("outer_test.rs"),
        "#[test] fn external_path(db: &MemoryDB) { let _ = db.conn.try_lock(); }",
    )
    .expect("write non-inline path child");
    std::fs::create_dir_all(source_root.join("inline")).expect("create inline module directory");
    std::fs::write(
        source_root.join("inline/child.rs"),
        "#[test] fn nested(db: &MemoryDB) { let _ = db._db.connect(); }",
    )
    .expect("write inline external child");
    std::fs::write(
        source_root.join("inline/renamed.rs"),
        "#[test] fn nested_path(db: &MemoryDB) { let _ = db.conn.try_lock(); }",
    )
    .expect("write inline path child");

    let analysis = analyze_repository(temp.path());
    assert!(analysis.errors.is_empty(), "{:?}", analysis.errors);
    assert_eq!(analysis.raw_uses.len(), 7);
    let production: Vec<&RawUse> = analysis
        .raw_uses
        .iter()
        .filter(|raw_use| !raw_use.test_only)
        .collect();
    assert_eq!(production.len(), 1);
    assert_eq!(
        production[0].identity.owner,
        "crate::not_exact::production_owner"
    );
    assert!(analysis
        .raw_uses
        .iter()
        .any(|raw_use| raw_use.identity.owner == "crate::external::inherited"));
    assert!(analysis
        .raw_uses
        .iter()
        .any(|raw_use| raw_use.identity.owner == "crate::renamed::direct"));
    assert!(analysis.raw_uses.iter().any(|raw_use| {
        raw_use.identity.owner == "crate::inline::child::nested"
            && raw_use.identity.path.ends_with("src/inline/child.rs")
    }));
    assert!(analysis.raw_uses.iter().any(|raw_use| {
        raw_use.identity.owner == "crate::inline::alternate_child::nested_path"
            && raw_use.identity.path.ends_with("src/inline/renamed.rs")
    }));
    assert!(analysis.raw_uses.iter().any(|raw_use| {
        raw_use.identity.owner == "crate::outer::tests::external_path"
            && raw_use.identity.path.ends_with("src/outer_test.rs")
    }));

    std::fs::write(source_root.join("lib.rs"), "mod missing;").expect("write unresolved root");
    let unresolved = analyze_repository(temp.path());
    assert!(
        unresolved
            .errors
            .iter()
            .any(|error| error.contains("unresolved module missing")),
        "{:?}",
        unresolved.errors
    );
}

#[test]
fn tracked_orphan_rust_file_fails_closed() {
    let temp = tempfile::tempdir().expect("orphan candidate root");
    let source_root = temp.path().join("crates/wenlan-core/src");
    std::fs::create_dir_all(&source_root).expect("create orphan source root");
    std::fs::write(source_root.join("lib.rs"), "").expect("write visited root");
    std::fs::write(
        source_root.join("orphan_test.rs"),
        "async fn orphan(db: &MemoryDB) { let _ = db.conn.lock().await; }",
    )
    .expect("write raw orphan");
    std::fs::write(
        source_root.join("clean_orphan.rs"),
        "fn clean_orphan() -> i64 { 1 }",
    )
    .expect("write clean orphan");
    let visited = BTreeSet::from(["crates/wenlan-core/src/lib.rs".to_string()]);
    let candidates = vec![
        "crates/wenlan-core/src/lib.rs".to_string(),
        "crates/wenlan-core/src/orphan_test.rs".to_string(),
        "crates/wenlan-core/src/clean_orphan.rs".to_string(),
    ];
    assert_eq!(
        unclassified_rust_file_errors(temp.path(), &candidates, &visited),
        vec![
            "crates/wenlan-core/src/orphan_test.rs: tracked Rust candidate is outside the classified module graph"
                .to_string(),
            "crates/wenlan-core/src/clean_orphan.rs: tracked Rust candidate is outside the classified module graph"
                .to_string(),
        ]
    );
}

#[test]
fn visible_raw_signature_and_escape_traits_fail_closed() {
    let violations = visible_raw_signature_violations(
        r#"
pub(crate) struct TestDbSession {
    private_connection: libsql::Connection,
}

impl TestDbSession {
    fn private_connection(&self) -> &libsql::Connection {
        &self.private_connection
    }

    pub(crate) fn exposed(&self, row: libsql::Row) -> libsql::Rows {
        todo!()
    }

    pub(crate) fn callback(
        &self,
        callback: impl FnOnce(&libsql::Connection),
    ) {
        todo!()
    }
}

impl std::ops::Deref for TestDbSession {
    type Target = libsql::Connection;
    fn deref(&self) -> &Self::Target { todo!() }
}

impl AsRef<libsql::Connection> for TestDbSession {
    fn as_ref(&self) -> &libsql::Connection { todo!() }
}

impl std::borrow::Borrow<libsql::Connection> for TestDbSession {
    fn borrow(&self) -> &libsql::Connection { todo!() }
}
"#,
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("exposed") && violation.contains("Row")),
        "{violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("raw escape trait Deref")),
        "{violations:?}"
    );
    for escape in [
        "callback",
        "raw escape trait AsRef",
        "raw escape trait Borrow",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(escape)),
            "positive control did not reject {escape}: {violations:?}"
        );
    }
    assert!(
        !violations
            .iter()
            .any(|violation| violation.contains("private_connection")),
        "private state and private methods remain allowed: {violations:?}"
    );
}

#[test]
fn db_owned_test_api_guard_rejects_raw_fields_exports_and_external_callers() {
    let db_source = r#"
pub(crate) struct MemoryDB {
    pub(crate) _db: libsql::Database,
    pub(crate) conn: Arc<tokio::sync::Mutex<libsql::Connection>>,
}

#[cfg(test)]
pub(crate) fn leak() -> Arc<tokio::sync::Mutex<libsql::Connection>> {
    todo!()
}

#[cfg(test)]
pub(crate) enum TestEscape {
    Connection(libsql::Connection),
}

#[cfg(test)]
pub(crate) struct TestWrapper(libsql::Connection);

#[cfg(test)]
impl Into<libsql::Connection> for TestWrapper {
    fn into(self) -> libsql::Connection { self.0 }
}

#[cfg(test)]
impl From<TestWrapper> for libsql::Connection {
    fn from(value: TestWrapper) -> Self { value.0 }
}

#[cfg(test)]
impl std::ops::Deref for TestWrapper {
    type Target = libsql::Connection;
    fn deref(&self) -> &Self::Target { &self.0 }
}

#[cfg(test)]
impl AsRef<libsql::Connection> for TestWrapper {
    fn as_ref(&self) -> &libsql::Connection { &self.0 }
}

#[cfg(test)]
impl std::borrow::Borrow<libsql::Connection> for TestWrapper {
    fn borrow(&self) -> &libsql::Connection { &self.0 }
}
"#;
    let external = r#"
#[test]
fn external_caller() {
    let _ = crate::db::leak();
}
"#;
    let violations = db_owned_test_api_violations(
        &[("crates/wenlan-core/src/db.rs", db_source, false)],
        &[("crates/wenlan-core/src/external_test.rs", external)],
    );
    for expected in [
        "MemoryDB::_db",
        "MemoryDB::conn",
        "leak",
        "TestEscape",
        "Into",
        "From",
        "Deref",
        "AsRef",
        "Borrow",
        "external caller",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "DB child guard did not reject {expected}: {violations:?}"
        );
    }
}

#[test]
fn db_owned_api_guard_resolves_production_capability_aliases_exactly() {
    let db_source = r#"
use libsql::{
    Connection as DbConn,
    Database,
    Row as SqlRow,
    Transaction,
};
use tokio::sync::OwnedMutexGuard as DbGuard;

mod domain {
    pub struct Connection;
}

pub(crate) fn direct_leak() -> libsql::Connection {
    todo!()
}

pub fn aliased_leak(connection: DbConn) -> Database {
    todo!()
}

pub(super) fn db_internal(connection: DbConn) -> Transaction {
    todo!()
}

pub(crate) fn unrelated(connection: domain::Connection) -> domain::Connection {
    connection
}

pub(crate) fn row_only(row: SqlRow) -> SqlRow {
    row
}

pub(crate) struct ProductionEscape {
    pub(crate) guard: DbGuard<DbConn>,
}

pub(crate) enum ProductionVariant {
    Transaction(Transaction),
}

pub(crate) struct Wrapper(DbConn);

impl Into<DbConn> for Wrapper {
    fn into(self) -> DbConn { self.0 }
}
"#;
    let external = r#"
fn caller() {
    let _ = crate::db::direct_leak();
    let _ = crate::db::aliased_leak(todo!());
}
"#;
    let violations = db_owned_test_api_violations(
        &[("crates/wenlan-core/src/db/production.rs", db_source, false)],
        &[("crates/wenlan-core/src/caller.rs", external)],
    );
    for expected in [
        "direct_leak",
        "aliased_leak",
        "ProductionEscape",
        "ProductionVariant",
        "Into",
        "external caller",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "production DB capability leak {expected} was not rejected: {violations:?}"
        );
    }
    for allowed in ["db_internal", "unrelated", "row_only"] {
        assert!(
            !violations
                .iter()
                .any(|violation| violation.contains(allowed)),
            "legitimate non-capability surface {allowed} was rejected: {violations:?}"
        );
    }
}

#[test]
fn db_visibility_and_capability_imports_are_path_aware() {
    let root_source = r#"
pub(super) fn root_super() -> libsql::Connection { todo!() }
pub(in crate) fn root_crate() -> libsql::Transaction { todo!() }
pub(in crate::db) fn root_db_internal() -> libsql::Database { todo!() }
"#;
    let child_source = r#"
pub(super) fn child_internal(connection: libsql::Connection) { todo!() }
pub(in crate) fn child_crate() -> libsql::Database { todo!() }
pub(in crate::db) fn child_db_internal() -> libsql::Transaction { todo!() }
"#;
    let module_alias = r#"
use libsql as sql;
pub(crate) fn module_alias() -> sql::Connection { todo!() }
"#;
    let glob_alias = r#"
use libsql::*;
pub(crate) fn glob_alias() -> Connection { todo!() }
"#;
    let unrelated = r#"
mod domain { pub struct Connection; }
use domain::Connection;
pub(crate) fn unrelated(value: Connection) -> Connection { value }
"#;
    let violations = db_owned_test_api_violations(
        &[
            ("crates/wenlan-core/src/db.rs", root_source, false),
            ("crates/wenlan-core/src/db/child.rs", child_source, false),
            (
                "crates/wenlan-core/src/db/module_alias.rs",
                module_alias,
                false,
            ),
            ("crates/wenlan-core/src/db/glob_alias.rs", glob_alias, false),
            ("crates/wenlan-core/src/db/unrelated.rs", unrelated, false),
        ],
        &[],
    );
    for rejected in [
        "root_super",
        "root_crate",
        "child_crate",
        "module_alias",
        "glob_alias",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(rejected)),
            "crate-visible DB capability {rejected} was not rejected: {violations:?}"
        );
    }
    for allowed in [
        "root_db_internal",
        "child_internal",
        "child_db_internal",
        "unrelated",
    ] {
        assert!(
            !violations
                .iter()
                .any(|violation| violation.contains(allowed)),
            "DB-internal or unrelated surface {allowed} was rejected: {violations:?}"
        );
    }
}

#[test]
fn eval_judge_relic_is_private_and_module_graph_classified() {
    let root = super::repo_root();
    let module_path = "crates/wenlan-core/src/eval/eval_judge.rs";
    assert!(
        root.join(module_path).is_file(),
        "the historical eval_judge relic must remain present"
    );
    let eval_module = std::fs::read_to_string(root.join("crates/wenlan-core/src/eval/mod.rs"))
        .expect("read eval module");
    assert!(
        eval_module
            .lines()
            .any(|line| line.trim() == "mod eval_judge;"),
        "eval_judge must be wired as a private empty module"
    );
    assert!(
        analyze_repository(&root)
            .visited_files
            .contains(module_path),
        "eval_judge must be classified by the compiler module graph"
    );
}

#[test]
fn opaque_test_support_has_no_visible_raw_signature() {
    let source = include_str!("../db/test_support_test.rs");
    assert_eq!(
        visible_raw_signature_violations(source),
        Vec::<String>::new()
    );
    assert_eq!(rows_lifetime_shape_violations(source), Vec::<String>::new());
    assert!(
        !rows_lifetime_shape_violations(
            "pub(crate) struct TestDbRows<'session> { rows: OpaqueRows }"
        )
        .is_empty(),
        "removing the session borrow marker must fail the lifetime tooth"
    );
}

#[test]
fn r4_test_support_raw_and_api_manifests_match_exactly() {
    let root = super::repo_root();
    let analysis = analyze_repository(&root);
    assert!(
        analysis.errors.is_empty(),
        "R4 source graph must resolve and parse completely: {:#?}",
        analysis.errors
    );
    let mut db_owned = Vec::new();
    let mut external = Vec::new();
    for path in &analysis.visited_files {
        let source = std::fs::read_to_string(root.join(path))
            .unwrap_or_else(|error| panic!("read classified source {path}: {error}"));
        if path == "crates/wenlan-core/src/db.rs" || path.starts_with("crates/wenlan-core/src/db/")
        {
            let file_name = Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            let inherited_test = file_name.contains("_test") || file_name.ends_with("tests.rs");
            db_owned.push((path.clone(), source, inherited_test));
        } else {
            external.push((path.clone(), source));
        }
    }
    let db_owned_refs: Vec<(&str, &str, bool)> = db_owned
        .iter()
        .map(|(path, source, inherited_test)| (path.as_str(), source.as_str(), *inherited_test))
        .collect();
    let external_refs: Vec<(&str, &str)> = external
        .iter()
        .map(|(path, source)| (path.as_str(), source.as_str()))
        .collect();
    assert_eq!(
        db_owned_test_api_violations(&db_owned_refs, &external_refs),
        Vec::<String>::new(),
        "DB-owned test APIs must not expose raw libSQL handles"
    );
    let raw_manifest =
        std::fs::read_to_string(root.join(RAW_MANIFEST_PATH)).expect("read R4 raw manifest");
    compare_raw_manifest(&analysis.raw_uses, &raw_manifest)
        .unwrap_or_else(|error| panic!("{}", error.0));
    let support_manifest = std::fs::read_to_string(root.join(SUPPORT_MANIFEST_PATH))
        .expect("read R4 support manifest");
    compare_support_manifest(&analysis.support_calls, &support_manifest)
        .unwrap_or_else(|error| panic!("{}", error.0));
}

#[test]
fn support_manifest_rejects_duplicate_removed_extra_and_moved_rows() {
    let actual = vec![SupportCall {
        identity: SupportIdentity {
            path: "src/one.rs".to_string(),
            owner: "crate::one::test_body".to_string(),
            callee: "test_primary_session".to_string(),
            ordinal: 1,
        },
        line: 8,
        test_only: true,
    }];
    let exact = "src/one.rs|crate::one::test_body|test_primary_session|1\n";
    assert!(compare_support_manifest(&actual, exact).is_ok());
    for stale in [
        "",
        concat!(
            "src/one.rs|crate::one::test_body|test_primary_session|1\n",
            "src/two.rs|crate::two::test_body|test_secondary_session|1\n",
        ),
        "src/moved.rs|crate::moved::test_body|test_primary_session|1\n",
        "src/one.rs|crate::one::test_body|test_secondary_session|1\n",
    ] {
        assert!(
            compare_support_manifest(&actual, stale).is_err(),
            "{stale:?} must fail closed"
        );
    }
    assert!(
        compare_support_manifest(&actual, &(exact.to_string() + exact)).is_err(),
        "duplicate support rows must fail before set comparison"
    );
}

#[test]
fn r4_test_support_layout_and_parser_dependencies_are_pinned() {
    let root = super::repo_root();
    let cargo = std::fs::read_to_string(root.join("crates/wenlan-core/Cargo.toml"))
        .expect("read core Cargo.toml");
    for exact_dependency in [
        r#"syn = { version = "=2.0.117", features = ["full", "visit"] }"#,
        r#"proc-macro2 = { version = "=1.0.106", features = ["span-locations"] }"#,
    ] {
        assert_eq!(
            cargo.matches(exact_dependency).count(),
            1,
            "R4 parser dependency must stay exact: {exact_dependency}"
        );
    }

    let drift_guard = std::fs::read_to_string(root.join("crates/wenlan-core/src/drift_guard.rs"))
        .expect("read drift guard");
    assert_eq!(
        drift_guard
            .matches("#[path = \"drift_guard/r4_test_support_test.rs\"]")
            .count(),
        1,
        "the giant guard owns only the child-module declaration"
    );
    for parser_symbol in [
        "enum RawShape",
        "struct SupportDataflow",
        "fn analyze_repository",
    ] {
        assert!(
            !drift_guard.contains(parser_symbol),
            "R4 parser implementation leaked back into drift_guard.rs: {parser_symbol}"
        );
    }

    let db =
        std::fs::read_to_string(root.join("crates/wenlan-core/src/db.rs")).expect("read db.rs");
    assert_eq!(
        db.matches("#[path = \"db/test_support_test.rs\"]").count(),
        1,
        "db.rs must wire exactly one opaque test-support module"
    );
}
