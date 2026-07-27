// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliSpaceOperation {
    Read,
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliSpaceSource {
    StrictEnv,
    Explicit,
    DefaultEnv,
    CwdConfig,
    CwdRepo,
    AllSpaces,
    Unscoped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliSpaceContext {
    pub space: Option<String>,
    pub source: CliSpaceSource,
}

#[derive(Debug, Clone)]
pub struct CliSpaceInputs {
    pub strict_space: Option<String>,
    pub default_space: Option<String>,
    pub explicit_space: Option<String>,
    pub all_spaces: bool,
    pub cwd: Option<PathBuf>,
    pub spaces_file: Option<PathBuf>,
}

impl CliSpaceInputs {
    pub fn from_process(
        explicit_space: Option<String>,
        all_spaces: bool,
        cwd: Option<PathBuf>,
    ) -> Self {
        Self {
            strict_space: nonempty(std::env::var("WENLAN_SPACE").ok().as_deref()),
            default_space: nonempty(std::env::var("WENLAN_DEFAULT_SPACE").ok().as_deref()),
            explicit_space: nonempty(explicit_space.as_deref()),
            all_spaces,
            cwd,
            spaces_file: dirs::home_dir().map(|home| home.join(".wenlan/spaces.toml")),
        }
    }
}

pub fn resolve_cli_space(
    explicit_space: Option<String>,
    all_spaces: bool,
    cwd: Option<PathBuf>,
    operation: CliSpaceOperation,
    registered_spaces: &HashSet<String>,
) -> Result<CliSpaceContext> {
    resolve_cli_space_with(
        &CliSpaceInputs::from_process(explicit_space, all_spaces, cwd),
        operation,
        registered_spaces,
    )
}

pub fn resolve_cli_space_with(
    inputs: &CliSpaceInputs,
    operation: CliSpaceOperation,
    registered_spaces: &HashSet<String>,
) -> Result<CliSpaceContext> {
    let strict = nonempty(inputs.strict_space.as_deref());
    let explicit = nonempty(inputs.explicit_space.as_deref());
    let fallback = nonempty(inputs.default_space.as_deref());

    if inputs.all_spaces && explicit.is_some() {
        bail!("--space and --all-spaces cannot be used together");
    }
    if inputs.all_spaces && operation == CliSpaceOperation::Write {
        bail!("--all-spaces is valid only for read commands");
    }

    if let Some(strict) = strict {
        if inputs.all_spaces {
            bail!("--all-spaces cannot bypass the WENLAN_SPACE strict pin");
        }
        if explicit.as_deref().is_some_and(|value| value != strict) {
            bail!(
                "--space conflicts with the WENLAN_SPACE strict pin '{}'",
                strict
            );
        }
        validate_registered(&strict, registered_spaces, "WENLAN_SPACE")?;
        return Ok(CliSpaceContext {
            space: Some(strict),
            source: CliSpaceSource::StrictEnv,
        });
    }

    if inputs.all_spaces {
        return Ok(CliSpaceContext {
            space: None,
            source: CliSpaceSource::AllSpaces,
        });
    }

    if let Some(explicit) = explicit {
        validate_registered(&explicit, registered_spaces, "--space")?;
        return Ok(CliSpaceContext {
            space: Some(explicit),
            source: CliSpaceSource::Explicit,
        });
    }

    if let Some(fallback) = fallback {
        validate_registered(&fallback, registered_spaces, "WENLAN_DEFAULT_SPACE")?;
        return Ok(CliSpaceContext {
            space: Some(fallback),
            source: CliSpaceSource::DefaultEnv,
        });
    }

    if let (Some(cwd), Some(config)) = (&inputs.cwd, &inputs.spaces_file) {
        if let Some(mapped) = longest_mapping(cwd, config) {
            validate_registered(&mapped, registered_spaces, "cwd mapping")?;
            return Ok(CliSpaceContext {
                space: Some(mapped),
                source: CliSpaceSource::CwdConfig,
            });
        }
    }

    if let Some(cwd) = &inputs.cwd {
        if let Some(repo) = repository_basename(cwd) {
            if registered_spaces.contains(&repo) {
                return Ok(CliSpaceContext {
                    space: Some(repo),
                    source: CliSpaceSource::CwdRepo,
                });
            }
        }
    }

    Ok(CliSpaceContext {
        space: None,
        source: CliSpaceSource::Unscoped,
    })
}

pub fn resolve_agent_name(explicit: Option<&str>, environment: Option<&str>) -> Option<String> {
    nonempty(explicit).or_else(|| nonempty(environment))
}

pub fn resolve_native_read_space(
    strict_space: Option<&str>,
    explicit_space: Option<&str>,
    all_spaces: bool,
) -> Result<Option<String>> {
    let strict = nonempty(strict_space);
    let explicit = nonempty(explicit_space);
    if all_spaces {
        if strict.is_some() {
            bail!("--all-spaces cannot bypass the WENLAN_SPACE strict pin");
        }
        return Ok(None);
    }
    if let Some(strict) = strict {
        if explicit.as_deref().is_some_and(|value| value != strict) {
            bail!(
                "--space conflicts with the WENLAN_SPACE strict pin '{}'",
                strict
            );
        }
        return Ok(Some(strict));
    }
    Ok(explicit)
}

fn nonempty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn validate_registered(
    name: &str,
    registered_spaces: &HashSet<String>,
    source: &str,
) -> Result<()> {
    if !registered_spaces.contains(name) {
        bail!("{source} names unregistered Space '{name}'");
    }
    Ok(())
}

fn longest_mapping(cwd: &Path, config: &Path) -> Option<String> {
    let body = std::fs::read_to_string(config).ok()?;
    let mut mappings: Vec<(String, String)> = Vec::new();
    let mut prefix: Option<String> = None;
    let mut space: Option<String> = None;

    let mut finish = |prefix: &mut Option<String>, space: &mut Option<String>| {
        if let (Some(prefix), Some(space)) = (prefix.take(), space.take()) {
            mappings.push((prefix, space));
        }
    };
    for line in body.lines() {
        let line = line.trim();
        if line == "[[mapping]]" {
            finish(&mut prefix, &mut space);
        } else if let Some(value) = parse_toml_string(line, "prefix") {
            prefix = Some(value);
        } else if let Some(value) = parse_toml_string(line, "space") {
            space = Some(value);
        }
    }
    finish(&mut prefix, &mut space);

    mappings
        .into_iter()
        .filter_map(|(prefix, space)| {
            let expanded = if prefix == "~" {
                dirs::home_dir()
            } else if let Some(suffix) = prefix.strip_prefix("~/") {
                dirs::home_dir().map(|home| home.join(suffix))
            } else {
                Some(PathBuf::from(prefix))
            }?;
            cwd.starts_with(&expanded)
                .then_some((expanded.as_os_str().len(), space))
        })
        .max_by_key(|(length, _)| *length)
        .map(|(_, space)| space)
}

fn parse_toml_string(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?.trim_start();
    let rest = rest.strip_prefix('=')?.trim();
    let value = rest.strip_prefix('"')?.strip_suffix('"')?.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn repository_basename(cwd: &Path) -> Option<String> {
    let common = Command::new("git")
        .args(["rev-parse", "--path-format=absolute", "--git-common-dir"])
        .current_dir(cwd)
        .output()
        .ok()?;
    if !common.status.success() {
        return None;
    }
    let common = PathBuf::from(String::from_utf8(common.stdout).ok()?.trim());
    if common.file_name().and_then(|name| name.to_str()) == Some(".git") {
        return common
            .parent()?
            .file_name()?
            .to_str()
            .map(ToOwned::to_owned);
    }

    let root = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(cwd)
        .output()
        .ok()?;
    if !root.status.success() {
        return None;
    }
    PathBuf::from(String::from_utf8(root.stdout).ok()?.trim())
        .file_name()?
        .to_str()
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn registered(names: &[&str]) -> HashSet<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn cli_space_strict_pin_rejects_conflicts() {
        let inputs = CliSpaceInputs {
            strict_space: Some("work".into()),
            default_space: Some("personal".into()),
            explicit_space: Some("personal".into()),
            all_spaces: false,
            cwd: None,
            spaces_file: None,
        };
        let error = resolve_cli_space_with(
            &inputs,
            CliSpaceOperation::Write,
            &registered(&["work", "personal"]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("WENLAN_SPACE"));
    }

    #[test]
    fn cli_space_explicit_beats_default_and_mapping() {
        let tmp = tempfile::tempdir().unwrap();
        let config = tmp.path().join("spaces.toml");
        std::fs::write(
            &config,
            format!(
                "[[mapping]]\nprefix = \"{}\"\nspace = \"mapped\"\n",
                tmp.path().display()
            ),
        )
        .unwrap();
        let inputs = CliSpaceInputs {
            strict_space: None,
            default_space: Some("fallback".into()),
            explicit_space: Some("explicit".into()),
            all_spaces: false,
            cwd: Some(tmp.path().to_path_buf()),
            spaces_file: Some(config),
        };
        let resolved = resolve_cli_space_with(
            &inputs,
            CliSpaceOperation::Write,
            &registered(&["explicit", "fallback", "mapped"]),
        )
        .unwrap();
        assert_eq!(resolved.space.as_deref(), Some("explicit"));
        assert_eq!(resolved.source, CliSpaceSource::Explicit);
    }

    #[test]
    fn cli_space_longest_mapping_then_registered_repo_basename() {
        let tmp = tempfile::tempdir().unwrap();
        let outer = tmp.path().join("repo");
        let inner = outer.join("nested");
        std::fs::create_dir_all(&inner).unwrap();
        let config = tmp.path().join("spaces.toml");
        std::fs::write(
            &config,
            format!(
                "[[mapping]]\nprefix = \"{}\"\nspace = \"outer\"\n\
                 [[mapping]]\nprefix = \"{}\"\nspace = \"inner\"\n",
                outer.display(),
                inner.display()
            ),
        )
        .unwrap();
        let inputs = CliSpaceInputs {
            strict_space: None,
            default_space: None,
            explicit_space: None,
            all_spaces: false,
            cwd: Some(inner.clone()),
            spaces_file: Some(config),
        };
        let resolved = resolve_cli_space_with(
            &inputs,
            CliSpaceOperation::Read,
            &registered(&["outer", "inner"]),
        )
        .unwrap();
        assert_eq!(resolved.space.as_deref(), Some("inner"));
        assert_eq!(resolved.source, CliSpaceSource::CwdConfig);
    }

    #[test]
    fn cli_space_all_spaces_is_read_only() {
        let inputs = CliSpaceInputs {
            strict_space: None,
            default_space: Some("fallback".into()),
            explicit_space: None,
            all_spaces: true,
            cwd: None,
            spaces_file: None,
        };
        let read =
            resolve_cli_space_with(&inputs, CliSpaceOperation::Read, &registered(&["fallback"]))
                .unwrap();
        assert_eq!(read.source, CliSpaceSource::AllSpaces);
        assert_eq!(read.space, None);
        assert!(resolve_cli_space_with(
            &inputs,
            CliSpaceOperation::Write,
            &registered(&["fallback"])
        )
        .is_err());
    }

    #[test]
    fn native_read_space_enforces_strict_pin_without_registration_preflight() {
        assert_eq!(
            resolve_native_read_space(Some("work"), Some("work"), false).unwrap(),
            Some("work".into())
        );
        assert!(resolve_native_read_space(Some("work"), Some("personal"), false).is_err());
        assert!(resolve_native_read_space(Some("work"), None, true).is_err());
        assert_eq!(
            resolve_native_read_space(None, Some("future-space"), false).unwrap(),
            Some("future-space".into())
        );
        assert_eq!(resolve_native_read_space(None, None, true).unwrap(), None);
    }

    #[test]
    fn agent_name_explicit_beats_environment() {
        assert_eq!(
            resolve_agent_name(Some("cli"), Some("environment")),
            Some("cli".to_string())
        );
        assert_eq!(
            resolve_agent_name(None, Some("environment")),
            Some("environment".to_string())
        );
    }
}
