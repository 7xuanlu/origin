// SPDX-License-Identifier: Apache-2.0
use assert_cmd::Command;
use serde_json::json;

fn cli() -> Command {
    Command::cargo_bin("wenlan").expect("wenlan binary built")
}

#[test]
fn pages_resolve_id_returns_frontmatter_origin_id_without_opening_the_page() {
    let root = tempfile::tempdir().unwrap();
    let data = root.path().join("data");
    let pages = root.path().join("pages");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&pages).unwrap();
    std::fs::write(
        data.join("config.json"),
        serde_json::to_vec(&json!({ "knowledge_path": pages })).unwrap(),
    )
    .unwrap();
    std::fs::write(
        pages.join("project-atlas.md"),
        "---\ntitle: Project Atlas\norigin_id: page_atlas\n---\n\nPrivate body\n",
    )
    .unwrap();

    cli()
        .env("WENLAN_DATA_DIR", &data)
        .args(["pages", "Project Atlas", "--resolve-id"])
        .assert()
        .success()
        .stdout("page_atlas\n")
        .stderr("");
}
