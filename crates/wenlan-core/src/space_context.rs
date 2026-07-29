// SPDX-License-Identifier: Apache-2.0

use wenlan_types::{Space, WriteSpaceSource};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedWriteSpace {
    pub space_id: Option<String>,
    pub space_name: Option<String>,
    pub source: WriteSpaceSource,
}

impl ResolvedWriteSpace {
    pub fn uncategorized() -> Self {
        Self {
            space_id: None,
            space_name: None,
            source: WriteSpaceSource::Uncategorized,
        }
    }
}

#[derive(Debug)]
pub struct LegacyDefaultImport {
    pub already_processed: bool,
    pub imported_space: Option<Space>,
    pub invalid_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use crate::db::tests::test_db;
    use crate::db::UNFILED_SPACE_ID;
    use crate::events::NoopEmitter;
    use crate::WenlanError;
    use wenlan_types::{WriteSpaceSource, WriteSpaceTarget};

    #[tokio::test]
    async fn default_space_set_replace_rename_delete_lifecycle() {
        let (db, _dir) = test_db().await;
        db.run_migrations(&NoopEmitter).await.unwrap();
        let work = db.create_space("work", None, false).await.unwrap();
        let personal = db.create_space("personal", None, false).await.unwrap();

        let sentinel_error = db.set_default_space(UNFILED_SPACE_ID).await.unwrap_err();
        assert!(matches!(sentinel_error, WenlanError::Validation(_)));

        let selected = db.set_default_space(&work.id).await.unwrap();
        assert_eq!(selected.id, work.id);
        assert!(selected.is_default);
        assert_eq!(
            db.list_spaces()
                .await
                .unwrap()
                .into_iter()
                .filter(|space| space.is_default)
                .count(),
            1
        );

        db.update_space("work", "career", None).await.unwrap();
        let renamed = db.get_default_space().await.unwrap().unwrap();
        assert_eq!(renamed.id, work.id);
        assert_eq!(renamed.name, "career");

        db.set_default_space(&personal.id).await.unwrap();
        assert_eq!(
            db.get_default_space().await.unwrap().unwrap().id,
            personal.id
        );
        assert!(!db.get_space("career").await.unwrap().unwrap().is_default);

        db.delete_space("personal", "keep").await.unwrap();
        assert!(db.get_default_space().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn legacy_default_import_runs_once_without_rewriting_toml() {
        let (db, dir) = test_db().await;
        db.run_migrations(&NoopEmitter).await.unwrap();
        let work = db.create_space("work", None, false).await.unwrap();
        let path = dir.path().join("spaces.toml");
        let original =
            b"default = \"work\"\n\n[[mapping]]\npath = \"/tmp/work\"\nspace = \"work\"\n";
        std::fs::write(&path, original).unwrap();

        let imported = db.import_legacy_default_once(&path).await.unwrap();
        assert!(!imported.already_processed);
        assert_eq!(
            imported
                .imported_space
                .as_ref()
                .map(|space| space.id.as_str()),
            Some(work.id.as_str())
        );
        assert_eq!(std::fs::read(&path).unwrap(), original);

        db.clear_default_space().await.unwrap();
        std::fs::write(&path, b"default = \"personal\"\n").unwrap();
        let second = db.import_legacy_default_once(&path).await.unwrap();
        assert!(second.already_processed);
        assert!(second.imported_space.is_none());
        assert!(db.get_default_space().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn write_space_resolution_carries_stable_id_through_rename() {
        let (db, _dir) = test_db().await;
        db.run_migrations(&NoopEmitter).await.unwrap();
        let work = db.create_space("work", None, false).await.unwrap();

        let resolved = db
            .resolve_write_space(&WriteSpaceTarget::Named("work".into()), None)
            .await
            .unwrap();
        assert_eq!(resolved.space_id.as_deref(), Some(work.id.as_str()));
        assert_eq!(resolved.source, WriteSpaceSource::Request);

        db.update_space("work", "career", None).await.unwrap();
        let finalized = db.finalize_write_space(&resolved).await.unwrap();
        assert_eq!(finalized.space_id, resolved.space_id);
        assert_eq!(finalized.space_name.as_deref(), Some("career"));
        assert_eq!(finalized.source, WriteSpaceSource::Request);

        let explicit = db
            .resolve_write_space(&WriteSpaceTarget::Named("career".into()), None)
            .await
            .unwrap();
        db.delete_space("career", "keep").await.unwrap();
        assert!(matches!(
            db.finalize_write_space(&explicit).await,
            Err(WenlanError::Validation(_))
        ));
    }

    #[tokio::test]
    async fn deleted_default_falls_back_to_uncategorized_at_finalization() {
        let (db, _dir) = test_db().await;
        db.run_migrations(&NoopEmitter).await.unwrap();
        let work = db.create_space("work", None, false).await.unwrap();
        db.set_default_space(&work.id).await.unwrap();

        let resolved = db
            .resolve_write_space(&WriteSpaceTarget::Inherit, None)
            .await
            .unwrap();
        assert_eq!(resolved.source, WriteSpaceSource::Default);

        db.delete_space("work", "keep").await.unwrap();
        let finalized = db.finalize_write_space(&resolved).await.unwrap();
        assert_eq!(finalized.space_id, None);
        assert_eq!(finalized.space_name, None);
        assert_eq!(finalized.source, WriteSpaceSource::Uncategorized);
    }
}
