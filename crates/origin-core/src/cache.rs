// SPDX-License-Identifier: AGPL-3.0-only
use lru::LruCache;
use std::num::NonZeroUsize;

/// LRU cache for text embeddings to avoid re-computing common queries
pub struct EmbeddingCache {
    cache: LruCache<String, Vec<f32>>,
}

impl EmbeddingCache {
    /// Create a new embedding cache with the specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
        }
    }

    /// Get a cached embedding or return None
    pub fn get(&mut self, query: &str) -> Option<Vec<f32>> {
        let normalized = Self::normalize_query(query);
        self.cache.get(&normalized).cloned()
    }

    /// Insert an embedding into the cache
    pub fn put(&mut self, query: &str, embedding: Vec<f32>) {
        let normalized = Self::normalize_query(query);
        self.cache.put(normalized, embedding);
    }

    /// Normalize query for consistent caching
    fn normalize_query(query: &str) -> String {
        query.to_lowercase().trim().to_string()
    }
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new(100) // Default to 100 cached embeddings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit() {
        let mut cache = EmbeddingCache::new(10);
        let embedding = vec![1.0, 2.0, 3.0];

        cache.put("test query", embedding.clone());

        let result = cache.get("test query");
        assert_eq!(result, Some(embedding));
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = EmbeddingCache::new(10);
        let result = cache.get("missing query");
        assert_eq!(result, None);
    }

    #[test]
    fn test_normalization() {
        let mut cache = EmbeddingCache::new(10);
        let embedding = vec![1.0, 2.0, 3.0];

        cache.put("Test Query", embedding.clone());

        // Different capitalization and whitespace should still hit cache
        assert_eq!(cache.get("test query"), Some(embedding.clone()));
        assert_eq!(cache.get("  TEST QUERY  "), Some(embedding));
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = EmbeddingCache::new(2);

        cache.put("query1", vec![1.0]);
        cache.put("query2", vec![2.0]);
        cache.put("query3", vec![3.0]); // Should evict query1

        assert_eq!(cache.get("query1"), None);
        assert_eq!(cache.get("query2"), Some(vec![2.0]));
        assert_eq!(cache.get("query3"), Some(vec![3.0]));
    }
}
