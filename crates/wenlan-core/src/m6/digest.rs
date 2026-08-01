//! The M6 digest primitive.
//!
//! Every M6 identity is a SHA-256 over **length-prefixed** parts under a
//! **domain tag**. Both properties are load-bearing:
//!
//! * Length prefixing makes the part boundaries unambiguous. Separator framing
//!   collides — `["x\0", "y"]` and `["x", "\0y"]` serialize identically under a
//!   NUL separator but are different inputs. A collision here is an identity
//!   collision, which is a wrong-page write.
//! * The domain tag keeps one part list from meaning two things in two
//!   namespaces, so a `slot_id` can never equal a `page_id` built from the same
//!   bytes.
//!
//! The digest is the full 64 lowercase hex characters. It is never truncated:
//! truncation trades collision resistance for column width, and these values
//! are primary keys.

use sha2::{Digest, Sha256};

/// Hash `parts` under `domain`, returning 64 lowercase hex characters.
pub fn m6_digest(domain: &str, parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    write_part(&mut hasher, domain.as_bytes());
    for part in parts {
        write_part(&mut hasher, part);
    }
    format!("{:x}", hasher.finalize())
}

/// Same as [`m6_digest`] for callers that built owned parts.
pub fn m6_digest_owned(domain: &str, parts: &[Vec<u8>]) -> String {
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    m6_digest(domain, &refs)
}

/// Absorb one part: an 8-byte little-endian length, then the bytes.
fn write_part(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Integer part encoding: ASCII decimal, no leading zeros, `-` for negatives.
/// `i64::to_string` is exactly that.
pub fn int_part(value: i64) -> Vec<u8> {
    value.to_string().into_bytes()
}

/// Boolean part encoding.
pub fn bool_part(value: bool) -> &'static [u8] {
    if value {
        b"t"
    } else {
        b"f"
    }
}

/// An absent optional field is a zero-length part — distinct from a present
/// empty string only because the *arity* of the part list is fixed, so position
/// carries the field identity.
pub const ABSENT_PART: &[u8] = b"";

/// Canonical set encoding: byte-lexicographic sort, exact dedup, then a count
/// part followed by one part per element.
///
/// The leading count is what makes this unambiguous when a set is embedded in a
/// longer part list — without it, a reader cannot tell where the set ends.
pub fn sorted_set<S: AsRef<str>>(items: &[S]) -> Vec<Vec<u8>> {
    let mut sorted: Vec<&[u8]> = items.iter().map(|s| s.as_ref().as_bytes()).collect();
    sorted.sort_unstable();
    sorted.dedup();

    let mut parts = Vec::with_capacity(sorted.len() + 1);
    parts.push(int_part(sorted.len() as i64));
    parts.extend(sorted.into_iter().map(<[u8]>::to_vec));
    parts
}
