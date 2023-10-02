use crate::{ID_STATE16, SemanticKey};
use std::{
    fmt::{Debug, Display, Formatter, LowerHex},
    hash::{Hash, Hasher},
    marker::PhantomData,
    num::ParseIntError,
    str::FromStr,
};

#[cfg(feature = "sea-orm")]
mod for_orm;
#[cfg(feature = "poem-openapi")]
mod for_poem;
#[cfg(feature = "serde")]
mod for_serde;

#[cfg(feature = "uuid")]
mod for_uuid;

#[cfg(feature = "sqlx")]
mod for_sqlx_pgsql;

/// `Semantic64<K>` is a struct that contains a 64-bit integer ID. This ID is composed of the following parts:
///
/// - 48-bit millisecond timestamp: Used to represent the time when the ID was created. The timestamp range starts from January 1, 1970, 00:00:00 UTC, and can represent approximately 279,000 years.
/// - 8-bit device ID: Used to identify the device or service instance that created the ID. It can represent up to 256 different devices.
/// - 8-bit sequential ID: Used to generate multiple IDs within a 1-millisecond interval. It can represent a sequential ID between 0 and 255.
///
/// This ID structure can guarantee uniqueness in a distributed system, while also providing time and device information. It is suitable for scenarios that require generating unique IDs, such as message IDs, event IDs, and so on.
pub struct Semantic32<K> {
    id: u32,
    kind: PhantomData<K>,
}

impl<K> Copy for Semantic32<K> {}

impl<K> Clone for Semantic32<K> {
    fn clone(&self) -> Self {
        Self { id: self.id, kind: self.kind }
    }
}

impl<K> Hash for Semantic32<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<K> PartialEq for Semantic32<K> {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}

impl<K> Eq for Semantic32<K> {}

impl<K: SemanticKey> Debug for Semantic32<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match K::KEY {
            "" => write!(f, "{}", self.id),
            s => write!(f, "{}-{}", s, self.id),
        }
    }
}

impl<K: SemanticKey> Display for Semantic32<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match K::KEY {
            "" => write!(f, "{}", self.id),
            s => write!(f, "{}-{:x}", s, self.id),
        }
    }
}

impl<K> LowerHex for Semantic32<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        LowerHex::fmt(&self.id, f)
    }
}

impl<K> Default for Semantic32<K> {
    fn default() -> Self {
        Self { id: 0, kind: Default::default() }
    }
}

impl<K> FromStr for Semantic32<K> {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.rsplit('-');
        match parts.next() {
            Some(s) => {
                let id = u32::from_str_radix(&s, 36)?;
                Ok(Self { id, kind: Default::default() })
            }
            None => Ok(Self::default()),
        }
    }
}

impl<K> From<i32> for Semantic32<K> {
    fn from(value: i32) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl<K> From<u32> for Semantic32<K> {
    fn from(value: u32) -> Self {
        Self { id: value, kind: Default::default() }
    }
}

impl<K> Semantic32<K> {
    pub fn new(unix_ms: u64) -> Semantic32<K> {
        Self { id: ID_STATE16.lock().unwrap().generate64_by(unix_ms) as u32, kind: Default::default() }
    }
    pub fn now() -> Self {
        Self { id: ID_STATE16.lock().unwrap().generate64_now() as u32, kind: Default::default() }
    }
    pub fn as_u32(&self) -> u32 {
        self.id
    }
    pub fn as_i32(&self) -> i32 {
        unsafe { std::mem::transmute(self.id) }
    }
}
