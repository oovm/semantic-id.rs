use crate::{Semantic64, SemanticKey};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};
use std::{marker::PhantomData, str::FromStr};

impl<K: SemanticKey> Serialize for Semantic64<K> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.id.serialize(serializer)
    }
}

impl<'de, K> Deserialize<'de> for Semantic64<K> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(Semantic64Visitor { kind: Default::default() })
    }
}

pub struct Semantic64Visitor<K> {
    kind: PhantomData<K>,
}

impl<'de, K> serde::de::Visitor<'de> for Semantic64Visitor<K> {
    type Value = Semantic64<K>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a string representing of the semantic id")
    }

    fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
    where
        E: Error,
    {
        Ok(Semantic64::from(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
    where
        E: Error,
    {
        Ok(Semantic64::from(v))
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: Error,
    {
        match Semantic64::from_str(v) {
            Ok(o) => Ok(o),
            Err(_) => Err(E::custom("Hexadecimal string parsing failed".to_string())),
        }
    }
}
