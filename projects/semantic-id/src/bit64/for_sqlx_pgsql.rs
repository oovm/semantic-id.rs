use crate::Semantic64;
use sqlx::{
    Database, Decode, Encode, Postgres, Type,
    encode::IsNull,
    error::BoxDynError,
    postgres::{PgHasArrayType, PgTypeInfo, types::Oid},
};

impl<'q, K> Encode<'q, Postgres> for Semantic64<K> {
    fn encode_by_ref(&self, buf: &mut <Postgres as Database>::ArgumentBuffer<'q>) -> Result<IsNull, BoxDynError> {
        <i64 as Encode<'q, Postgres>>::encode(self.as_i64(), buf)
    }
}

impl<'q, K> Decode<'q, Postgres> for Semantic64<K> {
    fn decode(value: <Postgres as Database>::ValueRef<'q>) -> Result<Self, BoxDynError> {
        Ok(Semantic64::from(<i64 as Decode<'q, Postgres>>::decode(value)?))
    }
}

impl<K> Type<Postgres> for Semantic64<K> {
    fn type_info() -> <Postgres as Database>::TypeInfo {
        PgTypeInfo::with_oid(Oid(20))
    }
}

impl<K> PgHasArrayType for Semantic64<K> {
    fn array_type_info() -> PgTypeInfo {
        PgTypeInfo::with_oid(Oid(1016))
    }
}
