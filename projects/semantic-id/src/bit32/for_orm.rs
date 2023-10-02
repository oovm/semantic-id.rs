use crate::{Semantic64, SemanticKey};
use sea_orm::{
    ColIdx, ColumnType, ConnectionTrait, DbErr, EntityTrait, IntoActiveModel, PrimaryKeyTrait, QueryResult, Select,
    TryGetError, TryGetable, Value,
    sea_query::{ArrayType, Nullable, ValueType, ValueTypeErr},
};

impl<K: SemanticKey> TryGetable for Semantic64<K> {
    fn try_get_by<I: ColIdx>(res: &QueryResult, index: I) -> Result<Self, TryGetError> {
        let value = i64::try_get_by(res, index)?;
        Ok(Semantic64::from(value))
    }
}

impl<K: SemanticKey> From<Semantic64<K>> for Value {
    fn from(value: Semantic64<K>) -> Self {
        Value::BigUnsigned(Some(value.id))
    }
}

impl<K: SemanticKey> Semantic64<K> {
    pub fn find(&self) -> Select<K::Entity>
    where
        <<K::Entity as EntityTrait>::PrimaryKey as PrimaryKeyTrait>::ValueType: From<i64>,
    {
        EntityTrait::find_by_id(self.as_i64())
    }
    pub async fn find_one<C: ConnectionTrait>(&self, db: &C) -> Result<<K::Entity as EntityTrait>::Model, DbErr>
    where
        <<K::Entity as EntityTrait>::PrimaryKey as PrimaryKeyTrait>::ValueType: From<i64>,
    {
        match self.find().one(db).await? {
            Some(s) => Ok(s),
            None => Err(DbErr::RecordNotFound(self.to_string())),
        }
    }
    pub async fn edit_one<C: ConnectionTrait>(&self, db: &C) -> Result<<K::Entity as EntityTrait>::ActiveModel, DbErr>
    where
        <<K::Entity as EntityTrait>::PrimaryKey as PrimaryKeyTrait>::ValueType: From<i64>,
        <K::Entity as EntityTrait>::Model: IntoActiveModel<<K::Entity as EntityTrait>::ActiveModel>,
    {
        Ok(self.find_one(db).await?.into_active_model())
    }
}

impl<K> Nullable for Semantic64<K> {
    fn null() -> Value {
        Value::BigUnsigned(Some(0))
    }
}

impl<K> ValueType for Semantic64<K> {
    fn try_from(v: Value) -> Result<Self, ValueTypeErr> {
        match v {
            Value::Unsigned(v) => Ok(Semantic64::from(v.unwrap_or(0) as u64)),
            Value::BigUnsigned(v) => Ok(Semantic64::from(v.unwrap_or(0))),
            Value::Int(v) => Ok(Semantic64::from(v.unwrap_or(0) as i64)),
            Value::BigInt(v) => Ok(Semantic64::from(v.unwrap_or(0))),
            _ => Err(ValueTypeErr),
        }
    }

    fn type_name() -> String {
        "Key64".to_string()
    }

    fn array_type() -> ArrayType {
        ArrayType::BigUnsigned
    }

    fn column_type() -> ColumnType {
        ColumnType::BigUnsigned
    }
}
