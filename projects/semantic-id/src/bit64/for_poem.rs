use super::*;

use poem_openapi::{
    __private::serde_json::Value,
    registry::{MetaSchema, MetaSchemaRef, Registry},
    types::{IsObjectType, ParseError, ParseFromJSON, ParseFromXML, ParseFromYAML, ToJSON, ToXML, ToYAML, Type},
};
use std::borrow::Cow;

impl<K: SemanticKey + Send + Sync> Type for Semantic64<K> {
    const IS_REQUIRED: bool = true;
    type RawValueType = u64;
    type RawElementValueType = u64;
    fn name() -> Cow<'static, str> {
        match K::KEY {
            "" => Cow::Borrowed("Key64"),
            s => Cow::Owned(format!("Semantic64<\"{}\">", s)),
        }
    }
    fn schema_ref() -> MetaSchemaRef {
        MetaSchemaRef::Inline(Box::new(Self::schema()))
    }
    fn register(registry: &mut Registry) {
        registry.create_schema::<Self, _>(Self::name().into_owned(), |registry| {
            u64::register(registry);
            Self::schema()
        })
    }
    fn as_raw_value(&self) -> Option<&Self::RawValueType> {
        Some(&self.id)
    }
    fn raw_element_iter<'a>(&'a self) -> Box<dyn Iterator<Item = &'a Self::RawElementValueType> + 'a> {
        Box::new(self.as_raw_value().into_iter())
    }
}
impl<Key: SemanticKey + Send + Sync> IsObjectType for Semantic64<Key> {}

impl<K: SemanticKey> ParseFromJSON for Semantic64<K> {
    fn parse_from_json(value: Option<Value>) -> Result<Self, ParseError<Self>> {
        match value.unwrap_or_default() {
            Value::Null => Ok(Semantic64 { id: 0, kind: Default::default() }),
            Value::String(s) => match Self::from_str(&s) {
                Ok(o) => Ok(o),
                Err(_) => Err(ParseError::custom(format!("Hexadecimal string parsing failed for `{}`", K::KEY))),
            },
            Value::Number(v) => match v.as_i64() {
                Some(s) => Ok(Semantic64::from(s)),
                None => Err(ParseError::custom(format!("Decimal integer parsing failed for `{}`", K::KEY))),
            },
            _ => Err(ParseError::custom(format!("Hexadecimal string parsing failed for `{}`", K::KEY))),
        }
    }
}
impl<Key: SemanticKey> ParseFromXML for Semantic64<Key> {
    fn parse_from_xml(value: Option<Value>) -> Result<Self, ParseError<Self>> {
        ParseFromJSON::parse_from_json(value)
    }
}
impl<Key: SemanticKey> ParseFromYAML for Semantic64<Key> {
    fn parse_from_yaml(value: Option<Value>) -> Result<Self, ParseError<Self>> {
        ParseFromJSON::parse_from_json(value)
    }
}

impl<Key: SemanticKey> ToJSON for Semantic64<Key> {
    fn to_json(&self) -> Option<Value> {
        Some(Value::String(self.to_string()))
    }
}

impl<Key: SemanticKey> ToXML for Semantic64<Key> {
    fn to_xml(&self) -> Option<Value> {
        self.to_json()
    }
}

impl<Key: SemanticKey> ToYAML for Semantic64<Key> {
    fn to_yaml(&self) -> Option<Value> {
        self.to_json()
    }
}

impl<K: SemanticKey> Semantic64<K> {
    pub fn schema() -> MetaSchema {
        MetaSchema {
            ty: "string",
            title: Some(Self::name().into_owned()),
            description: Some("The 64-bit hexadecimal string identifier"),
            default: Some(Value::String(format!("{}-0", K::KEY))),
            ..MetaSchema::ANY
        }
    }
}
