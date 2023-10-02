use semantic_id::{Semantic32, Semantic64, SemanticKey};
use std::str::FromStr;

#[test]
fn ready() {
    println!("it works!")
}

pub type Test64 = Semantic64<TestKey>;
pub type Test32 = Semantic32<TestKey>;
pub struct TestKey;

impl SemanticKey for TestKey {
    const KEY: &'static str = "test";
}

#[test]
fn test_display64() {
    assert_eq!(Test64::from(0u64).to_string(), "test-0");
    assert_eq!(Test64::from(1u64).to_string(), "test-1");
    assert_eq!(Test64::from(10u64).to_string(), "test-a");
    assert_eq!(Test64::from(100u64).to_string(), "test-2s");
    assert_eq!(Test64::from(1000u64).to_string(), "test-rs");
    assert_eq!(Test64::from(10000u64).to_string(), "test-7ps");
}

#[test]
fn test_parse64() {
    assert_eq!(Test64::from_str("test-0"), Ok(Test64::from(0u64)));
    assert_eq!(Test64::from_str("test-1"), Ok(Test64::from(1u64)));
    assert_eq!(Test64::from_str("test-A"), Ok(Test64::from(10u64)));
    assert_eq!(Test64::from_str("test-2S"), Ok(Test64::from(100u64)));
    assert_eq!(Test64::from_str("test-RS"), Ok(Test64::from(1000u64)));
    assert_eq!(Test64::from_str("test-7PS"), Ok(Test64::from(10000u64)));
}
