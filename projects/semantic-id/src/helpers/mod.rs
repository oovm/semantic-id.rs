pub trait SemanticKey: Send + Sync {
    const KEY: &'static str;
}
