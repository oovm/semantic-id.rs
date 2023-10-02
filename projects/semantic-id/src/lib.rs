mod bit32;
mod bit64;
mod state16;

mod helpers;

pub use crate::{
    bit32::Semantic32,
    bit64::Semantic64,
    helpers::SemanticKey,
    state16::{GenerateState16, ID_STATE16},
};
