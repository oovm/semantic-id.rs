use super::*;
use uuid::Uuid;

impl<K> From<Uuid> for Semantic32<K> {
    fn from(value: Uuid) -> Self {
        let (h, l) = value.as_u64_pair();
        // unix_ts_ms = 48bit
        let a = h & 0xFFFF_FFFF_FFFF_0000;
        // rand_b = 16 bit
        let b = l & 0x0000_0000_0000_FFFF;
        Self { id: a | b, kind: Default::default() }
    }
}

#[cfg(test)]
mod test {
    use crate::Semantic64;
    use std::time::SystemTime;
    use uuid::{Timestamp, Uuid};

    #[test]
    fn check_from_uuid() {
        let uuid = Uuid::parse_str("a1a2a3a4-b1b2-c1c2-d1d2-d3d4d5d6d7d8").expect("Invalid Uuid");
        let seid: Semantic64<()> = uuid.into();
        assert_eq!(format!("{:016x}", seid), "a1a2a3a4b1b2d7d8")
    }

    #[test]
    fn check_same_head() {
        let time = SystemTime::now();
        let unix = time.duration_since(SystemTime::UNIX_EPOCH).unwrap();
        let ts = Timestamp::from_unix_time(unix.as_secs(), unix.subsec_nanos(), 0, 0);
        let uuid = format!("{}", Uuid::new_v7(ts));
        let seid = format!("{:016x}", Semantic64::<()>::new(unix.as_millis() as u64));
        assert_eq!(uuid[0..8], seid[0..8]);
        assert_eq!(uuid[9..13], seid[8..12]);
    }
}
