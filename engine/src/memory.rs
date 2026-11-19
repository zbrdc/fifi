//! Engine-wide memory policy.
//!
//! One place turns measured headroom into a budget and charges against it, so
//! families share the rule instead of each keeping its own constants.

use crate::EngError;

/// Physical RAM and the bytes currently available to the process, before model-specific reserves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostMemory {
    pub(crate) total: u64,
    pub(crate) available: u64,
}

impl HostMemory {
    pub(crate) fn os_floor(self) -> u64 {
        self.total.div_ceil(10).max(8 << 30)
    }
}

/// 850 permille leaves the 15 percent that device and driver allocations need
/// beyond the engine's own accounting.
pub const DEFAULT_SHARE_PERMILLE: u16 = 850;

/// Fraction of measured available bytes the engine may commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryPolicy {
    share_permille: u16,
}

impl MemoryPolicy {
    pub fn from_permille(share_permille: u16) -> Result<Self, EngError> {
        if !(1..=1000).contains(&share_permille) {
            return Err(EngError::MemoryPlan(
                "memory policy share must be between 1 and 1000 permille".to_owned(),
            ));
        }
        Ok(Self { share_permille })
    }

    pub fn default_share() -> Self {
        Self {
            share_permille: DEFAULT_SHARE_PERMILLE,
        }
    }

    pub fn share_permille(&self) -> u16 {
        self.share_permille
    }

    pub fn limit(&self, available_bytes: u64) -> u64 {
        let share = u128::from(self.share_permille);
        let scaled = u128::from(available_bytes) * share / 1000;
        u64::try_from(scaled).unwrap_or(u64::MAX)
    }

    pub fn admit(&self, available_bytes: u64, charge_bytes: u64) -> Result<(), EngError> {
        let limit = self.limit(available_bytes);
        if charge_bytes > limit {
            return Err(EngError::MemoryPlan(format!(
                "memory policy: charge {charge_bytes} exceeds the {limit} byte budget"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MemoryPolicy;

    #[test]
    fn a_share_of_the_headroom_becomes_the_budget() {
        let policy = MemoryPolicy::from_permille(850).expect("valid share");
        assert_eq!(policy.limit(1000), 850);
        assert_eq!(policy.limit(24 * 1024 * 1024 * 1024), 21904333209);
    }

    #[test]
    fn a_full_share_admits_exactly_the_headroom() {
        let policy = MemoryPolicy::from_permille(1000).expect("valid share");
        assert!(policy.admit(100, 100).is_ok());
        assert!(policy.admit(100, 101).is_err());
    }

    #[test]
    fn a_charge_above_the_budget_is_refused() {
        let policy = MemoryPolicy::from_permille(500).expect("valid share");
        assert!(policy.admit(1000, 500).is_ok());
        assert!(policy.admit(1000, 501).is_err());
    }

    #[test]
    fn the_default_policy_keeps_fifteen_percent_back() {
        let policy = MemoryPolicy::default_share();
        assert_eq!(policy.share_permille(), 850);
        assert_eq!(policy.limit(1000), 850);
    }

    #[test]
    fn a_share_outside_one_to_a_thousand_is_refused() {
        assert!(MemoryPolicy::from_permille(0).is_err());
        assert!(MemoryPolicy::from_permille(1001).is_err());
    }
}
