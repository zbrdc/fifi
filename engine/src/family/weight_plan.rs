use std::collections::BTreeMap;

use crate::{EngError, weights::ShardHeader};

use super::{
    KeyTier, TierAssignment,
    keys::KeyRole,
    quantization::{ModuleQuantization, QuantizationMap},
    spec::FamilySpec,
};

/// One checkpoint key with its storage tier and quantization.
///
/// Keys of packed projections carry `ModuleQuantization::Quantized`. Keys of dense tensors
/// (norms, convolution weights, and recurrent auxiliaries) carry `Unquantized` whatever the
/// configuration default or an incidental override says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanEntry {
    pub tier: KeyTier,
    pub quantization: ModuleQuantization,
}

#[derive(Debug, PartialEq, Eq)]
pub struct WeightPlan {
    entries: BTreeMap<String, PlanEntry>,
}

impl WeightPlan {
    pub fn build(
        spec: &FamilySpec,
        quant: &QuantizationMap,
        headers: &[ShardHeader],
    ) -> Result<Self, EngError> {
        let keys = collect_keys(headers);
        let tiers = TierAssignment::from_keys(&keys, spec)?;
        check_override_keys(quant, &keys)?;
        let entries = resolve_entries(quant, &keys, &tiers)?;
        Ok(Self { entries })
    }

    pub fn tier(&self, key: &str) -> Option<KeyTier> {
        self.entries.get(key).map(|entry| entry.tier)
    }

    pub fn quantization(&self, key: &str) -> Option<ModuleQuantization> {
        self.entries.get(key).map(|entry| entry.quantization)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, PlanEntry)> {
        self.entries
            .iter()
            .map(|(key, entry)| (key.as_str(), *entry))
    }
}

fn collect_keys(headers: &[ShardHeader]) -> Vec<String> {
    headers
        .iter()
        .flat_map(|header| header.iter().map(|(key, _)| key.to_string()))
        .collect()
}

fn module_of(key: &str) -> Result<&str, EngError> {
    key.rsplit_once('.')
        .map(|(module, _)| module)
        .ok_or_else(|| EngError::Checkpoint(format!("weight key {key:?} has no module path")))
}

fn check_override_keys(quant: &QuantizationMap, keys: &[String]) -> Result<(), EngError> {
    for module in quant.override_modules() {
        let mapped = keys
            .iter()
            .any(|key| module_of(key).is_ok_and(|found| found == module));
        if !mapped {
            return Err(EngError::Checkpoint(format!(
                "override key {module:?} matches no checkpoint key"
            )));
        }
    }
    Ok(())
}

fn resolve_entries(
    quant: &QuantizationMap,
    keys: &[String],
    tiers: &TierAssignment,
) -> Result<BTreeMap<String, PlanEntry>, EngError> {
    let mut entries = BTreeMap::new();
    for key in keys {
        let tier = tiers
            .tier(key)
            .ok_or_else(|| EngError::Checkpoint(format!("weight key {key:?} has no tier")))?;
        let quantization = match tiers.role(key) {
            Some(KeyRole::Dense) => ModuleQuantization::Unquantized,
            _ => match quant.resolve(module_of(key)?)? {
                quantized @ ModuleQuantization::Quantized(_) => quantized,
                ModuleQuantization::Unquantized => {
                    return Err(EngError::Checkpoint(format!(
                        "weight key {key:?} has no quantization"
                    )));
                }
            },
        };
        entries.insert(key.clone(), PlanEntry { tier, quantization });
    }
    Ok(entries)
}

#[cfg(test)]
mod tests;
