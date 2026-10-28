use std::collections::BTreeMap;

use serde_json::Value;

use crate::{
    EngError,
    checkpoint_json::UniqueValue,
    kernels::{QuantizationMode, QuantizationSpec},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleQuantization {
    Unquantized,
    Quantized(QuantizationSpec),
}

#[derive(Debug, PartialEq, Eq)]
pub struct QuantizationMap {
    defaults: QuantizationSpec,
    overrides: BTreeMap<String, ModuleQuantization>,
}

impl QuantizationMap {
    pub fn from_json(config: &[u8]) -> Result<Self, EngError> {
        let config = serde_json::from_slice::<UniqueValue>(config)
            .map_err(|error| EngError::Checkpoint(format!("configuration JSON: {error}")))?
            .0;
        let primary = config
            .get("quantization")
            .map(|v| Self::parse(v, "quantization"))
            .transpose()?;
        let fallback = config
            .get("quantization_config")
            .map(|v| Self::parse(v, "quantization_config"))
            .transpose()?;
        match (primary, fallback) {
            (Some(primary), Some(fallback)) if primary != fallback => Err(EngError::Checkpoint(
                "quantization conflicts with quantization_config".into(),
            )),
            (Some(map), _) | (None, Some(map)) => Ok(map),
            (None, None) => Err(EngError::Checkpoint(
                "configuration requires quantization or quantization_config".into(),
            )),
        }
    }

    fn parse(value: &Value, context: &str) -> Result<Self, EngError> {
        if value.get("quant_method").is_some() {
            return Err(EngError::Checkpoint(format!(
                "{context}.quant_method is not MLX quantization metadata"
            )));
        }
        if value.get("mode").is_none() {
            return Err(EngError::Checkpoint(format!("{context}.mode is required")));
        }
        let defaults = parse_spec(value, context)?;
        let settings = value
            .as_object()
            .ok_or_else(|| EngError::Checkpoint(format!("{context} must be an object")))?;
        let mut overrides = BTreeMap::new();
        for (module, value) in settings {
            if matches!(module.as_str(), "mode" | "bits" | "group_size") {
                continue;
            }
            validate_module(module)?;
            let setting = if value == &Value::Bool(false) {
                ModuleQuantization::Unquantized
            } else {
                if let Some(object) = value.as_object() {
                    for field in object.keys() {
                        if !matches!(field.as_str(), "mode" | "bits" | "group_size") {
                            return Err(EngError::Checkpoint(format!(
                                "{context}.{module}: unexpected field {field:?}"
                            )));
                        }
                    }
                }
                ModuleQuantization::Quantized(parse_spec(value, &format!("{context}.{module}"))?)
            };
            overrides.insert(module.clone(), setting);
        }
        Ok(Self {
            defaults,
            overrides,
        })
    }

    pub(crate) fn defaults(&self) -> QuantizationSpec {
        self.defaults
    }

    pub(crate) fn override_modules(&self) -> impl Iterator<Item = &str> {
        self.overrides.keys().map(String::as_str)
    }

    pub fn resolve(&self, module: &str) -> Result<ModuleQuantization, EngError> {
        validate_module(module)?;
        Ok(self
            .overrides
            .get(module)
            .copied()
            .unwrap_or(ModuleQuantization::Quantized(self.defaults)))
    }
}

fn parse_spec(value: &Value, context: &str) -> Result<QuantizationSpec, EngError> {
    let mode = match value.get("mode") {
        None => QuantizationMode::Affine,
        Some(Value::String(mode)) if mode == "affine" => QuantizationMode::Affine,
        Some(Value::String(mode)) if mode == "mxfp4" => QuantizationMode::Mxfp4,
        Some(Value::String(mode)) if mode == "mxfp8" => QuantizationMode::Mxfp8,
        _ => {
            return Err(EngError::Checkpoint(format!(
                "{context}.mode is unsupported"
            )));
        }
    };
    let integer = |field: &str| {
        value[field]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| EngError::Checkpoint(format!("{context}.{field} must be a u32 integer")))
    };
    let spec = QuantizationSpec {
        mode,
        bits: integer("bits")?,
        group_size: integer("group_size")?,
    };
    spec.validate().map_err(|error| match error {
        EngError::Checkpoint(message) => EngError::Checkpoint(format!("{context}: {message}")),
        other => other,
    })?;
    Ok(spec)
}

fn validate_module(module: &str) -> Result<(), EngError> {
    if module.split('.').any(|segment| {
        segment.is_empty()
            || !segment
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    }) {
        return Err(EngError::Checkpoint(format!(
            "invalid module path {module:?}"
        )));
    }
    Ok(())
}
