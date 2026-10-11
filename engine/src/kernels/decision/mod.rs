// Decision head for small baked-weight classifiers: CPU adapter now, ANE
// adapter later. Checked weights carry their validated configuration, and the
// adapters read it from there. Callers build weights, then call predict. The
// two-layer SiLU math and the adapter dispatch stay inside.
use std::ffi::c_int;

use crate::EngError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadConfig {
    pub input: usize,
    pub hidden: usize,
    pub experts: usize,
    pub classes: usize,
}

impl HeadConfig {
    fn validate(&self) -> Result<(), EngError> {
        if self.input == 0 || self.hidden == 0 || self.experts == 0 || self.classes == 0 {
            return Err(EngError::BadRequest(
                "decision head dimensions must be positive".into(),
            ));
        }
        self.native_dimensions()?;
        Ok(())
    }

    /// The dimensions as the C int values that native adapters receive.
    /// A value above `c_int::MAX` is a request error, never a silent wrap.
    fn native_dimensions(&self) -> Result<[c_int; 4], EngError> {
        let narrow = |name: &str, value: usize| {
            c_int::try_from(value).map_err(|_| {
                EngError::BadRequest(format!(
                    "decision head {name} {value} exceeds the native limit {}",
                    c_int::MAX
                ))
            })
        };
        let dimensions = [
            narrow("input", self.input)?,
            narrow("hidden", self.hidden)?,
            narrow("experts", self.experts)?,
            narrow("classes", self.classes)?,
        ];
        narrow("outputs", self.outputs()?)?;
        Ok(dimensions)
    }

    fn outputs(&self) -> Result<usize, EngError> {
        self.experts
            .checked_add(self.classes)
            .ok_or_else(|| EngError::BadRequest("decision experts + classes overflow".into()))
    }
}

#[derive(Debug, Clone)]
pub struct HeadWeights {
    config: HeadConfig,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl HeadWeights {
    pub fn try_new(
        config: &HeadConfig,
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: Vec<f32>,
    ) -> Result<Self, EngError> {
        config.validate()?;
        let w1_len = config
            .hidden
            .checked_mul(config.input)
            .ok_or_else(|| EngError::BadRequest("decision w1 element count overflow".into()))?;
        let w2_len = config
            .outputs()?
            .checked_mul(config.hidden)
            .ok_or_else(|| EngError::BadRequest("decision w2 element count overflow".into()))?;
        if w1.len() != w1_len
            || b1.len() != config.hidden
            || w2.len() != w2_len
            || b2.len() != config.outputs()?
        {
            return Err(EngError::BadRequest(format!(
                "decision weights must be w1[{hidden},{input}] b1[{hidden}] w2[{outputs},{hidden}] b2[{outputs}], received w1={} b1={} w2={} b2={}",
                w1.len(),
                b1.len(),
                w2.len(),
                b2.len(),
                hidden = config.hidden,
                input = config.input,
                outputs = config.outputs().unwrap_or(0),
            )));
        }
        Ok(Self {
            config: *config,
            w1,
            b1,
            w2,
            b2,
        })
    }

    pub fn config(&self) -> HeadConfig {
        self.config
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub expert_logits: Vec<f32>,
    pub class_logits: Vec<f32>,
}

trait HeadBackend {
    fn predict(&self, weights: &HeadWeights, features: &[f32]) -> Result<Decision, EngError>;
}

#[cfg(all(feature = "native-ane", target_os = "macos", target_arch = "aarch64"))]
mod ane;

struct CpuBackend;

impl HeadBackend for CpuBackend {
    fn predict(&self, weights: &HeadWeights, features: &[f32]) -> Result<Decision, EngError> {
        let config = &weights.config;
        let mut hidden = vec![0.0f32; config.hidden];
        for (h, slot) in hidden.iter_mut().enumerate() {
            let mut acc = weights.b1[h];
            for (i, value) in features.iter().enumerate() {
                acc += weights.w1[h * config.input + i] * value;
            }
            *slot = acc / (1.0 + (-acc).exp());
        }
        let outputs = config.outputs()?;
        let mut logits = vec![0.0f32; outputs];
        for (o, slot) in logits.iter_mut().enumerate() {
            let mut acc = weights.b2[o];
            for (h, value) in hidden.iter().enumerate() {
                acc += weights.w2[o * config.hidden + h] * value;
            }
            *slot = acc;
        }
        let split = config.experts;
        Ok(Decision {
            expert_logits: logits[..split].to_vec(),
            class_logits: logits[split..].to_vec(),
        })
    }
}

pub struct DecisionHead {
    weights: HeadWeights,
    backend: Box<dyn HeadBackend>,
}

impl DecisionHead {
    pub fn cpu(weights: HeadWeights) -> Result<Self, EngError> {
        Ok(Self {
            weights,
            backend: Box::new(CpuBackend),
        })
    }

    #[cfg(all(feature = "native-ane", target_os = "macos", target_arch = "aarch64"))]
    pub fn ane(weights: HeadWeights) -> Result<Self, EngError> {
        Ok(Self {
            backend: Box::new(ane::AneBackend::try_new(&weights.config(), &weights)?),
            weights,
        })
    }

    pub fn predict(&self, features: &[f32]) -> Result<Decision, EngError> {
        let input = self.weights.config().input;
        if features.len() != input {
            return Err(EngError::BadRequest(format!(
                "decision features must be {input} values, received {}",
                features.len()
            )));
        }
        self.backend.predict(&self.weights, features)
    }
}
