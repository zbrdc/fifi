//! Flint block FFN: every projection consumes all T rows or all routed pairs.

mod plan;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) mod device;

#[cfg(test)]
mod qk_goldens;
#[cfg(test)]
mod real_chain;
#[cfg(test)]
mod real_model;
#[cfg(test)]
mod real_quality;
#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) fn qk_producer_oracle(
    q: &[f32],
    k: &[f32],
    gamma: [&[f32]; 2],
    position: Option<u32>,
) -> Result<[Vec<f32>; 2], String> {
    reference::real::qk_producer_oracle(q, k, gamma, position)
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
use device::decode;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod decode_tests;
