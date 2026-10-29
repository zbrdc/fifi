//! Files of the Flash-Next artifact directory that the server loads.
//!
//! The artifact manifest (`download-manifest.json`) maps each file name to `{"sha256", "size"}`.
//! The manifest is the authority for the tokenizer files. `TokenizerBundle::load` checks each file
//! against the pin that `tokenizer_pin` builds, so a stepping-stone template cannot load by mistake.

use crate::{EngError, tokenizer::TokenizerPin};
use serde_json::Value;

fn manifest_error(message: String) -> EngError {
    EngError::Checkpoint(format!("artifact manifest: {message}"))
}

/// Reads the SHA-256 of `name` from the parsed manifest.
fn digest(manifest: &Value, name: &str) -> Result<[u8; 32], EngError> {
    let text = manifest
        .get(name)
        .and_then(|entry| entry.get("sha256"))
        .and_then(Value::as_str)
        .ok_or_else(|| manifest_error(format!("no sha256 for {name}")))?;
    if text.len() != 64 {
        return Err(manifest_error(format!("sha256 of {name} is not 64 digits")));
    }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(text.as_bytes().as_chunks::<2>().0) {
        let digits = std::str::from_utf8(pair.as_slice())
            .ok()
            .and_then(|digits| u8::from_str_radix(digits, 16).ok())
            .ok_or_else(|| manifest_error(format!("sha256 of {name} is not hexadecimal")))?;
        *slot = digits;
    }
    Ok(out)
}

/// Builds the tokenizer pin from the artifact manifest. `vocab_size` is the vocabulary size of
/// `tokenizer.json` with its added tokens (248,077 for the pinned Flash-Next artifact).
pub(crate) fn tokenizer_pin(manifest: &[u8], vocab_size: u32) -> Result<TokenizerPin, EngError> {
    let value: Value =
        serde_json::from_slice(manifest).map_err(|error| manifest_error(error.to_string()))?;
    Ok(TokenizerPin {
        tokenizer_json: digest(&value, "tokenizer.json")?,
        tokenizer_config_json: digest(&value, "tokenizer_config.json")?,
        chat_template_jinja: digest(&value, "chat_template.jinja")?,
        generation_config_json: digest(&value, "generation_config.json")?,
        vocab_size,
    })
}

#[cfg(test)]
mod tests {
    use super::tokenizer_pin;
    use crate::EngError;

    /// The four tokenizer entries of the Mac artifact manifest, with one unrelated entry.
    const MANIFEST: &str = r#"{
        "config.json": {"sha256": "319b334a1abb705acf06035aa0331bcb7c25976ff93a5035b543548738a10824", "size": 183155},
        "tokenizer.json": {"sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3", "size": 12809320},
        "tokenizer_config.json": {"sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27", "size": 17928},
        "chat_template.jinja": {"sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041", "size": 8952},
        "generation_config.json": {"sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e", "size": 202}
    }"#;

    #[test]
    fn the_pin_holds_the_manifest_hash_of_each_tokenizer_file() {
        let pin = tokenizer_pin(MANIFEST.as_bytes(), 248_077).unwrap();
        assert_eq!(pin.vocab_size, 248_077);
        assert_eq!(pin.tokenizer_json[..4], [0x09, 0x97, 0xf4, 0x10]);
        assert_eq!(pin.tokenizer_json[28..], [0x72, 0x29, 0xb9, 0xf3]);
        assert_eq!(pin.tokenizer_config_json[..4], [0xb1, 0x13, 0x49, 0xaa]);
        assert_eq!(pin.chat_template_jinja[..4], [0xc3, 0xcf, 0x9e, 0x34]);
        assert_eq!(pin.generation_config_json[..4], [0xe7, 0x0c, 0x13, 0x6c]);
        assert_eq!(pin.generation_config_json[28..], [0x70, 0xbe, 0x55, 0x0e]);
    }

    #[test]
    fn a_missing_tokenizer_entry_names_the_file() {
        let manifest = MANIFEST.replace("chat_template.jinja", "chat_template.other");
        let error = tokenizer_pin(manifest.as_bytes(), 1)
            .err()
            .expect("the manifest must be rejected");
        assert_eq!(
            error,
            EngError::Checkpoint("artifact manifest: no sha256 for chat_template.jinja".into())
        );
    }

    #[test]
    fn a_short_or_non_hex_hash_is_rejected() {
        let short = MANIFEST.replace("c3cf9e34abf4", "c3cf9e34ab");
        let error = tokenizer_pin(short.as_bytes(), 1)
            .err()
            .expect("the manifest must be rejected");
        assert_eq!(
            error,
            EngError::Checkpoint(
                "artifact manifest: sha256 of chat_template.jinja is not 64 digits".into()
            )
        );
        let bad = MANIFEST.replace("c3cf9e34abf4", "c3cf9e34abzz");
        let error = tokenizer_pin(bad.as_bytes(), 1)
            .err()
            .expect("the manifest must be rejected");
        assert_eq!(
            error,
            EngError::Checkpoint(
                "artifact manifest: sha256 of chat_template.jinja is not hexadecimal".into()
            )
        );
    }

    #[test]
    fn text_that_is_not_json_is_rejected() {
        let error = tokenizer_pin(b"not json", 1)
            .err()
            .expect("the manifest must be rejected");
        assert!(
            matches!(&error, EngError::Checkpoint(text) if text.starts_with("artifact manifest: ")),
            "{error:?}"
        );
    }
}
