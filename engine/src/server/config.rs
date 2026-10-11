//! The `enginesrv` configuration file and command line.
//!
//! Decision 0026 fixes the file: one JSON object with the listen address, the name of the
//! environment variable that holds the API key, and an `engines` array. The file carries no key.

use crate::family::flash_next::{factory::FAMILY as FLASH_FAMILY, settings::FlashOptions};
use crate::family::qwen36::{factory::FAMILY as Q36_FAMILY, settings::Q36Options};
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

/// A startup failure. The text names the cause and, where it applies, the file or variable.
#[derive(Debug, PartialEq, Eq)]
pub struct StartupError(pub(crate) String);

impl std::fmt::Display for StartupError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for StartupError {}

/// The family names that this build can start.
const FAMILIES: [&str; 3] = [FAKE_FAMILY, FLASH_FAMILY, Q36_FAMILY];

/// The family that takes the `context_limit` and `fake` members.
const FAKE_FAMILY: &str = "fake";

/// The port of another service on the same Mac. The server never binds it.
const REFUSED_PORT: u16 = 8788;

/// The whole configuration file.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    /// The address to listen on, for example `127.0.0.1:8100`.
    pub(crate) listen: String,
    /// The name of the environment variable that holds the bearer key.
    pub(crate) api_key_env: String,
    /// True when the chat handler marks the last message boundary of each prompt, so that an
    /// engine can save and restore a prefix state (decision 0036, addendum 1). The default is
    /// false: the handler then sends no boundary, and no engine saves or restores. One value
    /// serves every engine of the process. An engine without prefix support ignores it.
    #[serde(default)]
    pub(crate) prefix_reuse: bool,
    pub(crate) engines: Vec<EngineConfig>,
}

/// One engine of the `engines` array.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EngineConfig {
    pub(crate) id: String,
    pub(crate) model_dir: PathBuf,
    pub(crate) tokenizer_dir: PathBuf,
    pub(crate) family: String,
    /// The model name for the `model` field. The default is the family name.
    pub(crate) model: Option<String>,
    /// The context limit of the `fake` family. A real family reports its own limit when it
    /// starts, and the file does not set it.
    pub(crate) context_limit: Option<u32>,
    /// Options of the `fake` family only.
    pub(crate) fake: Option<FakeOptions>,
    /// Options of the `qwen3_8_flash_next` family only. The family reads the manifest, the cache
    /// limit, the vocabulary size, and the prefill chunk from here (packet H8c).
    pub(crate) flash_next: Option<FlashOptions>,
    /// Options of the `qwen3_6_35b_a3b` family only. The family reads the cache limit, the context
    /// limit, the session cap, and the prefill chunk from here (packet Q36-F3c).
    pub(crate) qwen36: Option<Q36Options>,
}

/// Options of the development engine.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FakeOptions {
    /// The text that every job replies with.
    pub(crate) reply: Option<String>,
    /// The sleep before each token, in milliseconds.
    #[serde(default)]
    pub(crate) token_delay_ms: u64,
}

impl EngineConfig {
    /// The model name that clients put in the `model` field.
    pub(crate) fn model_name(&self) -> String {
        self.model.clone().unwrap_or_else(|| self.family.clone())
    }
}

impl Config {
    /// Reads and checks the file at `path`.
    pub(crate) fn from_file(path: &Path) -> Result<Config, StartupError> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| StartupError(format!("cannot read {}: {error}", path.display())))?;
        let config: Config = serde_json::from_str(&text)
            .map_err(|error| StartupError(format!("cannot parse {}: {error}", path.display())))?;
        config.check()?;
        Ok(config)
    }

    fn check(&self) -> Result<(), StartupError> {
        self.listen_address()?;
        if self.api_key_env.is_empty() {
            return Err(StartupError("api_key_env must name a variable".to_string()));
        }
        if self.engines.is_empty() {
            return Err(StartupError(
                "engines must list at least one engine".to_string(),
            ));
        }
        for (index, engine) in self.engines.iter().enumerate() {
            if engine.id.is_empty() {
                return Err(StartupError(format!(
                    "engines[{index}].id must not be empty"
                )));
            }
            if engine.model.as_deref() == Some("") {
                return Err(StartupError(format!(
                    "engines[{index}].model must not be empty"
                )));
            }
            if engine.model_dir.as_os_str().is_empty() {
                return Err(StartupError(format!(
                    "engines[{index}].model_dir must not be empty"
                )));
            }
            if self.engines[..index]
                .iter()
                .any(|other| other.id == engine.id)
            {
                return Err(StartupError(format!(
                    "engine id {} is used twice",
                    engine.id
                )));
            }
            if engine.context_limit == Some(0) {
                return Err(StartupError(format!(
                    "engines[{index}].context_limit must be greater than 0"
                )));
            }
            if engine.family != FAKE_FAMILY
                && (engine.context_limit.is_some() || engine.fake.is_some())
            {
                return Err(StartupError(format!(
                    "engines[{index}] sets context_limit or fake, which only the {FAKE_FAMILY} family takes"
                )));
            }
            if engine.family != FLASH_FAMILY && engine.flash_next.is_some() {
                return Err(StartupError(format!(
                    "engines[{index}] sets flash_next, which only the {FLASH_FAMILY} family takes"
                )));
            }
            if engine.family != Q36_FAMILY && engine.qwen36.is_some() {
                return Err(StartupError(format!(
                    "engines[{index}] sets qwen36, which only the {Q36_FAMILY} family takes"
                )));
            }
            if !FAMILIES.contains(&engine.family.as_str()) {
                return Err(StartupError(format!(
                    "engine {} has family {}, which this build does not support (supported: {})",
                    engine.id,
                    engine.family,
                    FAMILIES.join(", ")
                )));
            }
        }
        Ok(())
    }

    /// The parsed listen address. It must be a loopback address, and its port must not be 8788.
    pub(crate) fn listen_address(&self) -> Result<SocketAddr, StartupError> {
        let address: SocketAddr = self.listen.parse().map_err(|error| {
            StartupError(format!("listen {} is not an address: {error}", self.listen))
        })?;
        if !address.ip().is_loopback() {
            return Err(StartupError(format!(
                "listen {} must be a loopback address (127.0.0.1 or ::1)",
                self.listen
            )));
        }
        if address.port() == REFUSED_PORT {
            return Err(StartupError(format!(
                "listen {} uses port {REFUSED_PORT}, which belongs to another service",
                self.listen
            )));
        }
        Ok(address)
    }

    /// Reads the bearer key from the variable that the file names. `lookup` reads one variable.
    /// A missing or empty variable stops startup. The error never shows the value.
    pub(crate) fn api_key(
        &self,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<String, StartupError> {
        match lookup(&self.api_key_env) {
            Some(key) if !key.is_empty() => Ok(key),
            _ => Err(StartupError(format!(
                "environment variable {} is not set or is empty",
                self.api_key_env
            ))),
        }
    }
}

/// Reads the path after `--config` from the arguments that follow the program name.
pub(crate) fn config_path(args: &[String]) -> Result<PathBuf, StartupError> {
    match args {
        [flag, path] if flag == "--config" => Ok(PathBuf::from(path)),
        _ => Err(StartupError("usage: enginesrv --config <path>".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, FakeOptions, StartupError, config_path};
    use std::path::{Path, PathBuf};

    fn write(dir: &Path, text: &str) -> PathBuf {
        let path = dir.join("enginesrv.json");
        std::fs::write(&path, text).expect("write config");
        path
    }

    fn load(text: &str) -> Result<Config, StartupError> {
        let dir = tempfile::tempdir().expect("tempdir");
        Config::from_file(&write(dir.path(), text))
    }

    fn message(result: Result<Config, StartupError>) -> String {
        result.expect_err("must fail").to_string()
    }

    const VALID: &str = r#"{
        "listen": "127.0.0.1:8100",
        "api_key_env": "ENGINESRV_KEY",
        "engines": [
            {"id": "e1", "model_dir": "m", "tokenizer_dir": "t", "family": "fake"}
        ]
    }"#;

    #[test]
    fn a_valid_file_loads_with_the_defaults() {
        let config = load(VALID).expect("valid");
        assert_eq!(config.listen_address().expect("address").port(), 8100);
        assert_eq!(config.engines[0].model_name(), "fake");
        assert_eq!(config.engines[0].context_limit, None);
        assert!(config.engines[0].fake.is_none());
    }

    #[test]
    fn prefix_reuse_is_off_by_default_and_reads_a_boolean() {
        assert!(!load(VALID).expect("valid").prefix_reuse, "the default");
        let on = VALID.replace(r#""listen""#, r#""prefix_reuse": true, "listen""#);
        assert!(load(&on).expect("valid").prefix_reuse, "true");
        let off = VALID.replace(r#""listen""#, r#""prefix_reuse": false, "listen""#);
        assert!(!load(&off).expect("valid").prefix_reuse, "false");
        let text = VALID.replace(r#""listen""#, r#""prefix_reuse": "yes", "listen""#);
        assert!(
            message(load(&text)).contains("expected a boolean"),
            "a string"
        );
        let number = VALID.replace(r#""listen""#, r#""prefix_reuse": 1, "listen""#);
        assert!(
            message(load(&number)).contains("expected a boolean"),
            "a number"
        );
    }

    #[test]
    fn model_context_limit_and_fake_options_are_read() {
        let text = VALID.replace(
            r#""family": "fake""#,
            r#""family": "fake", "model": "demo", "context_limit": 99,
               "fake": {"reply": "Hi.", "token_delay_ms": 5}"#,
        );
        let config = load(&text).expect("valid");
        let engine = &config.engines[0];
        assert_eq!(engine.model_name(), "demo");
        assert_eq!(engine.context_limit, Some(99));
        let options: &FakeOptions = engine.fake.as_ref().expect("options");
        assert_eq!(options.reply.as_deref(), Some("Hi."));
        assert_eq!(options.token_delay_ms, 5);
    }

    #[test]
    fn a_missing_file_names_the_path() {
        let error = Config::from_file(Path::new("/nonexistent/enginesrv.json")).expect_err("error");
        assert!(
            error.to_string().contains("/nonexistent/enginesrv.json"),
            "{error}"
        );
    }

    #[test]
    fn a_parse_error_unknown_member_or_missing_member_stops_startup() {
        assert!(message(load("{")).contains("cannot parse"));
        let unknown = VALID.replace(r#""listen""#, r#""port": 1, "listen""#);
        assert!(message(load(&unknown)).contains("unknown field `port`"));
        let missing = VALID.replace(r#""api_key_env": "ENGINESRV_KEY","#, "");
        assert!(message(load(&missing)).contains("api_key_env"));
    }

    #[test]
    fn a_listen_value_that_is_not_an_address_is_rejected() {
        let text = VALID.replace("127.0.0.1:8100", "localhost");
        assert!(message(load(&text)).contains("listen localhost"));
    }

    #[test]
    fn a_listen_address_that_is_not_loopback_is_rejected() {
        for address in [
            "0.0.0.0:8100",
            "192.0.2.5:8100",
            "[::]:8100",
            "[2001:db8::1]:8100",
        ] {
            let text = VALID.replace("127.0.0.1:8100", address);
            let error = message(load(&text));
            assert!(
                error.contains(&format!("listen {address}")) && error.contains("loopback"),
                "{error}"
            );
        }
    }

    #[test]
    fn loopback_addresses_of_both_families_are_accepted() {
        for address in ["127.0.0.1:8100", "[::1]:8100"] {
            let text = VALID.replace("127.0.0.1:8100", address);
            let config = load(&text).expect(address);
            assert_eq!(config.listen_address().expect("address").port(), 8100);
        }
    }

    #[test]
    fn port_8788_is_refused_on_every_loopback_address() {
        for address in ["127.0.0.1:8788", "[::1]:8788"] {
            let text = VALID.replace("127.0.0.1:8100", address);
            let error = message(load(&text));
            assert!(
                error.contains(&format!("listen {address}")) && error.contains("8788"),
                "{error}"
            );
        }
    }

    #[test]
    fn a_context_limit_of_zero_is_rejected() {
        let text = VALID.replace(
            r#""family": "fake""#,
            r#""family": "fake", "context_limit": 0"#,
        );
        assert!(message(load(&text)).contains("engines[0].context_limit"));
    }

    #[test]
    fn context_limit_and_fake_options_on_another_family_are_rejected() {
        for member in [r#""context_limit": 99"#, r#""fake": {}"#] {
            let text = VALID.replace(
                r#""family": "fake""#,
                &format!(r#""family": "flash-next", {member}"#),
            );
            let error = message(load(&text));
            assert!(
                error.contains("engines[0]") && error.contains("only the fake family"),
                "{error}"
            );
        }
    }

    #[test]
    fn an_empty_model_name_is_rejected() {
        let text = VALID.replace(r#""family": "fake""#, r#""family": "fake", "model": """#);
        assert!(message(load(&text)).contains("engines[0].model must not be empty"));
    }

    #[test]
    fn empty_engine_lists_empty_ids_and_repeated_ids_are_rejected() {
        let none = r#"{"listen": "127.0.0.1:1", "api_key_env": "K", "engines": []}"#;
        assert!(message(load(none)).contains("at least one engine"));
        let empty_id = VALID.replace(r#""id": "e1""#, r#""id": """#);
        assert!(message(load(&empty_id)).contains("engines[0].id"));
        let entry = r#"{"id": "e1", "model_dir": "m", "tokenizer_dir": "t", "family": "fake"}"#;
        let twice = VALID.replace(entry, &format!("{entry}, {entry}"));
        assert!(message(load(&twice)).contains("e1 is used twice"));
    }

    #[test]
    fn an_empty_model_directory_is_rejected() {
        let text = VALID.replace(r#""model_dir": "m""#, r#""model_dir": """#);
        assert!(message(load(&text)).contains("engines[0].model_dir"));
    }

    #[test]
    fn an_unsupported_family_names_the_family() {
        let text = VALID.replace(r#""family": "fake""#, r#""family": "flash-next""#);
        let error = message(load(&text));
        assert!(
            error.contains("flash-next") && error.contains("does not support"),
            "{error}"
        );
    }

    const FLASH: &str = r#""family": "qwen3_8_flash_next""#;

    #[test]
    fn a_flash_next_entry_reads_its_object_and_names_the_model_after_the_family() {
        let text = VALID.replace(
            r#""family": "fake""#,
            &format!(
                r#"{FLASH}, "flash_next": {{"manifest": "/m.json", "cache_limit_bytes": 1024, "prefill_chunk": 4}}"#
            ),
        );
        let config = load(&text).expect("valid");
        let engine = &config.engines[0];
        assert_eq!(engine.model_name(), "qwen3_8_flash_next");
        let options = engine.flash_next.as_ref().expect("options");
        assert_eq!(options.manifest.as_deref(), Some(Path::new("/m.json")));
        assert_eq!(options.cache_limit_bytes, Some(1024));
        assert_eq!(options.prefill_chunk, Some(4));
        assert_eq!(options.tokenizer_vocab_size, None);
    }

    #[test]
    fn a_flash_next_entry_may_leave_out_the_object() {
        let text = VALID.replace(r#""family": "fake""#, FLASH);
        assert!(load(&text).expect("valid").engines[0].flash_next.is_none());
    }

    #[test]
    fn an_unknown_member_of_the_flash_next_object_is_rejected() {
        let text = VALID.replace(
            r#""family": "fake""#,
            &format!(r#"{FLASH}, "flash_next": {{"context_limit": 9}}"#),
        );
        assert!(message(load(&text)).contains("unknown field `context_limit`"));
    }

    #[test]
    fn the_flash_next_object_on_another_family_is_rejected() {
        let text = VALID.replace(
            r#""family": "fake""#,
            r#""family": "fake", "flash_next": {}"#,
        );
        let error = message(load(&text));
        assert!(
            error.contains("engines[0]") && error.contains("only the qwen3_8_flash_next family"),
            "{error}"
        );
    }

    #[test]
    fn context_limit_and_fake_options_on_the_flash_family_are_rejected() {
        for member in [r#""context_limit": 99"#, r#""fake": {}"#] {
            let text = VALID.replace(r#""family": "fake""#, &format!("{FLASH}, {member}"));
            let error = message(load(&text));
            assert!(
                error.contains("engines[0]") && error.contains("only the fake family"),
                "{error}"
            );
        }
    }

    const Q36: &str = r#""family": "qwen3_6_35b_a3b""#;

    #[test]
    fn a_q36_entry_reads_its_object_and_names_the_model_after_the_family() {
        let text = VALID.replace(
            r#""family": "fake""#,
            &format!(
                r#"{Q36}, "qwen36": {{"cache_limit_bytes": 1024, "context_limit": 4096, "session_cap_bytes": 99, "prefill_chunk": 4}}"#
            ),
        );
        let config = load(&text).expect("valid");
        let engine = &config.engines[0];
        assert_eq!(engine.model_name(), "qwen3_6_35b_a3b");
        let options = engine.qwen36.as_ref().expect("options");
        assert_eq!(options.cache_limit_bytes, Some(1024));
        assert_eq!(options.context_limit, Some(4096));
        assert_eq!(options.session_cap_bytes, Some(99));
        assert_eq!(options.prefill_chunk, Some(4));
        assert!(engine.flash_next.is_none());
    }

    #[test]
    fn a_q36_entry_may_leave_out_the_object() {
        let text = VALID.replace(r#""family": "fake""#, Q36);
        assert!(load(&text).expect("valid").engines[0].qwen36.is_none());
    }

    #[test]
    fn an_unknown_member_of_the_qwen36_object_is_rejected() {
        let text = VALID.replace(
            r#""family": "fake""#,
            &format!(r#"{Q36}, "qwen36": {{"manifest": "/m.json"}}"#),
        );
        assert!(message(load(&text)).contains("unknown field `manifest`"));
    }

    #[test]
    fn the_qwen36_object_on_another_family_is_rejected() {
        for family in [r#""family": "fake""#, FLASH] {
            let text = VALID.replace(
                r#""family": "fake""#,
                &format!(r#"{family}, "qwen36": {{}}"#),
            );
            let error = message(load(&text));
            assert!(
                error.contains("engines[0]") && error.contains("only the qwen3_6_35b_a3b family"),
                "{error}"
            );
        }
    }

    #[test]
    fn the_flash_next_object_on_the_q36_family_is_rejected() {
        let text = VALID.replace(
            r#""family": "fake""#,
            &format!(r#"{Q36}, "flash_next": {{}}"#),
        );
        let error = message(load(&text));
        assert!(
            error.contains("only the qwen3_8_flash_next family"),
            "{error}"
        );
    }

    #[test]
    fn context_limit_and_fake_options_at_the_top_of_a_q36_entry_are_rejected() {
        for member in [r#""context_limit": 99"#, r#""fake": {}"#] {
            let text = VALID.replace(r#""family": "fake""#, &format!("{Q36}, {member}"));
            let error = message(load(&text));
            assert!(
                error.contains("engines[0]") && error.contains("only the fake family"),
                "{error}"
            );
        }
    }

    #[test]
    fn an_empty_key_variable_name_is_rejected() {
        let text = VALID.replace("ENGINESRV_KEY", "");
        assert!(message(load(&text)).contains("api_key_env"));
    }

    #[test]
    fn the_api_key_comes_from_the_variable_that_the_file_names() {
        let config = load(VALID).expect("valid");
        let key = config.api_key(|name| (name == "ENGINESRV_KEY").then(|| "sk-secret".to_string()));
        assert_eq!(key, Ok("sk-secret".to_string()));
    }

    #[test]
    fn a_missing_or_empty_key_stops_startup_and_the_error_names_only_the_variable() {
        let config = load(VALID).expect("valid");
        let missing = config.api_key(|_| None).expect_err("missing");
        assert!(missing.to_string().contains("ENGINESRV_KEY"), "{missing}");
        let empty = config.api_key(|_| Some(String::new())).expect_err("empty");
        assert!(empty.to_string().contains("ENGINESRV_KEY"), "{empty}");
        let other = config.api_key(|name| (name == "OTHER").then(|| "sk-secret".to_string()));
        assert!(
            other.is_err(),
            "a different variable must not satisfy the file"
        );
    }

    #[test]
    fn the_command_line_takes_exactly_config_and_a_path() {
        let args = |items: &[&str]| {
            items
                .iter()
                .map(|item| (*item).to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            config_path(&args(&["--config", "a.json"])),
            Ok(PathBuf::from("a.json"))
        );
        for bad in [
            args(&[]),
            args(&["--config"]),
            args(&["a.json"]),
            args(&["--config", "a", "b"]),
            args(&["--conf", "a"]),
        ] {
            let error = config_path(&bad).expect_err("usage error");
            assert!(error.to_string().contains("usage"), "{error}");
        }
    }
}
