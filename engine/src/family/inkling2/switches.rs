//! The optional CUDA paths of Inkling 2 text serving, each with its environment name and default
//! (see [`crate::family::switch_table`] for how a value is read). The defaults are the set
//! measured fastest on the RTX 3090 with both published GGUF layouts (UD-IQ1_M and an IQ4_NL
//! fine-tune), with the same greedy text as the baseline paths. The diagnostic paths stay off.
crate::family::switch_table::switch_table! {
    #[cfg_attr(
        not(all(feature = "native-cuda", target_os = "linux")),
        allow(dead_code, reason = "most paths exist only in CUDA builds")
    )]
    pub(crate) enum Switch {
        AllocCache = "FIFI_INKLING_CUDA_ALLOC_CACHE", true;
        BatchAttentionProjections = "FIFI_INKLING_CUDA_BATCH_ATTENTION_PROJECTIONS", true;
        BatchControllers = "FIFI_INKLING_CUDA_BATCH_CONTROLLERS", true;
        BatchExperts = "FIFI_INKLING_CUDA_BATCH_EXPERTS", true;
        BatchRelativeAttention = "FIFI_INKLING_CUDA_BATCH_RELATIVE_ATTENTION", true;
        BatchRouter = "FIFI_INKLING_CUDA_BATCH_ROUTER", true;
        BatchStreamedExperts = "FIFI_INKLING_CUDA_BATCH_STREAMED_EXPERTS", false;
        CacheExperts = "FIFI_INKLING_CUDA_CACHE_EXPERTS", true;
        ChainAttentionInput = "FIFI_INKLING_CUDA_CHAIN_ATTENTION_INPUT", true;
        ChainAttentionOutput = "FIFI_INKLING_CUDA_CHAIN_ATTENTION_OUTPUT", true;
        ChainLocalDecode = "FIFI_INKLING_CUDA_CHAIN_LOCAL_DECODE", true;
        ChainMlpExperts = "FIFI_INKLING_CUDA_CHAIN_MLP_EXPERTS", true;
        DecodeExactQ2Down = "FIFI_INKLING_CUDA_DECODE_EXACT_Q2_DOWN", false;
        DecodeSplit = "FIFI_INKLING_CUDA_DECODE_SPLIT", true;
        DenseStream = "FIFI_INKLING_CUDA_DENSE_STREAM", true;
        DeviceExpertPrefill = "FIFI_INKLING_CUDA_DEVICE_EXPERT_PREFILL", true;
        DeviceExpertPrefillExactTile = "FIFI_INKLING_CUDA_DEVICE_EXPERT_PREFILL_EXACT_TILE", true;
        DeviceExpertPrefillGroupedDq = "FIFI_INKLING_CUDA_DEVICE_EXPERT_PREFILL_GROUPED_DQ", true;
        DeviceExpertPrefillWmma = "FIFI_INKLING_CUDA_DEVICE_EXPERT_PREFILL_WMMA", true;
        DeviceRouting = "FIFI_INKLING_CUDA_DEVICE_ROUTING", true;
        DirectResident = "FIFI_INKLING_CUDA_DIRECT_RESIDENT", true;
        GateDot = "FIFI_INKLING_CUDA_GATE_DOT", true;
        GroupAttention = "FIFI_INKLING_CUDA_GROUP_ATTENTION", true;
        GroupProjections = "FIFI_INKLING_CUDA_GROUP_PROJECTIONS", true;
        GroupedChainAttention = "FIFI_INKLING_CUDA_GROUPED_CHAIN_ATTENTION", true;
        GroupedVerify = "FIFI_INKLING_CUDA_GROUPED_VERIFY", false;
        Iq2Gate = "FIFI_INKLING_CUDA_IQ2_GATE", false;
        MultirowControllers = "FIFI_INKLING_CUDA_MULTIROW_CONTROLLERS", true;
        PackedGguf = "FIFI_INKLING_CUDA_PACKED_GGUF", true;
        PrefillAttentionSide = "FIFI_INKLING_CUDA_PREFILL_ATTENTION_SIDE", true;
        PrefillTensorCoreProjections = "FIFI_INKLING_CUDA_PREFILL_TENSOR_CORE_PROJECTIONS", false;
        PrefillTensorCoreExperts = "FIFI_INKLING_CUDA_PREFILL_TENSOR_CORE_EXPERTS", false;
        PrefillControllerWorkspace = "FIFI_INKLING_CUDA_PREFILL_CONTROLLER_WORKSPACE", true;
        PrefillOutputConv = "FIFI_INKLING_CUDA_PREFILL_OUTPUT_CONV", true;
        PrefillQ6RowReuse = "FIFI_INKLING_CUDA_PREFILL_Q6_ROW_REUSE", true;
        PrefillQ8OutputExact4 = "FIFI_INKLING_CUDA_PREFILL_Q8_OUTPUT_EXACT4", true;
        PrefillQ8RowReuse = "FIFI_INKLING_CUDA_PREFILL_Q8_ROW_REUSE", true;
        PrefillQ8RowReuseReport = "FIFI_INKLING_CUDA_PREFILL_Q8_ROW_REUSE_REPORT", false;
        PrefillQ8Visit = "FIFI_INKLING_CUDA_PREFILL_Q8_VISIT", true;
        PrefillQueryTiles = "FIFI_INKLING_CUDA_PREFILL_QUERY_TILES", true;
        PrefillRebuildGates = "FIFI_INKLING_CUDA_PREFILL_REBUILD_GATES", true;
        PrefillResidentExpertConv = "FIFI_INKLING_CUDA_PREFILL_RESIDENT_EXPERT_CONV", true;
        PrefillKvPrep = "FIFI_INKLING_CUDA_PREFILL_KV_PREP", true;
        PrefillDevicePrompt = "FIFI_INKLING_CUDA_PREFILL_DEVICE_PROMPT", true;
        PrefillDeviceBanked = "FIFI_INKLING_CUDA_PREFILL_DEVICE_BANKED", true;
        PrefillResidentQuery = "FIFI_INKLING_CUDA_PREFILL_RESIDENT_QUERY", true;
        PrefillReuseExpertStage = "FIFI_INKLING_CUDA_PREFILL_REUSE_EXPERT_STAGE", true;
        PrefillRouterInputReuse = "FIFI_INKLING_CUDA_PREFILL_ROUTER_INPUT_REUSE", true;
        PrefillWeightMlp = "FIFI_INKLING_CUDA_PREFILL_WEIGHT_MLP", true;
        PrefillWeightPanels = "FIFI_INKLING_CUDA_PREFILL_WEIGHT_PANELS", true;
        PrefillWeightWide = "FIFI_INKLING_CUDA_PREFILL_WEIGHT_WIDE", true;
        Profile = "FIFI_INKLING_CUDA_PROFILE", false;
        Q8Activations = "FIFI_INKLING_CUDA_Q8_ACTIVATIONS", false;
        ResidentAttention = "FIFI_INKLING_CUDA_RESIDENT_ATTENTION", true;
        ResidentAttentionReserve = "FIFI_INKLING_CUDA_RESIDENT_ATTENTION_RESERVE", false;
        ResidentControllers = "FIFI_INKLING_CUDA_RESIDENT_CONTROLLERS", true;
        ResidentPrefill = "FIFI_INKLING_CUDA_RESIDENT_PREFILL", true;
        ResidentPrefillMlp = "FIFI_INKLING_CUDA_RESIDENT_PREFILL_MLP", true;
        ResidentToken = "FIFI_INKLING_CUDA_RESIDENT_TOKEN", true;
        ResidentVisit = "FIFI_INKLING_CUDA_RESIDENT_VISIT", true;
        StagePrefillControllers = "FIFI_INKLING_CUDA_STAGE_PREFILL_CONTROLLERS", true;
        StreamExperts = "FIFI_INKLING_CUDA_STREAM_EXPERTS", true;
        VocabSoftcap = "FIFI_INKLING_CUDA_VOCAB_SOFTCAP", true;
    }
}

#[cfg(test)]
mod tests {
    use super::{Switch, TABLE};

    #[test]
    fn every_switch_has_its_own_row_and_name() {
        for (index, &(switch, name, _)) in TABLE.iter().enumerate() {
            assert_eq!(switch as usize, index, "{switch:?} is out of order");
            assert_eq!(switch.name(), name);
            assert!(name.starts_with("FIFI_INKLING_CUDA_"), "{name}");
        }
        let mut names: Vec<_> = TABLE.iter().map(|row| row.1).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), TABLE.len());
    }

    #[test]
    fn the_diagnostic_paths_are_off_by_default() {
        for switch in [Switch::Profile, Switch::PrefillQ8RowReuseReport] {
            assert!(!TABLE[switch as usize].2, "{switch:?}");
        }
    }
}
