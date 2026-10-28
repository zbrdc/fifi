use super::expert_sources;
use crate::{EngError, weights::GgufHeader};

#[derive(Clone)]
struct Entry {
    name: &'static str,
    dims: Vec<u64>,
    ty: u32,
    offset: u64,
}

fn entries() -> Vec<Entry> {
    [
        ("blk.2.ffn_gate_exps.weight", vec![2048, 512, 256], 40, 0),
        ("blk.2.ffn_gate_exps.scale", vec![256], 0, 150_994_944),
        (
            "blk.2.ffn_up_exps.weight",
            vec![2048, 512, 256],
            40,
            150_995_968,
        ),
        ("blk.2.ffn_up_exps.scale", vec![256], 0, 301_990_912),
        (
            "blk.2.ffn_down_exps.weight",
            vec![512, 2048, 256],
            40,
            301_991_936,
        ),
        ("blk.2.ffn_down_exps.scale", vec![256], 0, 452_986_880),
    ]
    .into_iter()
    .map(|(name, dims, ty, offset)| Entry {
        name,
        dims,
        ty,
        offset,
    })
    .collect()
}

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn header(entries: &[Entry], architecture: &str) -> GgufHeader {
    header_file_len(entries, architecture, 452_988_928)
}

fn header_file_len(entries: &[Entry], architecture: &str, file_len: u64) -> GgufHeader {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&2u64.to_le_bytes());
    string(&mut bytes, "general.alignment");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&1024u32.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    string(&mut bytes, architecture);
    for entry in entries {
        string(&mut bytes, entry.name);
        bytes.extend_from_slice(&(entry.dims.len() as u32).to_le_bytes());
        for dimension in &entry.dims {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&entry.ty.to_le_bytes());
        bytes.extend_from_slice(&entry.offset.to_le_bytes());
    }
    assert!(bytes.len() <= 1024);
    let parsed = GgufHeader::from_bytes(&bytes, file_len, 4096).unwrap();
    assert_eq!(parsed.data_start(), 1024);
    parsed
}

#[test]
fn dense_matrix_uses_one_scalar_and_checked_output_input_geometry() {
    let mut entries = vec![
        Entry {
            name: "blk.0.ffn_gate.weight",
            dims: vec![2048, 6144],
            ty: 40,
            offset: 0,
        },
        Entry {
            name: "blk.0.ffn_gate.scale",
            dims: vec![1],
            ty: 0,
            offset: 7_077_888,
        },
    ];
    let parsed = header(&entries, "flint");
    let source = super::matrix_source(&parsed, "blk.0.ffn_gate", [6144, 2048]).unwrap();
    assert_eq!(source.data, 1024..7_078_912);
    assert_eq!(source.scale, 7_078_912..7_078_916);
    assert_eq!(source.shape, [6144, 2048]);
    assert!(super::matrix_source(&parsed, "blk.0.ffn_gate", [2048, 6144]).is_err());
    entries[1].dims = vec![256];
    let parsed = header(&entries, "flint");
    assert!(super::matrix_source(&parsed, "blk.0.ffn_gate", [6144, 2048]).is_err());
}

#[test]
fn selected_expert_has_three_distinct_payloads_and_its_own_scalars() {
    let header = header(&entries(), "flint");
    let views = expert_sources(&header, 2, 1).unwrap();
    assert_eq!(views.gate.data, 590_848..1_180_672);
    assert_eq!(views.up.data, 151_586_816..152_176_640);
    assert_eq!(views.down.data, 302_582_784..303_172_608);
    assert_eq!(views.gate.scale, 150_995_972..150_995_976);
    assert_eq!(views.up.scale, 301_991_940..301_991_944);
    assert_eq!(views.down.scale, 452_987_908..452_987_912);
    assert_eq!(views.gate.shape, [512, 2048]);
    assert_eq!(views.up.shape, [512, 2048]);
    assert_eq!(views.down.shape, [2048, 512]);
}

#[test]
fn final_expert_stays_within_payload_and_uses_the_final_scale() {
    let header = header(&entries(), "flint");
    let views = expert_sources(&header, 2, 255).unwrap();
    assert_eq!(views.gate.data, 150_406_144..150_995_968);
    assert_eq!(views.up.data, 301_402_112..301_991_936);
    assert_eq!(views.down.data, 452_398_080..452_987_904);
    assert_eq!(views.gate.scale, 150_996_988..150_996_992);
    assert_eq!(views.down.scale, 452_988_924..452_988_928);
}

#[test]
fn rejects_dense_layers_and_out_of_range_experts() {
    let header = header(&entries(), "flint");
    for (layer, expert) in [
        (0, 0),
        (1, 0),
        (49, 0),
        (u32::MAX, 0),
        (2, 256),
        (2, u32::MAX),
    ] {
        assert!(matches!(
            expert_sources(&header, layer, expert),
            Err(EngError::BadRequest(_))
        ));
    }
}

#[test]
fn rejects_missing_roles_wrong_formats_and_expert_geometry() {
    let original = entries();
    let mut missing = original.clone();
    missing.remove(1);
    let mut wrong_format = original.clone();
    wrong_format[0].ty = 2; // Q4_0 has the same byte footprint, not the same encoding.
    let mut wrong_dims = original.clone();
    wrong_dims[4].dims = vec![2048, 512, 256];
    let mut wrong_count = original.clone();
    wrong_count[0].dims[2] = 128;
    let mut wrong_scale = original.clone();
    wrong_scale[1].dims = vec![1];
    let mut wrong_scale_type = original;
    wrong_scale_type[1].ty = 30;
    for entries in [
        missing,
        wrong_format,
        wrong_dims,
        wrong_count,
        wrong_scale,
        wrong_scale_type,
    ] {
        let header = header(&entries, "flint");
        assert!(matches!(
            expert_sources(&header, 2, 1),
            Err(EngError::Checkpoint(_))
        ));
    }
}

#[test]
fn rejects_other_architecture_even_when_tensor_names_and_shapes_match() {
    let header = header(&entries(), "qwen2moe");
    assert!(matches!(
        expert_sources(&header, 2, 0),
        Err(EngError::Checkpoint(_))
    ));
}

#[test]
fn bound_matrix_borrows_the_selected_payload_and_keeps_its_scalar() {
    let header = header(&entries(), "flint");
    let sources = expert_sources(&header, 2, 1).unwrap();
    let raw = vec![0x31; 589_824];
    let view = sources.gate.bind(&raw, &2.5f32.to_le_bytes()).unwrap();
    assert!(std::ptr::eq(view.data.as_ptr(), raw.as_ptr()));
    assert_eq!(view.data.len(), 589_824);
    assert_eq!(view.shape, [512, 2048]);
    assert_eq!(view.tensor_scale.to_bits(), 2.5f32.to_bits());
}

#[test]
fn matrix_binding_rejects_bad_lengths_and_nonfinite_scalars() {
    let header = header(&entries(), "flint");
    let sources = expert_sources(&header, 2, 1).unwrap();
    let raw = vec![0; 589_824];
    for bytes in [&raw[..0], &raw[..589_823]] {
        assert!(sources.down.bind(bytes, &1f32.to_le_bytes()).is_err());
    }
    for scalar in [vec![], vec![0; 3], vec![0; 5]] {
        assert!(sources.down.bind(&raw, &scalar).is_err());
    }
    for scalar in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(sources.down.bind(&raw, &scalar.to_le_bytes()).is_err());
    }
}

#[test]
fn raw_gguf_binding_does_not_impose_the_gpu_positive_scale_envelope() {
    let header = header(&entries(), "flint");
    let sources = expert_sources(&header, 2, 1).unwrap();
    let raw = vec![0; 589_824];
    for scalar in [0f32, -0f32, -2.5f32] {
        let view = sources.down.bind(&raw, &scalar.to_le_bytes()).unwrap();
        assert_eq!(view.tensor_scale.to_bits(), scalar.to_bits());
        assert_eq!(view.shape, [2048, 512]);
    }
}

fn embedding_entry() -> Entry {
    Entry {
        name: "token_embd.weight",
        dims: vec![2048, 150144],
        ty: 30,
        offset: 0,
    }
}

#[test]
fn embedding_source_selects_direct_bf16_rows_not_the_apple_int8_layout() {
    let header = header_file_len(&[embedding_entry()], "flint", 614_990_848);
    assert_eq!(super::embedding_row(&header, 1).unwrap(), 5120..9216);
    assert_eq!(
        super::embedding_row(&header, 150_143).unwrap(),
        614_986_752..614_990_848
    );
    let raw: Vec<_> = (0..2048).flat_map(|_| 0x3fc0u16.to_le_bytes()).collect();
    let mut row = vec![0f32; 2048];
    crate::weights::dequantize_into(
        crate::weights::GgufType::from_id(30).unwrap(),
        &raw,
        &mut row,
    )
    .unwrap();
    assert!(row.iter().all(|value| *value == 1.5));
}

#[test]
fn embedding_source_rejects_out_of_vocab_ids() {
    let header = header_file_len(&[embedding_entry()], "flint", 614_990_848);
    for token in [150_144, u32::MAX] {
        assert!(matches!(
            super::embedding_row(&header, token),
            Err(EngError::BadRequest(_))
        ));
    }
}

#[test]
fn embedding_source_rejects_other_formats_and_geometry() {
    let mut int8 = embedding_entry();
    int8.ty = 24;
    let mut wrong_width = embedding_entry();
    wrong_width.dims[0] = 1024;
    let mut wrong_vocab = embedding_entry();
    wrong_vocab.dims[1] = 150143;
    for entry in [int8, wrong_width, wrong_vocab] {
        let header = header_file_len(&[entry], "flint", 614_990_848);
        assert!(matches!(
            super::embedding_row(&header, 0),
            Err(EngError::Checkpoint(_))
        ));
    }
    let header = header_file_len(&[embedding_entry()], "qwen2moe", 614_990_848);
    assert!(matches!(
        super::embedding_row(&header, 0),
        Err(EngError::Checkpoint(_))
    ));
}
