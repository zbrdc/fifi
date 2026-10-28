//! Owned pending copies can retire through actual stream completion, without an event claim.
use super::*;
use crate::kernels::cuda::{CapturePlan, device_present};

struct Observed {
    refused: bool,
    pending_preserved: bool,
    cpu_access_refused: bool,
    h2d_completed: bool,
    roundtrip: bool,
    capture_refused: bool,
    recovery: bool,
}

fn observe() -> Result<Observed, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let other = ctx.stream()?;
    let device = ctx.allocate(4192)?;
    let data: Vec<u8> = (0..4192).map(|i| (i % 251) as u8).collect();
    let mut pinned = ctx.pinned_buf(data.len())?;
    // Fresh owners have no pending copy; stream completion is still a valid no-op proof.
    pinned.synchronize(&stream)?;
    pinned.write(0, &data)?;
    ctx.copy_from_pinned(&stream, &mut pinned, 0, device.range(0, data.len())?)?;
    let before = (stream.enqueued(), other.enqueued());
    let refused = matches!(pinned.synchronize(&other), Err(EngError::BadRequest(_)))
        && before == (stream.enqueued(), other.enqueued());
    let pending_preserved = pinned.is_pending();
    let mut got = vec![0; data.len()];
    let cpu_access_refused = matches!(pinned.read(0, &mut got), Err(EngError::BadRequest(_)))
        && matches!(pinned.write(0, &data), Err(EngError::BadRequest(_)));
    pinned.synchronize(&stream)?;
    let h2d_completed = !pinned.is_pending();
    pinned.write(0, &vec![0; data.len()])?;
    ctx.copy_to_pinned(&stream, device.range(0, data.len())?, &mut pinned, 0)?;
    pinned.synchronize(&stream)?;
    pinned.read(0, &mut got)?;
    let roundtrip = got == data && !pinned.is_pending();
    let mut capture_refused = false;
    let pins = [&device];
    let result = ctx.capture(
        &stream,
        CapturePlan {
            pins: &pins,
            holds: Vec::new(),
        },
        || {
            capture_refused = matches!(pinned.synchronize(&stream), Err(EngError::BadRequest(_)));
            Err::<(), _>(EngError::BadRequest(
                "end pinned completion probe".to_owned(),
            ))
        },
    );
    match result {
        Err(EngError::BadRequest(text)) if text == "end pinned completion probe" => {}
        Err(error) => return Err(error),
        Ok(_) => {
            return Err(EngError::BadRequest(
                "unexpected completion-probe graph".to_owned(),
            ));
        }
    }
    pinned.synchronize(&stream)?;
    pinned.read(0, &mut got)?;
    let recovery = got == data;
    Ok(Observed {
        refused,
        pending_preserved,
        cpu_access_refused,
        h2d_completed,
        roundtrip,
        capture_refused,
        recovery,
    })
}

#[test]
fn stream_completion_retires_owned_pins_without_forging_events() {
    if !device_present("stream_completion_retires_owned_pins_without_forging_events") {
        return;
    }
    let x = observe().expect("actual owned stream-completion proof");
    assert!(x.refused && x.pending_preserved && x.cpu_access_refused);
    assert!(x.h2d_completed && x.roundtrip && x.capture_refused && x.recovery);
    eprintln!(
        "PINNED_STREAM_SYNC wrong_stream=true pending_preserved=true roundtrip=true capture=true recovery=true"
    );
}
