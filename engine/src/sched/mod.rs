use std::{cell::Cell, rc::Rc};

use crate::EngError;

#[allow(
    dead_code,
    reason = "H4 wires the server to the engine handle; until then only tests call the job types"
)]
pub(crate) mod job;
#[allow(
    dead_code,
    reason = "H4 wires the server to the engine handle; until then only tests call the engine worker"
)]
pub(crate) mod worker;

pub struct SessionBudget {
    cap: u64,
    used: Rc<Cell<u64>>,
}

impl SessionBudget {
    pub fn new(cap: u64) -> Self {
        Self {
            cap,
            used: Rc::new(Cell::new(0)),
        }
    }

    pub fn used(&self) -> u64 {
        self.used.get()
    }

    pub fn acquire_growth(&mut self, bytes: u64) -> Result<GrowthPermit, EngError> {
        let next = self
            .used()
            .checked_add(bytes)
            .ok_or_else(|| EngError::MemoryPlan("session growth overflows u64".into()))?;
        if next > self.cap {
            return Err(EngError::MemoryPlan(format!(
                "session growth requires {bytes} bytes, available {}",
                self.cap.saturating_sub(self.used())
            )));
        }
        self.used.set(next);
        Ok(GrowthPermit {
            used: Rc::clone(&self.used),
            bytes,
        })
    }
}

pub struct GrowthPermit {
    used: Rc<Cell<u64>>,
    bytes: u64,
}

impl GrowthPermit {
    pub fn retained_bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for GrowthPermit {
    fn drop(&mut self) {
        self.used.set(self.used.get() - self.bytes);
    }
}
