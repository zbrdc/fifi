//! Flash-Next GGUF supplies its two hyperconnection sites to the shared visit walk.

use std::{ops::Range, time::Instant};

use super::{FlashStep, MoeBlock, Site, layer_name};
use crate::{
    EngError,
    family::{
        qwen::flash_next::cuda::driver::chunk::prof,
        step::walk::{self, Half, Visits},
    },
};

const HYPER_TIME: usize = 8;
const MIXER_TIME: usize = 9;
const MOE_TIME: usize = 10;

struct GgufVisits<'s, 'a> {
    step: &'s FlashStep<'a>,
    moe: &'s mut dyn MoeBlock,
    profile: bool,
    clock: Option<Instant>,
}

impl GgufVisits<'_, '_> {
    fn site(half: Half) -> (Site, &'static str) {
        match half {
            Half::Mixer => (Site::Attn, "attn_hyper_connection"),
            Half::Forward => (Site::Mlp, "mlp_hyper_connection"),
        }
    }

    fn tick(&mut self, slot: usize) {
        if let Some(last) = &mut self.clock {
            prof::add(slot, last.elapsed());
            *last = Instant::now();
        }
    }
}

impl Visits for GgufVisits<'_, '_> {
    fn range(&self) -> Range<usize> {
        self.step.layers.clone()
    }

    fn read(&mut self, layer: usize, half: Half, n: usize) -> Result<(), EngError> {
        if half == Half::Mixer {
            self.clock = self.profile.then(Instant::now);
        }
        let (site, suffix) = Self::site(half);
        self.step.hyper_read(site, &layer_name(layer, suffix), n)?;
        if half == Half::Forward {
            self.tick(HYPER_TIME);
        }
        Ok(())
    }

    fn branch(&mut self, layer: usize, half: Half, n: usize, p0: usize) -> Result<(), EngError> {
        let slot = match half {
            Half::Mixer => {
                self.step.mixer(layer, p0, n)?;
                MIXER_TIME
            }
            Half::Forward => {
                self.step.moe_block(layer, p0, self.moe, n)?;
                MOE_TIME
            }
        };
        self.tick(slot);
        Ok(())
    }

    fn apply(&mut self, _layer: usize, half: Half, n: usize) -> Result<(), EngError> {
        let result = self.step.hyper_apply(Self::site(half).0, n);
        if half == Half::Forward {
            self.tick(HYPER_TIME);
        }
        result
    }
}

impl FlashStep<'_> {
    pub(super) fn run_gguf_range(
        &self,
        range: Range<usize>,
        pos: usize,
        n: usize,
        moe: &mut dyn MoeBlock,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let mut visits = GgufVisits {
            step: self,
            moe,
            profile: prof::on(),
            clock: None,
        };
        self.poison_on(walk::visits_in(&mut visits, range, n, pos, |_, _| Ok(())))
    }
}
