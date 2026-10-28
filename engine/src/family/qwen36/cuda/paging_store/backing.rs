//! Immutable page versions and allocation charges retained by their last shared owner.
use std::{
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::EngError;

const ARC_HEADER: usize = 2 * size_of::<AtomicUsize>();

fn memory(what: impl std::fmt::Display) -> EngError {
    EngError::MemoryPlan(format!("paged KV host allocation: {what}"))
}

fn sum(a: usize, b: usize) -> Result<usize, EngError> {
    a.checked_add(b).ok_or_else(|| memory("size overflows"))
}

struct Quota {
    used: AtomicUsize,
    limit: usize,
}

impl Quota {
    fn take(&self, bytes: usize) -> Result<(), EngError> {
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|next| *next <= self.limit)
            })
            .map(|_| ())
            .map_err(|_| memory(format!("owned bytes exceed the {}-byte limit", self.limit)))
    }
}

pub(super) struct Charge {
    quota: Arc<Quota>,
    bytes: usize,
}

impl Charge {
    fn new(quota: &Arc<Quota>, bytes: usize) -> Result<Self, EngError> {
        quota.take(bytes)?;
        Ok(Self {
            quota: Arc::clone(quota),
            bytes,
        })
    }

    pub(super) fn grow(&mut self, bytes: usize) -> Result<(), EngError> {
        let total = sum(self.bytes, bytes)?;
        self.quota.take(bytes)?;
        self.bytes = total;
        Ok(())
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.quota.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

pub(super) struct Page {
    bytes: Vec<u8>,
    _charge: Charge,
}

impl Page {
    fn minimum_bytes(bytes: usize) -> Result<usize, EngError> {
        sum(bytes, sum(size_of::<Self>(), ARC_HEADER)?)
    }

    fn new(quota: &Arc<Quota>, bytes: usize) -> Result<Self, EngError> {
        let mut charge = Charge::new(quota, Self::minimum_bytes(bytes)?)?;
        let mut data = Vec::new();
        data.try_reserve_exact(bytes).map_err(memory)?;
        charge.grow(data.capacity() - bytes)?;
        data.resize(bytes, 0);
        Ok(Self {
            bytes: data,
            _charge: charge,
        })
    }

    pub(super) fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

struct Saved {
    pages: Vec<Arc<Page>>,
    groups: usize,
    _charge: Charge,
}

#[derive(Clone)]
pub(super) struct Snapshot(Arc<Saved>);

impl Snapshot {
    pub(super) fn charge(&self, bytes: usize) -> Result<Charge, EngError> {
        Charge::new(&self.0._charge.quota, bytes)
    }
}

pub(super) struct Backing {
    pages: Vec<Arc<Page>>,
    quota: Arc<Quota>,
    _fixed: Charge,
}

impl Backing {
    pub(super) fn minimum_bytes(
        count: usize,
        page_bytes: usize,
        fixed_bytes: usize,
    ) -> Result<usize, EngError> {
        let table = count
            .checked_mul(size_of::<Arc<Page>>())
            .ok_or_else(|| memory("table size overflows"))?;
        let pages = count
            .checked_mul(Page::minimum_bytes(page_bytes)?)
            .ok_or_else(|| memory("page charges overflow"))?;
        let metadata = sum(size_of::<Quota>(), ARC_HEADER)?;
        sum(metadata, sum(fixed_bytes, sum(table, pages)?)?)
    }

    pub(super) fn new(
        count: usize,
        page_bytes: usize,
        fixed: usize,
        limit: usize,
    ) -> Result<Self, EngError> {
        let minimum = Self::minimum_bytes(count, page_bytes, fixed)?;
        if minimum > limit {
            return Err(memory(format!(
                "minimum {minimum} owned bytes exceed {limit}-byte host limit"
            )));
        }
        let quota_bytes = sum(size_of::<Quota>(), ARC_HEADER)?;
        let quota = Arc::new(Quota {
            used: AtomicUsize::new(quota_bytes),
            limit,
        });
        let table_bytes = count
            .checked_mul(size_of::<Arc<Page>>())
            .ok_or_else(|| memory("table overflows"))?;
        let mut charge = Charge::new(&quota, sum(fixed, table_bytes)?)?;
        let mut pages = Vec::new();
        pages.try_reserve_exact(count).map_err(memory)?;
        charge.grow((pages.capacity() - count) * size_of::<Arc<Page>>())?;
        // Each payload and Arc header is admitted before its allocation.
        for _ in 0..count {
            pages.push(Arc::new(Page::new(&quota, page_bytes)?));
        }
        Ok(Self {
            pages,
            quota,
            _fixed: charge,
        })
    }

    pub(super) fn owned_bytes(&self) -> usize {
        self.quota.used.load(Ordering::Acquire)
    }

    pub(super) fn bytes(&self, index: usize) -> &[u8] {
        &self.pages[index].bytes
    }

    pub(super) fn detach(&self, index: usize) -> Result<Option<Page>, EngError> {
        if Arc::strong_count(&self.pages[index]) == 1 {
            return Ok(None);
        }
        Page::new(&self.quota, self.pages[index].bytes.len()).map(Some)
    }

    pub(super) fn unique_bytes(&mut self, index: usize) -> Result<&mut [u8], EngError> {
        Arc::get_mut(&mut self.pages[index])
            .map(|page| page.bytes.as_mut_slice())
            .ok_or_else(|| memory("a shared page needs detached storage"))
    }

    pub(super) fn publish(&mut self, index: usize, replacement: Option<Page>) {
        if let Some(page) = replacement {
            self.pages[index] = Arc::new(page);
        }
    }

    pub(super) fn snapshot(
        &self,
        layers: usize,
        groups: usize,
        visible: usize,
    ) -> Result<Snapshot, EngError> {
        let count = layers
            .checked_mul(visible)
            .ok_or_else(|| memory("snapshot table overflows"))?;
        let table_bytes = count
            .checked_mul(size_of::<Arc<Page>>())
            .ok_or_else(|| memory("snapshot table overflows"))?;
        let overhead = sum(
            size_of::<Saved>(),
            sum(ARC_HEADER, size_of::<super::PageSnapshot>())?,
        )?;
        let mut charge = Charge::new(&self.quota, sum(overhead, table_bytes)?)?;
        let mut pages = Vec::new();
        pages.try_reserve_exact(count).map_err(memory)?;
        charge.grow((pages.capacity() - count) * size_of::<Arc<Page>>())?;
        for layer in 0..layers {
            for group in 0..visible {
                pages.push(Arc::clone(&self.pages[layer * groups + group]));
            }
        }
        Ok(Snapshot(Arc::new(Saved {
            pages,
            groups: visible,
            _charge: charge,
        })))
    }

    pub(super) fn same_domain(&self, snapshot: &Snapshot) -> bool {
        Arc::ptr_eq(&self.quota, &snapshot.0._charge.quota)
    }

    /// The caller validated geometry and drained the owned stream before publication.
    pub(super) fn restore(&mut self, snapshot: &Snapshot, layers: usize, groups: usize) {
        for layer in 0..layers {
            for group in 0..snapshot.0.groups {
                self.pages[layer * groups + group] =
                    Arc::clone(&snapshot.0.pages[layer * snapshot.0.groups + group]);
            }
        }
    }
}

/// Staged spill-only host ownership for the device-first pager.
pub(super) struct SpillBacking {
    pages: Vec<Option<Arc<Page>>>,
    quota: Arc<Quota>,
    page_bytes: usize,
    _fixed: Charge,
}

impl SpillBacking {
    pub(super) fn payload_bytes(count: usize, page_bytes: usize) -> Result<usize, EngError> {
        count
            .checked_mul(Page::minimum_bytes(page_bytes)?)
            .ok_or_else(|| memory("spill payload size overflows"))
    }

    pub(super) fn grow_fixed(&mut self, bytes: usize) -> Result<(), EngError> {
        self._fixed.grow(bytes)
    }

    pub(super) fn minimum_bytes(count: usize, fixed: usize) -> Result<usize, EngError> {
        let table = count
            .checked_mul(size_of::<Option<Arc<Page>>>())
            .ok_or_else(|| memory("spill table size overflows"))?;
        sum(sum(size_of::<Quota>(), ARC_HEADER)?, sum(fixed, table)?)
    }

    pub(super) fn new(
        count: usize,
        page_bytes: usize,
        fixed: usize,
        limit: usize,
    ) -> Result<Self, EngError> {
        if page_bytes == 0 {
            return Err(EngError::BadRequest(
                "paged KV spill needs a positive page size".to_owned(),
            ));
        }
        Page::minimum_bytes(page_bytes)?;
        let minimum = Self::minimum_bytes(count, fixed)?;
        if minimum > limit {
            return Err(memory(format!(
                "minimum {minimum} owned bytes exceed {limit}-byte host limit"
            )));
        }
        let quota = Arc::new(Quota {
            used: AtomicUsize::new(sum(size_of::<Quota>(), ARC_HEADER)?),
            limit,
        });
        let mut charge = Charge::new(&quota, minimum - sum(size_of::<Quota>(), ARC_HEADER)?)?;
        let mut pages = Vec::new();
        pages.try_reserve_exact(count).map_err(memory)?;
        let extra = (pages.capacity() - count)
            .checked_mul(size_of::<Option<Arc<Page>>>())
            .ok_or_else(|| memory("spill table capacity overflows"))?;
        charge.grow(extra)?;
        pages.resize_with(count, || None);
        Ok(Self {
            pages,
            quota,
            page_bytes,
            _fixed: charge,
        })
    }

    pub(super) fn owned_bytes(&self) -> usize {
        self.quota.used.load(Ordering::Acquire)
    }

    fn vacant(&self, index: usize) -> Result<(), EngError> {
        match self.pages.get(index) {
            Some(None) => Ok(()),
            Some(Some(_)) => Err(EngError::BadRequest(
                "paged KV spill page is already published".to_owned(),
            )),
            None => Err(EngError::BadRequest(
                "paged KV spill page is outside the host table".to_owned(),
            )),
        }
    }

    pub(super) fn prepare(&self, index: usize) -> Result<Page, EngError> {
        self.vacant(index)?;
        Page::new(&self.quota, self.page_bytes)
    }

    pub(super) fn publish(&mut self, index: usize, page: Page) -> Result<(), EngError> {
        self.vacant(index)?;
        if !Arc::ptr_eq(&self.quota, &page._charge.quota) || page.bytes.len() != self.page_bytes {
            return Err(EngError::BadRequest(
                "paged KV spill page has a different owner or geometry".to_owned(),
            ));
        }
        self.pages[index] = Some(Arc::new(page));
        Ok(())
    }

    pub(super) fn bytes(&self, index: usize) -> Result<&[u8], EngError> {
        self.pages
            .get(index)
            .and_then(Option::as_ref)
            .map(|page| page.bytes.as_slice())
            .ok_or_else(|| EngError::BadRequest("paged KV spill page is not published".to_owned()))
    }

    pub(super) fn clear(&mut self) {
        self.pages.fill(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spill_payload_is_charged_only_while_prepared_or_published() {
        let minimum = SpillBacking::minimum_bytes(4, 128).unwrap();
        let page_charge = Page::minimum_bytes(256).unwrap();
        assert!(SpillBacking::new(4, 256, 128, minimum - 1).is_err());
        let mut backing = SpillBacking::new(4, 256, 128, minimum + page_charge).unwrap();
        assert_eq!(backing.owned_bytes(), minimum);
        assert!(backing.bytes(2).is_err());

        let pending = backing.prepare(2).unwrap();
        assert_eq!(backing.owned_bytes(), minimum + page_charge);
        assert!(backing.bytes(2).is_err());
        assert!(backing.prepare(0).is_err());
        drop(pending);
        assert_eq!(backing.owned_bytes(), minimum);

        let mut pending = backing.prepare(2).unwrap();
        pending.bytes_mut().fill(17);
        backing.publish(2, pending).unwrap();
        assert_eq!(backing.bytes(2).unwrap(), [17; 256]);
        assert_eq!(backing.owned_bytes(), minimum + page_charge);
        assert!(backing.prepare(2).is_err());
        assert!(backing.prepare(4).is_err());
        assert!(backing.bytes(4).is_err());

        backing.clear();
        assert_eq!(backing.owned_bytes(), minimum);
        assert!(backing.bytes(2).is_err());
        let other = SpillBacking::new(4, 256, 128, minimum + page_charge).unwrap();
        let foreign = other.prepare(0).unwrap();
        assert!(backing.publish(0, foreign).is_err());
        assert_eq!(other.owned_bytes(), minimum);
        assert_eq!(backing.owned_bytes(), minimum);
        assert!(SpillBacking::minimum_bytes(usize::MAX, 128).is_err());
        assert!(SpillBacking::new(1, 0, 128, 4096).is_err());
    }

    #[test]
    fn minimum_charge_matches_backing_and_refuses_one_byte_less() {
        let minimum = Backing::minimum_bytes(4, 256, 128).unwrap();
        assert!(matches!(
            Backing::new(4, 256, 128, minimum - 1),
            Err(EngError::MemoryPlan(_))
        ));
        let backing = Backing::new(4, 256, 128, minimum).unwrap();
        assert_eq!(backing.owned_bytes(), minimum);
        assert!(Backing::minimum_bytes(usize::MAX, 256, 128).is_err());
    }

    #[test]
    fn shared_versions_release_only_after_last_snapshot_drop() {
        let mut backing = Backing::new(4, 256, 128, 8192).expect("backing");
        let base = backing.owned_bytes();
        let saved = backing.snapshot(2, 2, 1).expect("snapshot");
        let copy = saved.clone();
        let mut page = backing.detach(0).expect("quota").expect("shared");
        page.bytes_mut().fill(17);
        backing.publish(0, Some(page));
        let held = backing.owned_bytes();
        assert!(held >= base + 256);
        assert_eq!(saved.0.pages[0].bytes, vec![0; 256]);
        assert_eq!(backing.bytes(0), vec![17; 256]);
        drop(saved);
        assert_eq!(backing.owned_bytes(), held);
        std::thread::spawn(move || drop(copy))
            .join()
            .expect("snapshot holder");
        assert_eq!(backing.owned_bytes(), base);
    }

    #[test]
    fn quota_refusal_and_abandoned_detach_do_not_publish_or_leak() {
        let mut backing = Backing::new(2, 256, 128, 4096).expect("backing");
        let saved = backing.snapshot(1, 2, 1).expect("snapshot");
        let held = backing.owned_bytes();
        let detached = backing.detach(0).expect("detached");
        drop(detached);
        assert_eq!(backing.owned_bytes(), held);
        assert!(backing.quota.take(4096).is_err());
        assert_eq!(backing.owned_bytes(), held);
        assert_eq!(backing.bytes(0), vec![0; 256]);
        assert!(backing.unique_bytes(0).is_err());
        let other = Backing::new(2, 256, 128, 4096).expect("other");
        assert!(!other.same_domain(&saved));
        assert!(backing.same_domain(&saved));
    }
}
