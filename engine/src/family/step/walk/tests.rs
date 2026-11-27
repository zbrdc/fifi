use super::*;
use std::cell::RefCell;

#[derive(Debug, PartialEq, Eq)]
enum Call {
    Begin(usize),
    Read(usize, Half, usize),
    Branch(usize, Half, usize, usize),
    Apply(usize, Half, usize),
    Tap(usize, Half),
    HostUpdate,
}

struct Driver<'a> {
    calls: &'a RefCell<Vec<Call>>,
    fail_branch: bool,
}

impl Visits for Driver<'_> {
    fn range(&self) -> Range<usize> {
        0..5
    }

    fn begin(&mut self, n: usize) -> Result<(), EngError> {
        self.calls.borrow_mut().push(Call::Begin(n));
        Ok(())
    }

    fn read(&mut self, visit: usize, half: Half, n: usize) -> Result<(), EngError> {
        self.calls.borrow_mut().push(Call::Read(visit, half, n));
        Ok(())
    }

    fn branch(&mut self, visit: usize, half: Half, n: usize, p0: usize) -> Result<(), EngError> {
        self.calls
            .borrow_mut()
            .push(Call::Branch(visit, half, n, p0));
        if self.fail_branch {
            return Err(EngError::BadRequest("branch refused".into()));
        }
        Ok(())
    }

    fn apply(&mut self, visit: usize, half: Half, n: usize) -> Result<(), EngError> {
        self.calls.borrow_mut().push(Call::Apply(visit, half, n));
        Ok(())
    }
}

#[test]
fn ranges_keep_absolute_visits_without_restarting_the_chunk() {
    let calls = RefCell::new(Vec::new());
    let mut driver = Driver {
        calls: &calls,
        fail_branch: false,
    };
    let tap = |visit, half| {
        calls.borrow_mut().push(Call::Tap(visit, half));
        Ok(())
    };
    visits_in(&mut driver, 2..3, 4, 17, tap).unwrap();
    calls.borrow_mut().push(Call::HostUpdate);
    visits_in(&mut driver, 3..5, 4, 17, tap).unwrap();

    let mut expected = Vec::new();
    for visit in 2..5 {
        if visit == 3 {
            expected.push(Call::HostUpdate);
        }
        for half in [Half::Mixer, Half::Forward] {
            expected.extend([
                Call::Read(visit, half, 4),
                Call::Branch(visit, half, 4, 17),
                Call::Apply(visit, half, 4),
                Call::Tap(visit, half),
            ]);
        }
    }
    assert_eq!(calls.into_inner(), expected);
}

#[test]
fn full_walk_begins_once_and_stops_on_a_branch_error() {
    let calls = RefCell::new(Vec::new());
    let mut driver = Driver {
        calls: &calls,
        fail_branch: true,
    };
    let err = visits(&mut driver, 1, 8, |_, _| panic!("tapped a failed half")).unwrap_err();
    assert!(matches!(err, EngError::BadRequest(message) if message == "branch refused"));
    assert_eq!(
        calls.into_inner(),
        [
            Call::Begin(1),
            Call::Read(0, Half::Mixer, 1),
            Call::Branch(0, Half::Mixer, 1, 8),
        ]
    );
}
