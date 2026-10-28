use std::path::PathBuf;

pub(super) fn sample() -> [u64; 5] {
    let membership = std::fs::read_to_string("/proc/self/cgroup").expect("cgroup v2 membership");
    let cgroup = cgroup_dir(&membership);
    let io = std::fs::read_to_string("/proc/self/io").expect("process I/O counters");
    let stat = std::fs::read_to_string(cgroup.join("memory.stat")).expect("cgroup memory.stat");
    let events =
        std::fs::read_to_string(cgroup.join("memory.events")).expect("cgroup memory.events");
    [
        counter(&io, "read_bytes:"),
        counter(&stat, "pgscan"),
        counter(&stat, "pgsteal"),
        counter(&stat, "workingset_refault_file"),
        counter(&events, "max"),
    ]
}

pub(super) fn delta(before: [u64; 5], after: [u64; 5]) -> [u64; 5] {
    std::array::from_fn(|index| {
        after[index]
            .checked_sub(before[index])
            .expect("kernel counter decreased")
    })
}

fn cgroup_dir(text: &str) -> PathBuf {
    let path = text
        .lines()
        .find_map(|line| line.strip_prefix("0::/"))
        .expect("cgroup v2 path missing");
    assert!(
        !path.starts_with('/') && !path.split('/').any(|part| part == ".."),
        "cgroup path escapes mount"
    );
    PathBuf::from("/sys/fs/cgroup").join(path)
}

fn counter(text: &str, key: &str) -> u64 {
    let mut matches = text
        .lines()
        .filter(|line| line.split_whitespace().next() == Some(key));
    let row = matches.next().expect("required kernel counter missing");
    assert!(matches.next().is_none(), "duplicate kernel counter");
    let mut fields = row.split_whitespace();
    let value = fields.nth(1).expect("kernel counter has no value");
    let value = value.parse::<u64>().expect("kernel counter is not u64");
    assert!(fields.next().is_none(), "extra kernel counter fields");
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_kernel_snapshot_is_readable_and_monotonic() {
        let before = sample();
        let after = sample();
        let _ = delta(before, after);
    }

    #[test]
    fn deltas_reject_each_decreasing_counter() {
        assert_eq!(delta([1; 5], [3; 5]), [2; 5]);
        assert_eq!(delta([3; 5], [3; 5]), [0; 5]);
        for index in 0..5 {
            let mut after = [3; 5];
            after[index] = 0;
            assert!(std::panic::catch_unwind(|| delta([1; 5], after)).is_err());
        }
    }

    #[test]
    fn cgroup_v2_path_stays_inside_the_mounted_hierarchy() {
        assert_eq!(cgroup_dir("0::/\n"), PathBuf::from("/sys/fs/cgroup"));
        assert_eq!(
            cgroup_dir("2:cpu:/ignored\n0::/user.slice/job\n"),
            PathBuf::from("/sys/fs/cgroup/user.slice/job")
        );
        for text in [
            "",
            "2:cpu:/x",
            "0::relative",
            "0:://outside",
            "0::/../outside",
            "0::/x/../outside",
        ] {
            assert!(
                std::panic::catch_unwind(|| cgroup_dir(text)).is_err(),
                "{text:?}"
            );
        }
    }

    #[test]
    fn counter_requires_one_complete_unsigned_field() {
        assert_eq!(counter("rchar: 99\nread_bytes: 7\n", "read_bytes:"), 7);
        assert_eq!(counter("pgscan 0\npgsteal 3\n", "pgscan"), 0);
        for text in [
            "",
            "other 1",
            "key",
            "key nope",
            "key -1",
            "key 18446744073709551616",
            "key 1 extra",
            "key 1\nkey 2",
        ] {
            assert!(
                std::panic::catch_unwind(|| counter(text, "key")).is_err(),
                "{text:?}"
            );
        }
    }
}
