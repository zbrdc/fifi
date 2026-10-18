"""ACC-B export extractor: design 4.6, rules 1 to 11, and the repr escaping of design 4.5.

The container prints its workspace as one `ACCB-EXPORT <base64>` line on the shared stdout
(design 4.4 step 6). That stream is untrusted (design 4.5, T14, T23): the model shares the
container stdout and can forge, truncate, or hostile-shape the archive. This module takes the
last marker line, decodes it with strict base64, and unpacks only regular files and directories
into an empty destination, one member at a time with `extract(member, path, filter="data")`.
It never calls `extractall`, never follows or creates links, and never runs what it wrote: the
only later use of the files on the host is `sha256sum` and a data read by the checker (4.6
rule 10). Every skip is recorded in the manifest with a reason; every name in the manifest is
`repr`-escaped and cut at 200 characters, so model text cannot forge a manifest line (4.5).
Run as dan with `python3 -I` (host Python 3.14, so `filter="data"` of 3.12+ is available).
"""

import argparse
import base64
import hashlib
import io
import os
import pathlib
import sys
import tarfile
import unicodedata

MARKER = "ACCB-EXPORT "
END_BLOCKS = b"\x00" * 1024       # a POSIX tar ends with two zero blocks; less means truncation
EXPORT_MAX = 16 * 1024 * 1024   # 16 MiB for the decoded archive and for the total (rule 7)
FILE_MAX = 4 * 1024 * 1024      # 4 MiB for one file (rule 7)
MAX_ENTRIES = 1000              # rule 7: stop after the 1,000th entry
MAX_PARTS = 8                   # rule 6: at most 8 name parts
MAX_NAME = 200                  # rule 6 and 4.5: names and repr fields cut at 200 characters


def repr_cut(text, limit=MAX_NAME):
    """The `repr` of model text, cut at `limit` characters (design 4.5)."""
    return repr(text)[:limit]


def find_export_payload(stream_text):
    """The payload of the LAST line that starts with the marker, or None (rule 1).

    Lines split on `\\n` only: str.splitlines also splits on U+2028, U+2029, and U+0085, so a
    separator inside a JSON string of the stream would turn the rest of that line into a later
    `ACCB-EXPORT` candidate and could make a forged payload the last one (design 4.5).
    """
    payload = None
    for line in stream_text.split("\n"):
        if line.startswith(MARKER):
            payload = line[len(MARKER):]
    return payload


def normalized(name):
    """(name, reason): exactly one is None. A silent root entry gives (None, None).

    Strips one leading `./` and one trailing `/` (rule 6), so the names of a real
    `tar -C <dir> -cf - .` archive (`./`, `./x`, `./d/`) become ``, `x`, `d`.
    """
    text = name[2:] if name.startswith("./") else name
    text = text[:-1] if text.endswith("/") else text
    if text in ("", "."):
        return None, None                      # the root entry is skipped silently
    if text.startswith("/"):
        return None, "absolute name"
    parts = text.split("/")
    if len(parts) > MAX_PARTS:
        return None, "more than 8 parts"
    for part in parts:
        if part == "":
            return None, "empty part"
        if part == "..":
            return None, "dot-dot part"
        if part == ".":
            return None, "dot part"
    if "\\" in text:
        return None, "backslash"
    if len(text) > MAX_NAME:
        return None, "name over 200 characters"
    for char in text:
        if ord(char) < 0x20 or ord(char) == 0x7f:
            return None, "control character"
    return text, None


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_members(data):
    """The tar members of `data`, complete ones first: a truncated archive keeps what it has
    (rule 3). Returns (members, partial)."""
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            return archive.getmembers(), False
    except (tarfile.ReadError, EOFError):
        pass
    members, partial = [], True
    try:
        archive = tarfile.open(fileobj=io.BytesIO(data), mode="r:")
    except (tarfile.ReadError, EOFError):
        return [], True
    while True:
        try:
            member = archive.next()
        except (tarfile.ReadError, EOFError):
            break
        if member is None:
            partial = False     # the archive ended cleanly after all
            break
        members.append(member)
    return members, partial


def extract_export(stream_text, dest_dir):
    """Extracts the export of the last marker line into dest_dir (design 4.6).

    Returns the manifest lines: one per entry, `action\\treason\\tsize\\tsha256\\trepr(name)`.
    The lines never contain a raw tab or newline inside the name field, because the name is
    repr-escaped (design 4.5).
    """
    dest = pathlib.Path(dest_dir)
    dest.mkdir(parents=True, mode=0o700, exist_ok=True)   # rule 4: pi/work is new and 0700
    lines = []
    payload = find_export_payload(stream_text)
    if payload is None:
        return ["skipped\tno export\t-\t-\t''"]
    try:
        data = base64.b64decode(payload, validate=True)
    except ValueError:
        return ["skipped\tbad base64\t-\t-\t''"]
    if len(data) > EXPORT_MAX:    # `head -c` already caps the real stream; keep the rule anyway
        data = data[:EXPORT_MAX]

    def skip(name, reason, size):
        lines.append(f"skipped\t{reason}\t{size}\t-\t{repr_cut(name)}")

    members, partial = _read_members(data)
    if len(data) < len(END_BLOCKS) or not data.endswith(END_BLOCKS):
        partial = True                         # no end-of-archive marker: rule 3
    try:
        archive = tarfile.open(fileobj=io.BytesIO(data), mode="r:")
    except (tarfile.ReadError, EOFError):
        # Rule 3 says never crash: an empty or non-tar payload records a manifest line naming
        # the failure and refuses cleanly.
        return ["skipped\tunreadable archive\t-\t-\t''"]
    seen, total, entries, uid, gid = {}, 0, 0, os.getuid(), os.getgid()
    truncated = False
    for member in members:
        entries += 1
        if entries > MAX_ENTRIES:            # rule 7: stop after the 1,000th entry
            truncated = True
            break
        if not (member.isreg() or member.isdir()):
            skip(member.name, "not a regular file or directory", member.size)  # rule 5
            continue
        name, reason = normalized(member.name)
        if name is None and reason is None:
            continue                          # the root entry, silently (rule 6)
        if reason is not None:
            skip(member.name, reason, member.size)
            continue
        key = unicodedata.normalize("NFC", name).lower()                       # rule 9
        if key in seen:
            skip(member.name, "case or normalization duplicate", member.size)
            continue
        seen[key] = True
        if member.isreg() and member.size > FILE_MAX:                          # rule 7
            skip(member.name, "file over 4 MiB", member.size)
            continue
        if total + member.size > EXPORT_MAX:
            skip(member.name, "total over 16 MiB", member.size)
            continue
        target = dest.joinpath(*name.split("/"))
        try:
            if member.isdir():
                archive.extract(member, dest, filter="data")                   # rule 4
                os.chmod(target, 0o755)                                        # rule 8
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                archive.extract(member, dest, filter="data")
                os.chmod(target, 0o644)
                total += member.size
            os.chown(target, uid, gid)
            os.utime(target, None)             # rule 8: the extraction time
            digest = _file_digest(target) if member.isreg() else "-"
            lines.append(f"kept\tok\t{member.size}\t{digest}\t{repr_cut(member.name)}")
        except (tarfile.ReadError, EOFError, OSError):
            partial = True                                                     # rule 3
            if target.exists() and not target.is_dir():
                target.unlink(missing_ok=True)
            skip(member.name, "partial export", member.size)
    archive.close()
    if truncated:
        lines.append("skipped\tentry list truncated at 1000\t-\t-\t''")
    if partial:
        lines.append("skipped\tpartial export\t-\t-\t''")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stream", required=True, help="the stamped transcript file (pi.jsonl)")
    parser.add_argument("--dest", required=True, help="a new, empty destination directory")
    parser.add_argument("--manifest", required=True, help="the manifest file to write")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    dest = pathlib.Path(args.dest)
    if dest.exists():
        if not dest.is_dir() or any(dest.iterdir()):
            print(f"the destination {dest} exists and is not an empty directory", file=sys.stderr)
            return 1
    else:
        dest.mkdir(parents=True, mode=0o700)   # rule 4: a new, empty pi/work, mode 0700
    stream = pathlib.Path(args.stream).read_text(encoding="utf-8", errors="replace")
    lines = extract_export(stream, dest)
    pathlib.Path(args.manifest).write_text("".join(line + "\n" for line in lines),
                                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
