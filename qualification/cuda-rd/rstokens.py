"""A small Rust lexer for the public-surface source check (RD-10a-2a): comments (line and nested block) and string, raw-string, byte-string
and char literals are removed or replaced by STR/CHR, lifetimes are kept as identifiers, so a `pub` inside a literal or a comment is never
a token. CPU only, stdlib only, `python3 -I`. Design guidance only, no code copied."""
import re


def tokenize(text):
    """[(line, token)]; identifiers and punctuation, literals replaced by STR or CHR, comments dropped."""
    out, i, n, line = [], 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
        elif c.isspace():
            i += 1
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            depth = 1
            i += 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth += 1
                    i += 2
                elif text.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    line += text[i] == "\n"
                    i += 1
        elif c in "rb" and re.match(r'(?:br|rb|r)#*"', text[i:i + 12]):
            m = re.match(r'(?:br|rb|r)(#*)"', text[i:])
            end = '"' + m.group(1)
            j = text.find(end, i + m.end())
            j = n if j < 0 else j + len(end)
            line += text.count("\n", i, j)
            out.append((line, "STR"))
            i = j
        elif c == '"' or (c == "b" and text.startswith('b"', i)):
            i += 2 if c == "b" else 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    i += 1
                line += text[i] == "\n"
                i += 1
            out.append((line, "STR"))
            i += 1
        elif c == "'":
            m = re.match(r"'(?:\\.[^']*|[^\\'])'", text[i:])
            if m:
                out.append((line, "CHR"))
                i += m.end()
            else:
                i += 1  # a lifetime tick; the identifier that follows is tokenized normally
        elif c.isalpha() or c == "_":
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[i:])
            out.append((line, m.group(0)))
            i += m.end()
        elif c.isdigit():
            m = re.match(r"[0-9][0-9A-Za-z_.]*", text[i:])
            out.append((line, "NUM"))
            i += m.end()
        else:
            out.append((line, c))
            i += 1
    return out


def skip_group(toks, i, open_c, close_c):
    """Index after the group that starts at toks[i] == open_c."""
    depth = 0
    while i < len(toks):
        depth += toks[i][1] == open_c
        depth -= toks[i][1] == close_c
        i += 1
        if depth == 0:
            break
    return i
