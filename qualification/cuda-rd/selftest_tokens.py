#!/usr/bin/env python3
"""CPU self-test of rstokens.py: what the lexer keeps and what it drops. Usage: python3 -I selftest_tokens.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rstokens import skip_group, tokenize  # noqa: E402

results = []


def check(name, cond):
    results.append(cond)
    print(("ok   " if cond else "FAIL ") + name)


def words(text):
    return [t for _, t in tokenize(text)]


def main():
    check("identifiers and punctuation", words("pub fn a(b: u32) {}") == ["pub", "fn", "a", "(", "b", ":", "u32", ")", "{", "}"])
    check("line comment dropped", words("a // pub fn x\nb") == ["a", "b"])
    check("block comment dropped", words("a /* pub fn x */ b") == ["a", "b"])
    check("nested block comment dropped", words("a /* x /* pub */ pub */ b") == ["a", "b"])
    check("string literal becomes STR", words('let s = "pub fn x";') == ["let", "s", "=", "STR", ";"])
    check("string with an escaped quote", words(r'"a\"pub" b') == ["STR", "b"])
    check("raw string with hashes", words('r#"pub " fn"# b') == ["STR", "b"])
    check("byte string", words('b"pub" b') == ["STR", "b"])
    check("char literal becomes CHR, lifetime stays an identifier", words("('p', &'a str)") == ["(", "CHR", ",", "&", "a", "str", ")"])
    check("escaped char literal", words(r"'\n' x") == ["CHR", "x"])
    check("line numbers follow newlines inside block comments and strings", tokenize('/* a\nb */ x "q\nr" y') == [(2, "x"), (3, "STR"), (3, "y")])
    check("pub split across lines is two tokens on two lines", tokenize("pub\nfn") == [(1, "pub"), (2, "fn")])
    toks = tokenize("a(b(c)) d")
    check("skip_group returns the index after the matching close", toks[skip_group(toks, 1, "(", ")")][1] == "d")
    print("TOKENS SELFTEST", "OK" if all(results) else "FAILED", f"({len(results)} checks)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
