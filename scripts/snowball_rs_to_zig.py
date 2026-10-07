#!/usr/bin/env python3
"""Translate a generated Snowball stemmer from Rust to Zig, line by line.

    scripts/snowball_rs_to_zig.py <qdrant-rust-stemmers>/src/snowball/algorithms/english.rs \
        | zig fmt --stdin > src/text/stem_english.zig

strawmANN has to stem token for token as Qdrant's text index does, and Qdrant
links `qdrant-rust-stemmers` 1.2.2, whose algorithms the Snowball compiler
generated as Rust (decisions.md, 2026-10-07). Generated code is regular: a
closed set of line shapes (labeled loops, cursor saves, among searches, slice
edits). Translating those shapes mechanically keeps the port a function of the
crate's source, which a hand port of 1,249 lines would not be, and a line in a
shape not listed here stops the translation instead of being guessed at.

The runtime it targets is `src/text/snowball.zig`. The translation is checked
end to end against the crate: `conformance stem-vocabulary` writes each word's
stem from the crate, and `src/text/text.zig`'s tests require the port to agree
on Snowball's English vocabulary (`src/text/testdata/stem_english.tsv`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

AMONG_TABLE = re.compile(r"^static (A_\d+): &'static \[Among<Context>; (\d+)\] = &\[$")
AMONG_ENTRY = re.compile(r'^Among\("((?:[^"\\]|\\.)*)", (-?\d+), (-?\d+), None\),$')
GROUPING = re.compile(r"^static (G_\w+): &'static \[u8; \d+\] = &\[([\d, ]+)\];$")
FN = re.compile(r"^fn (r_\w+)\(env: &mut SnowballEnv, context: &mut Context\) -> bool \{$")
STEM = "pub fn stem(env: &mut SnowballEnv) -> bool {"

#: Statement shapes, applied to a stripped line in order; the first match wins.
RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^'(\w+): loop ?\{$"), r"\1: while (true) {"),
    (re.compile(r"^'(\w+): for _ in 0\.\.1 \{$"), r"\1: for (0..1) |_| {"),
    (re.compile(r"^break '(\w+);$"), r"break :\1;"),
    (re.compile(r"^continue '(\w+);$"), r"continue :\1;"),
    (re.compile(r"^let mut among_var;$"), "var among_var: i32 = 0;"),
    (re.compile(r"^among_var = env\.(find_among(?:_b)?)\((A_\d+), context\);$"),
     r"among_var = env.\1(&\2);"),
    (re.compile(r"^if env\.(find_among(?:_b)?)\((A_\d+), context\) == (\d+) \{$"),
     r"if (env.\1(&\2) == \3) {"),
    (re.compile(r'^if !env\.(eq_s(?:_b)?)\(&"((?:[^"\\]|\\.)*)"\) \{$'), r'if (!env.\1("\2")) {'),
    (re.compile(r'^if !env\.slice_from\("((?:[^"\\]|\\.)*)"\) \{$'), r'if (!env.slice_from("\1")) {'),
    (re.compile(r"^if !env\.slice_del\(\) \{$"), "if (!env.slice_del()) {"),
    (re.compile(r"^if !env\.((?:in|out)_grouping(?:_b)?)\((G_\w+), (\d+), (\d+)\) \{$"),
     r"if (!env.\1(&\2, \3, \4)) {"),
    (re.compile(r"^if !(r_\w+)\(env, context\) \{$"), r"if (!\1(env, ctx)) {"),
    (re.compile(r"^if among_var == (\d+) \{$"), r"if (among_var == \1) {"),
    (re.compile(r"^\} else if among_var == (\d+) \{$"), r"} else if (among_var == \1) {"),
    (re.compile(r"^if !\(context\.(i_p\d) <= env\.cursor\)\{$"), r"if (!(ctx.\1 <= env.cursor)) {"),
    (re.compile(r"^if !context\.b_Y_found \{$"), "if (!ctx.b_Y_found) {"),
    (re.compile(r"^if env\.cursor != context\.(i_p\d) \{$"), r"if (env.cursor != ctx.\1) {"),
    (re.compile(r"^if env\.cursor (>=|<=|>|<) env\.(limit|limit_backward) \{$"),
     r"if (env.cursor \1 env.\2) {"),
    (re.compile(r"^let (v_\d+) = env\.limit - env\.cursor;$"), r"const \1 = env.limit - env.cursor;"),
    (re.compile(r"^let (v_\d+) = env\.cursor;$"), r"const \1 = env.cursor;"),
    (re.compile(r"^let c = env\.cursor;$"), "const c = env.cursor;"),
    (re.compile(r"^let c = env\.byte_index_for_hop\((-?\d+)\);$"),
     r"const c = env.byte_index_for_hop(\1);"),
    (re.compile(r"^if env\.limit_backward as i32 > c \|\| c > env\.limit as i32 \{$"),
     "if (@as(i32, @intCast(env.limit_backward)) > c or c > @as(i32, @intCast(env.limit))) {"),
    (re.compile(r"^if 0 as i32 > c \|\| c > env\.limit as i32 \{$"),
     "if (0 > c or c > @as(i32, @intCast(env.limit))) {"),
    (re.compile(r"^env\.cursor = c as usize;$"), "env.cursor = @intCast(c);"),
    (re.compile(r"^let \(bra, ket\) = \(env\.cursor, env\.cursor\);$"),
     "const bra = env.cursor;\nconst ket = env.cursor;"),
    (re.compile(r'^env\.insert\(bra, ket, "((?:[^"\\]|\\.)*)"\);$'), r'env.insert(bra, ket, "\1");'),
    (re.compile(r"^env\.cursor = env\.limit - (v_\d+);$"), r"env.cursor = env.limit - \1;"),
    (re.compile(r"^env\.cursor = (v_\d+|c|env\.limit|env\.limit_backward);$"), r"env.cursor = \1;"),
    (re.compile(r"^env\.(bra|ket) = env\.cursor;$"), r"env.\1 = env.cursor;"),
    (re.compile(r"^env\.limit_backward = env\.cursor;$"), "env.limit_backward = env.cursor;"),
    (re.compile(r"^env\.(next_char|previous_char)\(\);$"), r"env.\1();"),
    (re.compile(r"^context\.(\w+) = (true|false|env\.limit|env\.cursor);$"), r"ctx.\1 = \2;"),
    (re.compile(r"^return (true|false);$"), r"return \1;"),
    (re.compile(r"^\}$"), "}"),
]


def translate_line(line: str, where: str) -> list[str]:
    for pattern, repl in RULES:
        if pattern.match(line):
            return pattern.sub(repl, line).split("\n")
    raise SystemExit(f"{where}: no rule for {line!r}")


def finish_function(name: str, body: list[str]) -> list[str]:
    """Zig refuses an unused local or parameter, which generated Rust allows."""
    text = "\n".join(body)
    out = []
    if not re.search(r"\bctx\b", text):
        out.append("_ = ctx;")
    for line in body:
        out.append(line)
        m = re.match(r"const (\w+) = ", line)
        if m and len(re.findall(rf"\b{m.group(1)}\b", text)) == 1:
            out.append(f"_ = {m.group(1)};")
    return out


def indent(lines: list[str]) -> list[str]:
    out, depth = [], 1
    for line in lines:
        if line.startswith("}"):
            depth -= 1
        out.append("    " * depth + line)
        if line.endswith("{"):
            depth += 1
    return out


def main(argv: list[str]) -> int:
    src = Path(argv[1]).read_text(encoding="utf-8").splitlines()
    tables: list[tuple[str, list[tuple[str, int, int]]]] = []
    groupings: list[tuple[str, str]] = []
    functions: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(src):
        line = src[i].strip()
        if m := AMONG_TABLE.match(line):
            name, n = m.group(1), int(m.group(2))
            entries = []
            i += 1
            while src[i].strip() != "];":
                e = AMONG_ENTRY.match(src[i].strip())
                if not e:
                    raise SystemExit(f"line {i + 1}: an among entry with a routine: {src[i]!r}")
                entries.append((e.group(1), int(e.group(2)), int(e.group(3))))
                i += 1
            assert len(entries) == n, name
            tables.append((name, entries))
        elif m := GROUPING.match(line):
            groupings.append((m.group(1), m.group(2)))
        elif (m := FN.match(line)) or line == STEM:
            name = m.group(1) if m else "stem"
            body = []
            i += 1
            if name == "stem":
                # `let mut context = &mut Context { b_Y_found: false, i_p2: 0, i_p1: 0, };`
                assert src[i].strip() == "let mut context = &mut Context {", src[i]
                fields = []
                i += 1
                while src[i].strip() != "};":
                    k, v = src[i].strip().rstrip(",").split(": ")
                    fields.append(f".{k} = {v}")
                    i += 1
                i += 1
                body.append(f"var context = Context{{ {', '.join(fields)} }};")
                body.append("const ctx = &context;")
            depth = 1
            while depth > 0:
                raw = src[i].strip()
                i += 1
                if not raw or raw.startswith("//"):
                    continue
                depth += raw.count("{") - raw.count("}")
                if depth == 0:
                    break
                body += translate_line(raw, f"{argv[1]}:{i}")
            functions.append((name, body))
            continue
        i += 1

    out = [
        "//! The English (Porter2) Snowball stemmer, translated mechanically from",
        "//! `qdrant-rust-stemmers` 1.2.2's generated `src/snowball/algorithms/english.rs`",
        "//! by `scripts/snowball_rs_to_zig.py` (then `zig fmt`). Do not edit: regenerate.",
        "",
        'const snowball = @import("snowball.zig");',
        "const Env = snowball.Env;",
        "const Among = snowball.Among;",
        "",
    ]
    for name, entries in tables:
        out.append(f"const {name} = [_]Among{{")
        for s, sub, res in entries:
            out.append(f'    .{{ .s = "{s}", .substring_i = {sub}, .result = {res} }},')
        out.append("};")
        out.append("")
    for name, values in groupings:
        out.append(f"const {name} = [_]u8{{ {values} }};")
    out += ["", "const Context = struct {", "    b_Y_found: bool,", "    i_p2: usize,",
            "    i_p1: usize,", "};", ""]
    for name, body in functions:
        if name == "stem":
            out.append("/// Stem the word in `env` in place, as `Stemmer::stem` does.")
            out.append("pub fn stem(env: *Env) bool {")
            out += indent(body)
        else:
            out.append(f"fn {name}(env: *Env, ctx: *Context) bool {{")
            out += indent(finish_function(name, body))
        out.append("}")
        out.append("")
    sys.stdout.write("\n".join(out).rstrip("\n") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
