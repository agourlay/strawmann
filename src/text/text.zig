//! BM25 over a payload field's text index, Qdrant dev's `Query.text`
//! (decisions.md, 2026-10-07): tokenizer, stemmer and, in time, the index.

const std = @import("std");

pub const snowball = @import("snowball.zig");
pub const stem_english = @import("stem_english.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const index = @import("index.zig");
pub const TextIndex = index.TextIndex;
pub const unicode_tables = @import("unicode_tables.zig");
pub const stopwords_english = @import("stopwords_english.zig");

/// `word` stemmed with the English Snowball stemmer, as Qdrant's
/// `qdrant-rust-stemmers` 1.2.2 stems it, in `buf`, which must hold
/// `word.len + snowball.capacity_slack` bytes. The input is expected
/// lowercase, as Qdrant's text index lowercases before it stems.
pub fn stemEnglish(buf: []u8, word: []const u8) []const u8 {
    var env = snowball.Env.init(buf, word);
    _ = stem_english.stem(&env);
    return env.current();
}

test "the English stemmer stems as Porter2 does" {
    var buf: [64]u8 = undefined;
    const cases = [_][2][]const u8{
        .{ "running", "run" },         .{ "runners", "runner" },
        .{ "dogs", "dog" },            .{ "fruitlessly", "fruitless" },
        .{ "generously", "generous" }, .{ "skies", "sky" },
        .{ "dying", "die" },           .{ "communism", "communism" },
        .{ "caresses", "caress" },     .{ "ponies", "poni" },
        .{ "a", "a" },                 .{ "", "" },
    };
    for (cases) |c| try std.testing.expectEqualStrings(c[1], stemEnglish(&buf, c[0]));
}

/// Snowball's English test vocabulary (`snowball-data/english/voc.txt`, 42,649
/// words) with each word's stem from `qdrant-rust-stemmers` 1.2.2, written by
/// `conformance stem-vocabulary --words voc.txt`. The port must agree on all.
const stem_vocabulary = @embedFile("testdata/stem_english.tsv");

/// Every `word\tstem` line of `vocabulary` the port stems differently, counted,
/// with the first few printed.
fn stemMismatches(vocabulary: []const u8) !usize {
    var buf: [1024]u8 = undefined;
    var mismatches: usize = 0;
    var lines = std.mem.splitScalar(u8, vocabulary, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const tab = std.mem.indexOfScalar(u8, line, '\t') orelse return error.MalformedVocabulary;
        const word = line[0..tab];
        const want = line[tab + 1 ..];
        if (word.len + snowball.capacity_slack > buf.len) return error.WordTooLong;
        const got = stemEnglish(&buf, word);
        if (!std.mem.eql(u8, got, want)) {
            if (mismatches < 10) std.debug.print("stem({s}) = {s}, crate says {s}\n", .{ word, got, want });
            mismatches += 1;
        }
    }
    return mismatches;
}

test "the English stemmer agrees with qdrant-rust-stemmers on Snowball's vocabulary" {
    try std.testing.expectEqual(@as(usize, 0), try stemMismatches(stem_vocabulary));
}

test {
    std.testing.refAllDecls(@This());
}
