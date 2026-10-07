//! The English (Porter2) Snowball stemmer, translated mechanically from
//! `qdrant-rust-stemmers` 1.2.2's generated `src/snowball/algorithms/english.rs`
//! by `scripts/snowball_rs_to_zig.py` (then `zig fmt`). Do not edit: regenerate.

const snowball = @import("snowball.zig");
const Env = snowball.Env;
const Among = snowball.Among;

const A_0 = [_]Among{
    .{ .s = "arsen", .substring_i = -1, .result = -1 },
    .{ .s = "commun", .substring_i = -1, .result = -1 },
    .{ .s = "gener", .substring_i = -1, .result = -1 },
};

const A_1 = [_]Among{
    .{ .s = "'", .substring_i = -1, .result = 1 },
    .{ .s = "'s'", .substring_i = 0, .result = 1 },
    .{ .s = "'s", .substring_i = -1, .result = 1 },
};

const A_2 = [_]Among{
    .{ .s = "ied", .substring_i = -1, .result = 2 },
    .{ .s = "s", .substring_i = -1, .result = 3 },
    .{ .s = "ies", .substring_i = 1, .result = 2 },
    .{ .s = "sses", .substring_i = 1, .result = 1 },
    .{ .s = "ss", .substring_i = 1, .result = -1 },
    .{ .s = "us", .substring_i = 1, .result = -1 },
};

const A_3 = [_]Among{
    .{ .s = "", .substring_i = -1, .result = 3 },
    .{ .s = "bb", .substring_i = 0, .result = 2 },
    .{ .s = "dd", .substring_i = 0, .result = 2 },
    .{ .s = "ff", .substring_i = 0, .result = 2 },
    .{ .s = "gg", .substring_i = 0, .result = 2 },
    .{ .s = "bl", .substring_i = 0, .result = 1 },
    .{ .s = "mm", .substring_i = 0, .result = 2 },
    .{ .s = "nn", .substring_i = 0, .result = 2 },
    .{ .s = "pp", .substring_i = 0, .result = 2 },
    .{ .s = "rr", .substring_i = 0, .result = 2 },
    .{ .s = "at", .substring_i = 0, .result = 1 },
    .{ .s = "tt", .substring_i = 0, .result = 2 },
    .{ .s = "iz", .substring_i = 0, .result = 1 },
};

const A_4 = [_]Among{
    .{ .s = "ed", .substring_i = -1, .result = 2 },
    .{ .s = "eed", .substring_i = 0, .result = 1 },
    .{ .s = "ing", .substring_i = -1, .result = 2 },
    .{ .s = "edly", .substring_i = -1, .result = 2 },
    .{ .s = "eedly", .substring_i = 3, .result = 1 },
    .{ .s = "ingly", .substring_i = -1, .result = 2 },
};

const A_5 = [_]Among{
    .{ .s = "anci", .substring_i = -1, .result = 3 },
    .{ .s = "enci", .substring_i = -1, .result = 2 },
    .{ .s = "ogi", .substring_i = -1, .result = 13 },
    .{ .s = "li", .substring_i = -1, .result = 16 },
    .{ .s = "bli", .substring_i = 3, .result = 12 },
    .{ .s = "abli", .substring_i = 4, .result = 4 },
    .{ .s = "alli", .substring_i = 3, .result = 8 },
    .{ .s = "fulli", .substring_i = 3, .result = 14 },
    .{ .s = "lessli", .substring_i = 3, .result = 15 },
    .{ .s = "ousli", .substring_i = 3, .result = 10 },
    .{ .s = "entli", .substring_i = 3, .result = 5 },
    .{ .s = "aliti", .substring_i = -1, .result = 8 },
    .{ .s = "biliti", .substring_i = -1, .result = 12 },
    .{ .s = "iviti", .substring_i = -1, .result = 11 },
    .{ .s = "tional", .substring_i = -1, .result = 1 },
    .{ .s = "ational", .substring_i = 14, .result = 7 },
    .{ .s = "alism", .substring_i = -1, .result = 8 },
    .{ .s = "ation", .substring_i = -1, .result = 7 },
    .{ .s = "ization", .substring_i = 17, .result = 6 },
    .{ .s = "izer", .substring_i = -1, .result = 6 },
    .{ .s = "ator", .substring_i = -1, .result = 7 },
    .{ .s = "iveness", .substring_i = -1, .result = 11 },
    .{ .s = "fulness", .substring_i = -1, .result = 9 },
    .{ .s = "ousness", .substring_i = -1, .result = 10 },
};

const A_6 = [_]Among{
    .{ .s = "icate", .substring_i = -1, .result = 4 },
    .{ .s = "ative", .substring_i = -1, .result = 6 },
    .{ .s = "alize", .substring_i = -1, .result = 3 },
    .{ .s = "iciti", .substring_i = -1, .result = 4 },
    .{ .s = "ical", .substring_i = -1, .result = 4 },
    .{ .s = "tional", .substring_i = -1, .result = 1 },
    .{ .s = "ational", .substring_i = 5, .result = 2 },
    .{ .s = "ful", .substring_i = -1, .result = 5 },
    .{ .s = "ness", .substring_i = -1, .result = 5 },
};

const A_7 = [_]Among{
    .{ .s = "ic", .substring_i = -1, .result = 1 },
    .{ .s = "ance", .substring_i = -1, .result = 1 },
    .{ .s = "ence", .substring_i = -1, .result = 1 },
    .{ .s = "able", .substring_i = -1, .result = 1 },
    .{ .s = "ible", .substring_i = -1, .result = 1 },
    .{ .s = "ate", .substring_i = -1, .result = 1 },
    .{ .s = "ive", .substring_i = -1, .result = 1 },
    .{ .s = "ize", .substring_i = -1, .result = 1 },
    .{ .s = "iti", .substring_i = -1, .result = 1 },
    .{ .s = "al", .substring_i = -1, .result = 1 },
    .{ .s = "ism", .substring_i = -1, .result = 1 },
    .{ .s = "ion", .substring_i = -1, .result = 2 },
    .{ .s = "er", .substring_i = -1, .result = 1 },
    .{ .s = "ous", .substring_i = -1, .result = 1 },
    .{ .s = "ant", .substring_i = -1, .result = 1 },
    .{ .s = "ent", .substring_i = -1, .result = 1 },
    .{ .s = "ment", .substring_i = 15, .result = 1 },
    .{ .s = "ement", .substring_i = 16, .result = 1 },
};

const A_8 = [_]Among{
    .{ .s = "e", .substring_i = -1, .result = 1 },
    .{ .s = "l", .substring_i = -1, .result = 2 },
};

const A_9 = [_]Among{
    .{ .s = "succeed", .substring_i = -1, .result = -1 },
    .{ .s = "proceed", .substring_i = -1, .result = -1 },
    .{ .s = "exceed", .substring_i = -1, .result = -1 },
    .{ .s = "canning", .substring_i = -1, .result = -1 },
    .{ .s = "inning", .substring_i = -1, .result = -1 },
    .{ .s = "earring", .substring_i = -1, .result = -1 },
    .{ .s = "herring", .substring_i = -1, .result = -1 },
    .{ .s = "outing", .substring_i = -1, .result = -1 },
};

const A_10 = [_]Among{
    .{ .s = "andes", .substring_i = -1, .result = -1 },
    .{ .s = "atlas", .substring_i = -1, .result = -1 },
    .{ .s = "bias", .substring_i = -1, .result = -1 },
    .{ .s = "cosmos", .substring_i = -1, .result = -1 },
    .{ .s = "dying", .substring_i = -1, .result = 3 },
    .{ .s = "early", .substring_i = -1, .result = 9 },
    .{ .s = "gently", .substring_i = -1, .result = 7 },
    .{ .s = "howe", .substring_i = -1, .result = -1 },
    .{ .s = "idly", .substring_i = -1, .result = 6 },
    .{ .s = "lying", .substring_i = -1, .result = 4 },
    .{ .s = "news", .substring_i = -1, .result = -1 },
    .{ .s = "only", .substring_i = -1, .result = 10 },
    .{ .s = "singly", .substring_i = -1, .result = 11 },
    .{ .s = "skies", .substring_i = -1, .result = 2 },
    .{ .s = "skis", .substring_i = -1, .result = 1 },
    .{ .s = "sky", .substring_i = -1, .result = -1 },
    .{ .s = "tying", .substring_i = -1, .result = 5 },
    .{ .s = "ugly", .substring_i = -1, .result = 8 },
};

const G_v = [_]u8{ 17, 65, 16, 1 };
const G_v_WXY = [_]u8{ 1, 17, 65, 208, 1 };
const G_valid_LI = [_]u8{ 55, 141, 2 };

const Context = struct {
    b_Y_found: bool,
    i_p2: usize,
    i_p1: usize,
};

fn r_prelude(env: *Env, ctx: *Context) bool {
    ctx.b_Y_found = false;
    const v_1 = env.cursor;
    lab0: while (true) {
        env.bra = env.cursor;
        if (!env.eq_s("'")) {
            break :lab0;
        }
        env.ket = env.cursor;
        if (!env.slice_del()) {
            return false;
        }
        break :lab0;
    }
    env.cursor = v_1;
    const v_2 = env.cursor;
    lab1: while (true) {
        env.bra = env.cursor;
        if (!env.eq_s("y")) {
            break :lab1;
        }
        env.ket = env.cursor;
        if (!env.slice_from("Y")) {
            return false;
        }
        ctx.b_Y_found = true;
        break :lab1;
    }
    env.cursor = v_2;
    const v_3 = env.cursor;
    lab2: while (true) {
        replab3: while (true) {
            const v_4 = env.cursor;
            lab4: for (0..1) |_| {
                golab5: while (true) {
                    const v_5 = env.cursor;
                    lab6: while (true) {
                        if (!env.in_grouping(&G_v, 97, 121)) {
                            break :lab6;
                        }
                        env.bra = env.cursor;
                        if (!env.eq_s("y")) {
                            break :lab6;
                        }
                        env.ket = env.cursor;
                        env.cursor = v_5;
                        break :golab5;
                    }
                    env.cursor = v_5;
                    if (env.cursor >= env.limit) {
                        break :lab4;
                    }
                    env.next_char();
                }
                if (!env.slice_from("Y")) {
                    return false;
                }
                ctx.b_Y_found = true;
                continue :replab3;
            }
            env.cursor = v_4;
            break :replab3;
        }
        break :lab2;
    }
    env.cursor = v_3;
    return true;
}

fn r_mark_regions(env: *Env, ctx: *Context) bool {
    ctx.i_p1 = env.limit;
    ctx.i_p2 = env.limit;
    const v_1 = env.cursor;
    lab0: while (true) {
        lab1: while (true) {
            const v_2 = env.cursor;
            lab2: while (true) {
                if (env.find_among(&A_0) == 0) {
                    break :lab2;
                }
                break :lab1;
            }
            env.cursor = v_2;
            golab3: while (true) {
                lab4: while (true) {
                    if (!env.in_grouping(&G_v, 97, 121)) {
                        break :lab4;
                    }
                    break :golab3;
                }
                if (env.cursor >= env.limit) {
                    break :lab0;
                }
                env.next_char();
            }
            golab5: while (true) {
                lab6: while (true) {
                    if (!env.out_grouping(&G_v, 97, 121)) {
                        break :lab6;
                    }
                    break :golab5;
                }
                if (env.cursor >= env.limit) {
                    break :lab0;
                }
                env.next_char();
            }
            break :lab1;
        }
        ctx.i_p1 = env.cursor;
        golab7: while (true) {
            lab8: while (true) {
                if (!env.in_grouping(&G_v, 97, 121)) {
                    break :lab8;
                }
                break :golab7;
            }
            if (env.cursor >= env.limit) {
                break :lab0;
            }
            env.next_char();
        }
        golab9: while (true) {
            lab10: while (true) {
                if (!env.out_grouping(&G_v, 97, 121)) {
                    break :lab10;
                }
                break :golab9;
            }
            if (env.cursor >= env.limit) {
                break :lab0;
            }
            env.next_char();
        }
        ctx.i_p2 = env.cursor;
        break :lab0;
    }
    env.cursor = v_1;
    return true;
}

fn r_shortv(env: *Env, ctx: *Context) bool {
    _ = ctx;
    lab0: while (true) {
        const v_1 = env.limit - env.cursor;
        lab1: while (true) {
            if (!env.out_grouping_b(&G_v_WXY, 89, 121)) {
                break :lab1;
            }
            if (!env.in_grouping_b(&G_v, 97, 121)) {
                break :lab1;
            }
            if (!env.out_grouping_b(&G_v, 97, 121)) {
                break :lab1;
            }
            break :lab0;
        }
        env.cursor = env.limit - v_1;
        if (!env.out_grouping_b(&G_v, 97, 121)) {
            return false;
        }
        if (!env.in_grouping_b(&G_v, 97, 121)) {
            return false;
        }
        if (env.cursor > env.limit_backward) {
            return false;
        }
        break :lab0;
    }
    return true;
}

fn r_R1(env: *Env, ctx: *Context) bool {
    if (!(ctx.i_p1 <= env.cursor)) {
        return false;
    }
    return true;
}

fn r_R2(env: *Env, ctx: *Context) bool {
    if (!(ctx.i_p2 <= env.cursor)) {
        return false;
    }
    return true;
}

fn r_Step_1a(env: *Env, ctx: *Context) bool {
    _ = ctx;
    var among_var: i32 = 0;
    const v_1 = env.limit - env.cursor;
    lab0: while (true) {
        env.ket = env.cursor;
        among_var = env.find_among_b(&A_1);
        if (among_var == 0) {
            env.cursor = env.limit - v_1;
            break :lab0;
        }
        env.bra = env.cursor;
        if (among_var == 0) {
            env.cursor = env.limit - v_1;
            break :lab0;
        } else if (among_var == 1) {
            if (!env.slice_del()) {
                return false;
            }
        }
        break :lab0;
    }
    env.ket = env.cursor;
    among_var = env.find_among_b(&A_2);
    if (among_var == 0) {
        return false;
    }
    env.bra = env.cursor;
    if (among_var == 0) {
        return false;
    } else if (among_var == 1) {
        if (!env.slice_from("ss")) {
            return false;
        }
    } else if (among_var == 2) {
        lab1: while (true) {
            const v_2 = env.limit - env.cursor;
            lab2: while (true) {
                const c = env.byte_index_for_hop(-2);
                if (@as(i32, @intCast(env.limit_backward)) > c or c > @as(i32, @intCast(env.limit))) {
                    break :lab2;
                }
                env.cursor = @intCast(c);
                if (!env.slice_from("i")) {
                    return false;
                }
                break :lab1;
            }
            env.cursor = env.limit - v_2;
            if (!env.slice_from("ie")) {
                return false;
            }
            break :lab1;
        }
    } else if (among_var == 3) {
        if (env.cursor <= env.limit_backward) {
            return false;
        }
        env.previous_char();
        golab3: while (true) {
            lab4: while (true) {
                if (!env.in_grouping_b(&G_v, 97, 121)) {
                    break :lab4;
                }
                break :golab3;
            }
            if (env.cursor <= env.limit_backward) {
                return false;
            }
            env.previous_char();
        }
        if (!env.slice_del()) {
            return false;
        }
    }
    return true;
}

fn r_Step_1b(env: *Env, ctx: *Context) bool {
    var among_var: i32 = 0;
    env.ket = env.cursor;
    among_var = env.find_among_b(&A_4);
    if (among_var == 0) {
        return false;
    }
    env.bra = env.cursor;
    if (among_var == 0) {
        return false;
    } else if (among_var == 1) {
        if (!r_R1(env, ctx)) {
            return false;
        }
        if (!env.slice_from("ee")) {
            return false;
        }
    } else if (among_var == 2) {
        const v_1 = env.limit - env.cursor;
        golab0: while (true) {
            lab1: while (true) {
                if (!env.in_grouping_b(&G_v, 97, 121)) {
                    break :lab1;
                }
                break :golab0;
            }
            if (env.cursor <= env.limit_backward) {
                return false;
            }
            env.previous_char();
        }
        env.cursor = env.limit - v_1;
        if (!env.slice_del()) {
            return false;
        }
        const v_3 = env.limit - env.cursor;
        among_var = env.find_among_b(&A_3);
        if (among_var == 0) {
            return false;
        }
        env.cursor = env.limit - v_3;
        if (among_var == 0) {
            return false;
        } else if (among_var == 1) {
            const c = env.cursor;
            const bra = env.cursor;
            const ket = env.cursor;
            env.insert(bra, ket, "e");
            env.cursor = c;
        } else if (among_var == 2) {
            env.ket = env.cursor;
            if (env.cursor <= env.limit_backward) {
                return false;
            }
            env.previous_char();
            env.bra = env.cursor;
            if (!env.slice_del()) {
                return false;
            }
        } else if (among_var == 3) {
            if (env.cursor != ctx.i_p1) {
                return false;
            }
            const v_4 = env.limit - env.cursor;
            if (!r_shortv(env, ctx)) {
                return false;
            }
            env.cursor = env.limit - v_4;
            const c = env.cursor;
            const bra = env.cursor;
            const ket = env.cursor;
            env.insert(bra, ket, "e");
            env.cursor = c;
        }
    }
    return true;
}

fn r_Step_1c(env: *Env, ctx: *Context) bool {
    _ = ctx;
    env.ket = env.cursor;
    lab0: while (true) {
        const v_1 = env.limit - env.cursor;
        lab1: while (true) {
            if (!env.eq_s_b("y")) {
                break :lab1;
            }
            break :lab0;
        }
        env.cursor = env.limit - v_1;
        if (!env.eq_s_b("Y")) {
            return false;
        }
        break :lab0;
    }
    env.bra = env.cursor;
    if (!env.out_grouping_b(&G_v, 97, 121)) {
        return false;
    }
    const v_2 = env.limit - env.cursor;
    lab2: while (true) {
        if (env.cursor > env.limit_backward) {
            break :lab2;
        }
        return false;
    }
    env.cursor = env.limit - v_2;
    if (!env.slice_from("i")) {
        return false;
    }
    return true;
}

fn r_Step_2(env: *Env, ctx: *Context) bool {
    var among_var: i32 = 0;
    env.ket = env.cursor;
    among_var = env.find_among_b(&A_5);
    if (among_var == 0) {
        return false;
    }
    env.bra = env.cursor;
    if (!r_R1(env, ctx)) {
        return false;
    }
    if (among_var == 0) {
        return false;
    } else if (among_var == 1) {
        if (!env.slice_from("tion")) {
            return false;
        }
    } else if (among_var == 2) {
        if (!env.slice_from("ence")) {
            return false;
        }
    } else if (among_var == 3) {
        if (!env.slice_from("ance")) {
            return false;
        }
    } else if (among_var == 4) {
        if (!env.slice_from("able")) {
            return false;
        }
    } else if (among_var == 5) {
        if (!env.slice_from("ent")) {
            return false;
        }
    } else if (among_var == 6) {
        if (!env.slice_from("ize")) {
            return false;
        }
    } else if (among_var == 7) {
        if (!env.slice_from("ate")) {
            return false;
        }
    } else if (among_var == 8) {
        if (!env.slice_from("al")) {
            return false;
        }
    } else if (among_var == 9) {
        if (!env.slice_from("ful")) {
            return false;
        }
    } else if (among_var == 10) {
        if (!env.slice_from("ous")) {
            return false;
        }
    } else if (among_var == 11) {
        if (!env.slice_from("ive")) {
            return false;
        }
    } else if (among_var == 12) {
        if (!env.slice_from("ble")) {
            return false;
        }
    } else if (among_var == 13) {
        if (!env.eq_s_b("l")) {
            return false;
        }
        if (!env.slice_from("og")) {
            return false;
        }
    } else if (among_var == 14) {
        if (!env.slice_from("ful")) {
            return false;
        }
    } else if (among_var == 15) {
        if (!env.slice_from("less")) {
            return false;
        }
    } else if (among_var == 16) {
        if (!env.in_grouping_b(&G_valid_LI, 99, 116)) {
            return false;
        }
        if (!env.slice_del()) {
            return false;
        }
    }
    return true;
}

fn r_Step_3(env: *Env, ctx: *Context) bool {
    var among_var: i32 = 0;
    env.ket = env.cursor;
    among_var = env.find_among_b(&A_6);
    if (among_var == 0) {
        return false;
    }
    env.bra = env.cursor;
    if (!r_R1(env, ctx)) {
        return false;
    }
    if (among_var == 0) {
        return false;
    } else if (among_var == 1) {
        if (!env.slice_from("tion")) {
            return false;
        }
    } else if (among_var == 2) {
        if (!env.slice_from("ate")) {
            return false;
        }
    } else if (among_var == 3) {
        if (!env.slice_from("al")) {
            return false;
        }
    } else if (among_var == 4) {
        if (!env.slice_from("ic")) {
            return false;
        }
    } else if (among_var == 5) {
        if (!env.slice_del()) {
            return false;
        }
    } else if (among_var == 6) {
        if (!r_R2(env, ctx)) {
            return false;
        }
        if (!env.slice_del()) {
            return false;
        }
    }
    return true;
}

fn r_Step_4(env: *Env, ctx: *Context) bool {
    var among_var: i32 = 0;
    env.ket = env.cursor;
    among_var = env.find_among_b(&A_7);
    if (among_var == 0) {
        return false;
    }
    env.bra = env.cursor;
    if (!r_R2(env, ctx)) {
        return false;
    }
    if (among_var == 0) {
        return false;
    } else if (among_var == 1) {
        if (!env.slice_del()) {
            return false;
        }
    } else if (among_var == 2) {
        lab0: while (true) {
            const v_1 = env.limit - env.cursor;
            lab1: while (true) {
                if (!env.eq_s_b("s")) {
                    break :lab1;
                }
                break :lab0;
            }
            env.cursor = env.limit - v_1;
            if (!env.eq_s_b("t")) {
                return false;
            }
            break :lab0;
        }
        if (!env.slice_del()) {
            return false;
        }
    }
    return true;
}

fn r_Step_5(env: *Env, ctx: *Context) bool {
    var among_var: i32 = 0;
    env.ket = env.cursor;
    among_var = env.find_among_b(&A_8);
    if (among_var == 0) {
        return false;
    }
    env.bra = env.cursor;
    if (among_var == 0) {
        return false;
    } else if (among_var == 1) {
        lab0: while (true) {
            const v_1 = env.limit - env.cursor;
            lab1: while (true) {
                if (!r_R2(env, ctx)) {
                    break :lab1;
                }
                break :lab0;
            }
            env.cursor = env.limit - v_1;
            if (!r_R1(env, ctx)) {
                return false;
            }
            const v_2 = env.limit - env.cursor;
            lab2: while (true) {
                if (!r_shortv(env, ctx)) {
                    break :lab2;
                }
                return false;
            }
            env.cursor = env.limit - v_2;
            break :lab0;
        }
        if (!env.slice_del()) {
            return false;
        }
    } else if (among_var == 2) {
        if (!r_R2(env, ctx)) {
            return false;
        }
        if (!env.eq_s_b("l")) {
            return false;
        }
        if (!env.slice_del()) {
            return false;
        }
    }
    return true;
}

fn r_exception2(env: *Env, ctx: *Context) bool {
    _ = ctx;
    env.ket = env.cursor;
    if (env.find_among_b(&A_9) == 0) {
        return false;
    }
    env.bra = env.cursor;
    if (env.cursor > env.limit_backward) {
        return false;
    }
    return true;
}

fn r_exception1(env: *Env, ctx: *Context) bool {
    _ = ctx;
    var among_var: i32 = 0;
    env.bra = env.cursor;
    among_var = env.find_among(&A_10);
    if (among_var == 0) {
        return false;
    }
    env.ket = env.cursor;
    if (env.cursor < env.limit) {
        return false;
    }
    if (among_var == 0) {
        return false;
    } else if (among_var == 1) {
        if (!env.slice_from("ski")) {
            return false;
        }
    } else if (among_var == 2) {
        if (!env.slice_from("sky")) {
            return false;
        }
    } else if (among_var == 3) {
        if (!env.slice_from("die")) {
            return false;
        }
    } else if (among_var == 4) {
        if (!env.slice_from("lie")) {
            return false;
        }
    } else if (among_var == 5) {
        if (!env.slice_from("tie")) {
            return false;
        }
    } else if (among_var == 6) {
        if (!env.slice_from("idl")) {
            return false;
        }
    } else if (among_var == 7) {
        if (!env.slice_from("gentl")) {
            return false;
        }
    } else if (among_var == 8) {
        if (!env.slice_from("ugli")) {
            return false;
        }
    } else if (among_var == 9) {
        if (!env.slice_from("earli")) {
            return false;
        }
    } else if (among_var == 10) {
        if (!env.slice_from("onli")) {
            return false;
        }
    } else if (among_var == 11) {
        if (!env.slice_from("singl")) {
            return false;
        }
    }
    return true;
}

fn r_postlude(env: *Env, ctx: *Context) bool {
    if (!ctx.b_Y_found) {
        return false;
    }
    replab0: while (true) {
        const v_1 = env.cursor;
        lab1: for (0..1) |_| {
            golab2: while (true) {
                const v_2 = env.cursor;
                lab3: while (true) {
                    env.bra = env.cursor;
                    if (!env.eq_s("Y")) {
                        break :lab3;
                    }
                    env.ket = env.cursor;
                    env.cursor = v_2;
                    break :golab2;
                }
                env.cursor = v_2;
                if (env.cursor >= env.limit) {
                    break :lab1;
                }
                env.next_char();
            }
            if (!env.slice_from("y")) {
                return false;
            }
            continue :replab0;
        }
        env.cursor = v_1;
        break :replab0;
    }
    return true;
}

/// Stem the word in `env` in place, as `Stemmer::stem` does.
pub fn stem(env: *Env) bool {
    var context = Context{ .b_Y_found = false, .i_p2 = 0, .i_p1 = 0 };
    const ctx = &context;
    lab0: while (true) {
        const v_1 = env.cursor;
        lab1: while (true) {
            if (!r_exception1(env, ctx)) {
                break :lab1;
            }
            break :lab0;
        }
        env.cursor = v_1;
        lab2: while (true) {
            const v_2 = env.cursor;
            lab3: while (true) {
                const c = env.byte_index_for_hop(3);
                if (0 > c or c > @as(i32, @intCast(env.limit))) {
                    break :lab3;
                }
                env.cursor = @intCast(c);
                break :lab2;
            }
            env.cursor = v_2;
            break :lab0;
        }
        env.cursor = v_1;
        const v_3 = env.cursor;
        lab4: while (true) {
            if (!r_prelude(env, ctx)) {
                break :lab4;
            }
            break :lab4;
        }
        env.cursor = v_3;
        const v_4 = env.cursor;
        lab5: while (true) {
            if (!r_mark_regions(env, ctx)) {
                break :lab5;
            }
            break :lab5;
        }
        env.cursor = v_4;
        env.limit_backward = env.cursor;
        env.cursor = env.limit;
        const v_5 = env.limit - env.cursor;
        lab6: while (true) {
            if (!r_Step_1a(env, ctx)) {
                break :lab6;
            }
            break :lab6;
        }
        env.cursor = env.limit - v_5;
        lab7: while (true) {
            const v_6 = env.limit - env.cursor;
            lab8: while (true) {
                if (!r_exception2(env, ctx)) {
                    break :lab8;
                }
                break :lab7;
            }
            env.cursor = env.limit - v_6;
            const v_7 = env.limit - env.cursor;
            lab9: while (true) {
                if (!r_Step_1b(env, ctx)) {
                    break :lab9;
                }
                break :lab9;
            }
            env.cursor = env.limit - v_7;
            const v_8 = env.limit - env.cursor;
            lab10: while (true) {
                if (!r_Step_1c(env, ctx)) {
                    break :lab10;
                }
                break :lab10;
            }
            env.cursor = env.limit - v_8;
            const v_9 = env.limit - env.cursor;
            lab11: while (true) {
                if (!r_Step_2(env, ctx)) {
                    break :lab11;
                }
                break :lab11;
            }
            env.cursor = env.limit - v_9;
            const v_10 = env.limit - env.cursor;
            lab12: while (true) {
                if (!r_Step_3(env, ctx)) {
                    break :lab12;
                }
                break :lab12;
            }
            env.cursor = env.limit - v_10;
            const v_11 = env.limit - env.cursor;
            lab13: while (true) {
                if (!r_Step_4(env, ctx)) {
                    break :lab13;
                }
                break :lab13;
            }
            env.cursor = env.limit - v_11;
            const v_12 = env.limit - env.cursor;
            lab14: while (true) {
                if (!r_Step_5(env, ctx)) {
                    break :lab14;
                }
                break :lab14;
            }
            env.cursor = env.limit - v_12;
            break :lab7;
        }
        env.cursor = env.limit_backward;
        const v_13 = env.cursor;
        lab15: while (true) {
            if (!r_postlude(env, ctx)) {
                break :lab15;
            }
            break :lab15;
        }
        env.cursor = v_13;
        break :lab0;
    }
    return true;
}
