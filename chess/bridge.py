"""
Browser bridge for Quantum Chess.

quantum_chess.py is copied unchanged from github.com/IDES0/QuantumChess
(commit ea7cef7). This module wraps it for the web page: every function takes
and returns plain values or JSON strings so the JavaScript side never touches
Python objects directly. It also caps the number of branches, since the
browser runs the engine on a single thread.
"""

import json
import math

import chess
from quantum_chess import QuantumChessGame, piece_id_name

# A split doubles the branches. Cost is ~0.3 ms per branch per action in the
# browser, so 4096 branches is roughly 1.3 s per action at worst.
MAX_BRANCHES = 4096

game = QuantumChessGame()


def _color(c: bool) -> str:
    return "White" if c == chess.WHITE else "Black"


def _loc(loc) -> str:
    return "captured" if loc is None else chess.square_name(loc)


def _amp(a: complex) -> str:
    """Amplitude as text, with exact forms for the common values (as in the GUI)."""
    a = complex(a)
    if abs(a.imag) > 1e-9:
        return f"{a.real:+.3f}{a.imag:+.3f}i"
    sign = "−" if a.real < -1e-10 else ""
    val = abs(a)
    for target, text in ((1.0, "1"), (1 / math.sqrt(2), "1/√2"), (0.5, "1/2"),
                         (0.5 / math.sqrt(2), "1/(2√2)"), (0.25, "1/4")):
        if abs(val - target) < 1e-6:
            return sign + text
    return f"{sign}{val:.4f}"


# ── LaTeX (same notation as the desktop GUI) ─────────────────────────────────

PIECE_LATEX = {
    'K': r'\mathrm{W\!K}', 'Q': r'\mathrm{W\!Q}', 'R': r'\mathrm{W\!R}',
    'B': r'\mathrm{W\!B}', 'N': r'\mathrm{W\!N}', 'P': r'\mathrm{W\!P}',
    'k': r'\mathrm{B\!K}', 'q': r'\mathrm{B\!Q}', 'r': r'\mathrm{B\!R}',
    'b': r'\mathrm{B\!B}', 'n': r'\mathrm{B\!N}', 'p': r'\mathrm{B\!P}',
}
MAX_TYPESET_TERMS = 8


def _amp_tex(amp) -> str:
    a = complex(amp)
    sign = "" if a.real >= -1e-10 else "-"
    val = abs(a)
    if val < 1e-10:
        return "0"
    if abs(a.imag) < 1e-10:
        s2 = 1 / math.sqrt(2)
        for target, tex in ((1.0, "1"), (s2, r"\frac{1}{\sqrt{2}}"), (0.5, r"\frac{1}{2}"),
                            (0.5 * s2, r"\frac{1}{2\sqrt{2}}"), (0.25, r"\frac{1}{4}")):
            if abs(val - target) < 1e-6:
                return sign + tex
        return f"{sign}{val:.4f}"
    return f"({a.real:.3f}{'+' if a.imag >= 0 else ''}{a.imag:.3f}i)"


def _piece_tex(pid: str) -> str:
    symbol, start = pid.split("@")
    return rf"\mathrm{{{symbol}}}_{{{start}}}"


def _loc_tex(loc) -> str:
    return r"\times" if loc is None else rf"\mathrm{{{chess.square_name(loc)}}}"


def _join_terms(terms) -> str:
    """Join braced terms with + (or − when a term's coefficient is negative)."""
    out = terms[0]
    for t in terms[1:]:
        out += r"\;-\;{" + t[2:] if t.startswith("{-") else r"\;+\;" + t
    return out


def square_info(square: str) -> dict:
    """The square's reduced state as a ket, plus who may be on it."""
    sq = chess.parse_square(square)
    name = square.upper()
    probs, neg, empty = {}, {}, 0.0
    for b in game.branches.values():
        p = b.get_probability()
        piece = b.board.piece_at(sq)
        if piece is None:
            empty += p
            continue
        sym = piece.symbol()
        probs[sym] = probs.get(sym, 0.0) + p
        neg[sym] = neg.get(sym, True) and complex(b.amplitude).real < 0
    parts = [rf"{{{_amp_tex(-math.sqrt(p) if neg[s] else math.sqrt(p))}\,|{PIECE_LATEX[s]}\rangle}}"
             for s, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p > 1e-10]
    if empty > 1e-10:
        parts.append(rf"{{{_amp_tex(math.sqrt(empty))}\,|\emptyset\rangle}}")
    occupied = sum(probs.values())
    on = game.pieces_on_square(sq)
    return {
        "square": square,
        "latex": rf"|\mathrm{{{name}}}\rangle \;=\; " + (_join_terms(parts) if parts else r"|\emptyset\rangle"),
        "occupied": occupied,
        "who": [{"name": piece_id_name(pid), "p": p}
                for pid, p in sorted(on.items(), key=lambda kv: -kv[1])],
        "empty": max(0.0, 1 - occupied),
    }


def _state_latex(rows, qp) -> dict:
    n = len(rows)
    if not qp:
        return {"title": "Classical position",
                "latex": r"|\psi\rangle \;=\; 1\,|\mathrm{board}\rangle",
                "subtitle": "No superpositions active."}
    if n <= MAX_TYPESET_TERMS:
        terms = []
        for amp, locs in rows:
            ket = r",\,".join(rf"{_piece_tex(pid)}\!:\!{_loc_tex(locs[pid])}" for pid in qp)
            terms.append(rf"{{{_amp_tex(amp)}\,|{ket}\rangle}}")
        return {"title": f"{n} branches",
                "latex": r"|\psi\rangle \;=\; " + _join_terms(terms),
                "subtitle": "Only pieces in superposition are written in each ket; × means captured."}
    return {"title": f"{n} branches",
            "latex": r"|\psi\rangle \;=\; \sum_{i=1}^{" + str(n) + r"} \alpha_i\,|b_i\rangle",
            "subtitle": f"Too many terms to typeset; the table lists them."}


def _outcome(ok: bool) -> str:
    out = {"ok": ok}
    if ok:
        out["notes"] = list(game.last_notes)
        m = game.last_mate_measurement
        if m is not None:
            out["mate"] = {"probability": m["probability"], "mate": m["mate"]}
            if game.result is not None:
                out["mate"]["winner"] = _color(game.result["winner"])
    else:
        out["error"] = game.last_error or "That action isn't possible."
    return json.dumps(out)


def _too_many(limit: int) -> str:
    return json.dumps({"ok": False, "error": (
        f"The board is already a superposition of {game.get_branch_count()} positions, "
        "which is the limit for running in a browser. Measure a square or merge a "
        "piece to collapse it first.")})


# ── Actions ──────────────────────────────────────────────────────────────────

def new_game() -> None:
    global game
    game = QuantumChessGame()


def move(frm: str, to: str) -> str:
    return _outcome(game.classical_move(frm + to))


def split(frm: str, t1: str, t2: str) -> str:
    if game.get_branch_count() * 2 > MAX_BRANCHES:
        return _too_many(MAX_BRANCHES)
    sq = chess.parse_square
    return _outcome(game.quantum_move_split(sq(frm), sq(t1), sq(t2)))


def merge(a: str, b: str, t: str) -> str:
    # A merge briefly doubles the branches holding the piece, then interference
    # recombines them, so it is allowed further past the cap than a split.
    if game.get_branch_count() > MAX_BRANCHES * 2:
        return _too_many(MAX_BRANCHES)
    sq = chess.parse_square
    return _outcome(game.quantum_merge(sq(a), sq(b), sq(t)))


def measure(square: str) -> str:
    ok, outcome = game.measure_square(chess.parse_square(square))
    result = json.loads(_outcome(ok))
    result["outcome"] = outcome
    return json.dumps(result)


def phase(square: str) -> str:
    return _outcome(game.apply_phase_shift(chess.parse_square(square), -1))


# ── Queries ──────────────────────────────────────────────────────────────────

def legal_targets(square: str) -> str:
    """Squares the current player's piece on `square` can reach in some branch."""
    s = chess.parse_square(square)
    color = game.get_current_turn()
    targets, occupied = set(), set()
    for branch in game.branches.values():
        board = branch.board
        occupied |= set(board.piece_map())
        p = board.piece_at(s)
        if p is not None and p.color == color:
            targets |= {m.to_square for m in board.legal_moves if m.from_square == s}
    return json.dumps([{"sq": chess.square_name(t), "occupied": t in occupied}
                       for t in sorted(targets)])


def piece_on(square: str):
    """The most likely piece on a square, or None."""
    on = game.pieces_on_square(chess.parse_square(square))
    return max(on, key=on.get) if on else None


def view(inspect_pid=None, inspect_square=None) -> str:
    """Everything the page needs to draw the board and the side panel."""
    branches = list(game.branches.values())
    turn = game.get_current_turn()

    # Per square and piece symbol: total probability, and whether every branch
    # holding it has a negative phase.
    squares = {}
    for b in branches:
        p = b.get_probability()
        negative = complex(b.amplitude).real < 0
        for sq, piece in b.board.piece_map().items():
            entry = squares.setdefault(chess.square_name(sq), {}).setdefault(
                piece.symbol(), [0.0, True])
            entry[0] += p
            entry[1] = entry[1] and negative
    squares = {name: sorted(({"s": sym, "p": v[0], "neg": v[1]} for sym, v in syms.items()),
                            key=lambda e: -e["p"])
               for name, syms in squares.items()}

    # Where the king of the side to move is in check (it may be superposed).
    check = {}
    if game.result is None:
        for b in branches:
            if b.board.is_check():
                k = b.board.king(turn)
                if k is not None:
                    name = chess.square_name(k)
                    check[name] = check.get(name, 0.0) + b.get_probability()

    pieces = []
    for pid in game.piece_ids():
        dist = game.piece_distribution(pid)
        dead = dist.get(None, 0.0)
        if game.is_quantum_piece(pid) or dead > 1e-9:
            pieces.append({
                "pid": pid, "name": piece_id_name(pid), "alive": 1 - dead,
                "where": [{"loc": _loc(l), "p": p}
                          for l, p in sorted(dist.items(), key=lambda kv: -kv[1])],
            })

    qp = game.quantum_pieces()
    rows = game.branch_summaries()
    state = {
        **_state_latex(rows, qp),
        "pieces": [piece_id_name(pid) for pid in qp],
        "rows": [{"amp": _amp(a), "p": abs(a) ** 2,
                  "locs": [_loc(locs[pid]) for pid in qp]} for a, locs in rows[:24]],
        "hidden": max(0, len(rows) - 24),
    }

    out = {
        "turn": _color(turn),
        "squares": squares,
        "check": check,
        "checkTotal": sum(check.values()),
        "branches": len(branches),
        "maxBranches": MAX_BRANCHES,
        "canPhase": game.can_phase_shift(),
        "result": None if game.result is None else {
            "winner": _color(game.result["winner"]),
            "probability": game.result["probability"]},
        "pieces": pieces,
        "state": state,
        "inspected": _inspect(inspect_pid) if inspect_pid else None,
        "square": square_info(inspect_square) if inspect_square else None,
    }
    return json.dumps(out)


def _inspect(pid: str) -> dict:
    dist = game.piece_distribution(pid)
    where = []
    for loc, p in sorted(dist.items(), key=lambda kv: -kv[1]):
        signs = {complex(b.amplitude).real < 0 for b in game.branches.values()
                 if game._location(b, pid) == loc}
        where.append({"loc": _loc(loc), "p": p,
                      "phase": "−" if signs == {True} else "+" if signs == {False} else "±"})
    corr = []
    for c in game.piece_correlations(pid)[:4]:
        corr.append({
            "name": piece_id_name(c["piece"]),
            "bits": c["information"],
            "given": [{"loc": _loc(x),
                       "then": [{"loc": _loc(y), "p": p}
                                for y, p in sorted(ys.items(), key=lambda kv: -kv[1])]}
                      for x, ys in c["conditional"].items()],
        })
    symbol = None
    for loc in dist:
        if loc is not None:
            symbol = game.piece_symbol_at(pid, loc)
            break
    return {
        "pid": pid,
        "name": piece_id_name(pid),
        "symbol": symbol,
        "quantum": game.is_quantum_piece(pid),
        "alive": 1 - dist.get(None, 0.0),
        "where": where,
        "squares": [w["loc"] for w in where if w["loc"] != "captured"],
        "correlations": corr,
    }
