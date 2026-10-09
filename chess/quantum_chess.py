"""
Quantum Chess Implementation

The game state is a superposition of ordinary chess positions ("branches"),
each carrying a complex amplitude.  Every action is applied branch by branch.
A branch in which the action is impossible simply passes its turn, so all
branches always share the same side to move and Σ|amplitude|² stays 1.

Every piece carries an identity (e.g. "N@g1" = the knight that started on g1)
that follows it through moves, so a single piece can be tracked across the
whole superposition.
"""

import chess
import math
import random
import uuid
from typing import Dict, Set, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}

# A location of a piece: a square, or None when it has been captured.
Location = Optional[chess.Square]


def initial_piece_ids(board: chess.Board) -> Dict[chess.Square, str]:
    """Name each piece after its symbol and starting square, e.g. 'N@g1'."""
    return {sq: f"{p.symbol()}@{chess.square_name(sq)}"
            for sq, p in board.piece_map().items()}


def piece_id_name(pid: str) -> str:
    """'N@g1' -> 'White knight (g1)'."""
    symbol, start = pid.split("@")
    piece = chess.Piece.from_symbol(symbol)
    color = "White" if piece.color == chess.WHITE else "Black"
    return f"{color} {PIECE_NAMES[piece.piece_type]} ({start})"


@dataclass
class QuantumBranch:
    """A single classical position in the superposition."""
    branch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    amplitude: complex = 1.0
    board: chess.Board = field(default_factory=chess.Board)
    # square -> identity of the piece standing there
    piece_ids: Dict[chess.Square, str] = field(default_factory=dict)

    def copy(self) -> 'QuantumBranch':
        return QuantumBranch(
            branch_id=str(uuid.uuid4()),
            amplitude=self.amplitude,
            board=self.board.copy(),
            piece_ids=dict(self.piece_ids),
        )

    def get_probability(self) -> float:
        return abs(self.amplitude) ** 2

    def push(self, move: chess.Move) -> None:
        """Play a move (or null move), carrying piece identities along."""
        if move:
            board, ids = self.board, self.piece_ids
            mover = ids.pop(move.from_square, None)
            if board.is_castling(move):
                # Accepts both e1g1 and "king takes own rook" e1h1 notation.
                rank = chess.square_rank(move.from_square)
                kingside = board.is_kingside_castling(move)
                target = board.piece_at(move.to_square)
                if target is not None and target.piece_type == chess.ROOK:
                    rook_from = move.to_square
                else:
                    rook_from = chess.square(7 if kingside else 0, rank)
                rook = ids.pop(rook_from, None)
                ids[chess.square(6 if kingside else 2, rank)] = mover
                ids[chess.square(5 if kingside else 3, rank)] = rook
            else:
                if board.is_en_passant(move):
                    ids.pop(move.to_square + (-8 if board.turn == chess.WHITE else 8), None)
                else:
                    ids.pop(move.to_square, None)
                if mover is not None:
                    ids[move.to_square] = mover
        self.board.push(move)

    def relocate(self, from_square: chess.Square, to_square: chess.Square) -> None:
        """Move a piece without a chess move (used by the merge's bounce leg)."""
        piece = self.board.remove_piece_at(from_square)
        self.board.set_piece_at(to_square, piece)
        self.board.castling_rights = self.board.clean_castling_rights()
        pid = self.piece_ids.pop(from_square, None)
        if pid is not None:
            self.piece_ids[to_square] = pid

    def key(self) -> tuple:
        """Branches with equal keys are the same basis state (move clocks ignored)."""
        b = self.board
        return (b.board_fen(), b.turn, b.castling_rights, b.ep_square,
                frozenset(self.piece_ids.items()))


class GameStateManager:
    """
    Holds the superposition and an occupancy cache (square -> branch ids with
    a piece on it) for fast probability lookups.
    """

    def __init__(self, fen: Optional[str] = None):
        self.branches: Dict[str, QuantumBranch] = {}
        self.occupancy_cache: Dict[chess.Square, Set[str]] = defaultdict(set)

        board = chess.Board(fen) if fen else chess.Board()
        initial = QuantumBranch(board=board, piece_ids=initial_piece_ids(board))
        self.set_branches({initial.branch_id: initial})

    def set_branches(self, branches: Dict[str, QuantumBranch]) -> None:
        """Replace the state vector and rebuild the occupancy cache."""
        self.branches = branches
        self.occupancy_cache = defaultdict(set)
        for branch_id, branch in branches.items():
            for square in branch.board.piece_map():
                self.occupancy_cache[square].add(branch_id)

    def get_occupancy_probability(self, square: chess.Square) -> float:
        """Probability that a square is occupied."""
        return sum(self.branches[bid].get_probability()
                   for bid in self.occupancy_cache.get(square, ()))

    def get_occupancy_amplitude(self, square: chess.Square) -> complex:
        """Sum of amplitudes of the branches where the square is occupied."""
        return sum(self.branches[bid].amplitude
                   for bid in self.occupancy_cache.get(square, ()))

    def _normalize_amplitudes(self) -> None:
        """Renormalize amplitudes so total probability is 1 (after a measurement)."""
        total_prob = sum(b.get_probability() for b in self.branches.values())
        if total_prob > 1e-10:
            factor = 1.0 / math.sqrt(total_prob)
            for branch in self.branches.values():
                branch.amplitude *= factor

    def combine_identical_branches(self, coherent: bool) -> None:
        """
        Branches that reached the same position are combined.

        coherent=True  (unitary actions such as the merge): amplitudes ADD, so
                       opposite phases cancel -- this is interference.
        coherent=False (ordinary moves): two different histories can collapse
                       onto one position (e.g. a capture erases what told them
                       apart).  That is not a reversible process, so the
                       probabilities add instead and no interference occurs.
        """
        groups: Dict[tuple, List[QuantumBranch]] = defaultdict(list)
        for branch in self.branches.values():
            groups[branch.key()].append(branch)

        combined: Dict[str, QuantumBranch] = {}
        for members in groups.values():
            head = members[0]
            if len(members) > 1:
                if coherent:
                    head.amplitude = sum(b.amplitude for b in members)
                else:
                    lead = max(members, key=lambda b: b.get_probability())
                    phase = lead.amplitude / abs(lead.amplitude)
                    head.amplitude = math.sqrt(sum(b.get_probability() for b in members)) * phase
            if abs(head.amplitude) > 1e-9:
                combined[head.branch_id] = head
        self.set_branches(combined)


class QuantumChessGame:
    """
    Quantum Chess rules.

    A turn is one main action -- move, split, merge or measurement -- plus an
    optional free phase shift beforehand.  After every action, if the side to
    move is checkmated (or has lost its king) in some branches, the question
    "is it mate?" is measured automatically.
    """

    def __init__(self, fen: Optional[str] = None):
        self.state_manager = GameStateManager(fen)
        self._initial_id_list = sorted(
            next(iter(self.branches.values())).piece_ids.values())
        # Count of main actions; each one ends a turn.
        self.ply = 0
        # ply on which the free phase shift was last used (one per turn).
        self._phase_used_ply: Optional[int] = None
        # Set when the game ends: {'winner': chess.WHITE/BLACK, 'probability': p}
        self.result: Optional[dict] = None
        # Outcome of the automatic mate measurement after the last action:
        # {'probability': p, 'mate': bool}, or None if no branch was mate.
        self.last_mate_measurement: Optional[dict] = None
        # Why the last rejected action failed.
        self.last_error: str = ""
        # Informational notes about the last successful action.
        self.last_notes: List[str] = []

    # ==================== HELPERS ====================

    ABSENT = "the piece isn't there"

    @property
    def branches(self) -> Dict[str, QuantumBranch]:
        return self.state_manager.branches

    def get_current_turn(self) -> bool:
        """True = White to move.  All branches share the side to move."""
        return next(iter(self.branches.values())).board.turn

    def _turn_name(self) -> str:
        return "White" if self.get_current_turn() == chess.WHITE else "Black"

    def _wrong_turn(self) -> str:
        mover = self._turn_name()
        other = "Black" if mover == "White" else "White"
        return (f"It's {mover}'s turn — that piece belongs to {other}. "
                f"{other} has to wait until {mover} has moved.")

    def _fail(self, reason: str) -> bool:
        self.last_error = reason
        return False

    @staticmethod
    def _make_move(board: chess.Board, from_square: chess.Square,
                   to_square: chess.Square) -> chess.Move:
        """Build the move from -> to, auto-promoting pawns to queens."""
        move = chess.Move(from_square, to_square)
        piece = board.piece_at(from_square)
        if (piece is not None and piece.piece_type == chess.PAWN
                and chess.square_rank(to_square) in (0, 7)):
            move.promotion = chess.QUEEN
        return move

    @staticmethod
    def _why_illegal(board: chess.Board, move: chess.Move) -> str:
        """Plain-English reason a move is illegal on one classical board."""
        frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
        piece = board.piece_at(move.from_square)
        if piece is None:
            return QuantumChessGame.ABSENT
        name = PIECE_NAMES[piece.piece_type]
        if move in board.pseudo_legal_moves:
            if board.is_check():
                return "your king is in check and this doesn't stop it"
            return "it would expose your king to check"
        target = board.piece_at(move.to_square)
        if target is not None and target.color == piece.color:
            return f"your own {PIECE_NAMES[target.piece_type]} is on {to}"
        if piece.piece_type == chess.PAWN:
            if chess.square_file(move.from_square) != chess.square_file(move.to_square):
                return f"pawns only move diagonally to capture, and {to} is empty"
            return "the pawn is blocked"
        if piece.piece_type == chess.KING and abs(move.from_square - move.to_square) == 2:
            return "castling isn't possible (rights lost, path blocked or attacked)"
        lone = chess.Board(None)
        lone.set_piece_at(move.from_square, piece)
        lone.turn = piece.color
        if move in lone.pseudo_legal_moves:
            return f"the path from {frm} to {to} is blocked"
        return f"a {name} can't move from {frm} to {to}"

    @staticmethod
    def _summarize_reasons(reasons: Dict[str, float]) -> str:
        if len(reasons) == 1:
            reason = next(iter(reasons))
            return reason[0].upper() + reason[1:] + "."
        ranked = sorted(reasons.items(), key=lambda kv: -kv[1])
        return "Illegal in every branch:\n" + "\n".join(
            f"  • {r} ({p:.0%} of the time)" for r, p in ranked)

    def _own_piece_at(self, square: chess.Square) -> Tuple[Optional[chess.Piece], str]:
        """The current player's piece on `square` in any branch, or a reason why not."""
        color = self.get_current_turn()
        other = None
        for branch in self.branches.values():
            p = branch.board.piece_at(square)
            if p is not None:
                if p.color == color:
                    return p, ""
                other = p
        name = chess.square_name(square)
        if other is not None:
            return None, self._wrong_turn()
        return None, f"You have no piece on {name}."

    def _note_partial(self, what: str, square: chess.Square, passed: float,
                      reasons: Dict[str, float]) -> None:
        """Explain an action that only happened in some branches."""
        if passed < 1e-9:
            return
        absent = reasons.get(self.ABSENT, 0.0)
        if absent > 1e-9:
            self.last_notes.append(
                f"Your piece is on {chess.square_name(square)} only {1 - absent:.0%} of the "
                f"time, so the {what} only happened there.")
        other = {r: p for r, p in reasons.items() if r != self.ABSENT}
        if other:
            self.last_notes.append(
                f"In {sum(other.values()):.0%} of the branches the {what} was impossible "
                f"({'; '.join(other)}), so your turn passed there.")

    def _self_overlap(self, from_square: chess.Square, to_square: chess.Square) -> bool:
        """Could the piece on from_square already be on to_square in another branch?"""
        color = self.get_current_turn()
        movers = {b.piece_ids.get(from_square) for b in self.branches.values()
                  if (p := b.board.piece_at(from_square)) is not None and p.color == color}
        return any(b.piece_ids.get(to_square) in movers for b in self.branches.values())

    def _start_action(self) -> bool:
        self.last_error = ""
        self.last_notes = []
        if self.result is not None:
            return self._fail("The game is over.")
        return True

    def _finish_action(self, coherent: bool = False) -> None:
        """Bookkeeping after every main action: combine, end turn, test for mate."""
        self.state_manager.combine_identical_branches(coherent)
        self.ply += 1
        self.last_error = ""
        self._resolve_checkmate()
        exposed = self.get_exposed_king_probability()
        if exposed > 1e-9 and self.result is None:
            self.last_notes.append(
                f"Your king is still attacked in {exposed:.0%} of the branches -- "
                "your opponent can capture it there.")

    @staticmethod
    def _is_lost(board: chess.Board) -> bool:
        """Side to move is checkmated, or its king was captured in this branch."""
        return board.king(board.turn) is None or board.is_checkmate()

    def _resolve_checkmate(self) -> None:
        """
        If checkmate exists in some branches, measure the single observable
        "is the side to move checkmated?":
          - mate     -> keep the mate branches; game over.
          - not mate -> keep the other branches, renormalize, play continues.
        """
        self.last_mate_measurement = None
        p_mate = sum(b.get_probability() for b in self.branches.values()
                     if self._is_lost(b.board))
        if p_mate < 1e-9:
            return

        is_mate = random.random() < p_mate
        kept = {bid: b for bid, b in self.branches.items()
                if self._is_lost(b.board) == is_mate}
        self.state_manager.set_branches(kept)
        self.state_manager._normalize_amplitudes()

        self.last_mate_measurement = {'probability': p_mate, 'mate': is_mate}
        if is_mate:
            loser = next(iter(kept.values())).board.turn
            self.result = {'winner': not loser, 'probability': p_mate}

    # ==================== STATUS ====================

    def get_check_probability(self) -> float:
        """Probability that the side to move is in check."""
        return sum(b.get_probability() for b in self.branches.values()
                   if b.board.is_check())

    def get_exposed_king_probability(self) -> float:
        """Probability that the side who just moved left its king capturable."""
        return sum(b.get_probability() for b in self.branches.values()
                   if b.board.was_into_check())

    def is_quantum_check(self) -> bool:
        return self.get_check_probability() > 1e-9

    def can_phase_shift(self) -> bool:
        """The free phase shift is available once per turn."""
        return self.result is None and self._phase_used_ply != self.ply

    def get_branch_count(self) -> int:
        return len(self.branches)

    # ==================== MAIN ACTIONS ====================

    def classical_move(self, move_uci: str) -> bool:
        """
        Make an ordinary move in every branch where it is legal; branches where
        it is not (piece absent, path blocked by a superposed piece, own king in
        check, ...) pass their turn.  Capturing a superposed piece therefore
        captures it exactly in the branches where it is present.
        """
        if not self._start_action():
            return False
        try:
            parsed = chess.Move.from_uci(move_uci)
        except ValueError:
            return self._fail(f"'{move_uci}' is not a move.")
        frm, to = parsed.from_square, parsed.to_square

        piece, reason = self._own_piece_at(frm)
        if piece is None:
            return self._fail(reason)
        if self._self_overlap(frm, to):
            return self._fail(
                f"Part of this same {PIECE_NAMES[piece.piece_type]} is already on "
                f"{chess.square_name(to)}. Use Quantum Merge to recombine its halves.")

        reasons: Dict[str, float] = defaultdict(float)
        passed = 0.0
        for branch in self.branches.values():
            move = self._make_move(branch.board, frm, to)
            if move not in branch.board.legal_moves:
                reasons[self._why_illegal(branch.board, move)] += branch.get_probability()
                passed += branch.get_probability()
        if passed > 1 - 1e-9:
            return self._fail(self._summarize_reasons(reasons))

        for branch in self.branches.values():
            move = self._make_move(branch.board, frm, to)
            branch.push(move if move in branch.board.legal_moves else chess.Move.null())

        self._note_partial("move", frm, passed, reasons)
        self._finish_action()
        return True

    def quantum_move_split(self, from_square: chess.Square, to_square1: chess.Square,
                           to_square2: chess.Square) -> bool:
        """
        Split: move a piece into an equal superposition of two squares.
        Each leg must be a legal move (captures included).  Branches where the
        piece is absent, or either leg is impossible, pass their turn.
        """
        if not self._start_action():
            return False
        if len({from_square, to_square1, to_square2}) < 3:
            return self._fail("Pick two different destination squares.")

        piece, reason = self._own_piece_at(from_square)
        if piece is None:
            return self._fail(reason)
        for t in (to_square1, to_square2):
            if self._self_overlap(from_square, t):
                return self._fail(
                    f"Part of this same {PIECE_NAMES[piece.piece_type]} is already on "
                    f"{chess.square_name(t)}. Use Quantum Merge to recombine its halves.")

        amp = 1 / math.sqrt(2)
        new_branches: Dict[str, QuantumBranch] = {}
        reasons: Dict[str, float] = defaultdict(float)
        passed = 0.0
        for branch in self.branches.values():
            board = branch.board
            legs = [self._make_move(board, from_square, t) for t in (to_square1, to_square2)]
            bad = [m for m in legs if m not in board.legal_moves]
            if board.piece_at(from_square) != piece:
                reasons[self.ABSENT] += branch.get_probability()
            elif bad:
                reasons[f"can't go to {chess.square_name(bad[0].to_square)}: "
                        f"{self._why_illegal(board, bad[0])}"] += branch.get_probability()
            else:
                for move in legs:
                    child = branch.copy()
                    child.amplitude = branch.amplitude * amp
                    child.push(move)
                    new_branches[child.branch_id] = child
                continue
            passed += branch.get_probability()
            child = branch.copy()
            child.push(chess.Move.null())
            new_branches[child.branch_id] = child

        if passed > 1 - 1e-9:
            return self._fail(self._summarize_reasons(reasons))

        self._note_partial("split", from_square, passed, reasons)
        self.state_manager.set_branches(new_branches)
        self._finish_action()
        return True

    def quantum_merge(self, from_square1: chess.Square, from_square2: chess.Square,
                      to_square: chess.Square) -> bool:
        """
        Merge: recombine the two halves of a split piece onto one empty square.

        A beam splitter on the piece's position (t = target, a = from_square1,
        b = from_square2):
            |a>  ->  ( |t> + |a> ) / √2
            |b>  ->  ( |t> - |a> ) / √2
        Halves with equal phase, (|a> + |b>)/√2, interfere into |t> exactly.
        If one half's phase was flipped, (|a> - |b>)/√2, they interfere into |a>:
        the merge "bounces" and the whole piece lands on from_square1.
        Branches without the piece pass their turn.
        """
        if not self._start_action():
            return False
        if len({from_square1, from_square2, to_square}) < 3:
            return self._fail("Pick two different halves and a different target square.")
        if self.state_manager.get_occupancy_probability(to_square) > 1e-9:
            return self._fail("The merge target must be empty in every branch.")

        color = self.get_current_turn()
        found: Dict[chess.Square, Set[str]] = {from_square1: set(), from_square2: set()}
        for branch in self.branches.values():
            for sq in found:
                p = branch.board.piece_at(sq)
                if p is not None and p.color == color:
                    found[sq].add(branch.piece_ids[sq])
        common = found[from_square1] & found[from_square2]
        if not found[from_square1] and not found[from_square2] and any(
                b.board.piece_at(sq) is not None
                for b in self.branches.values() for sq in found):
            return self._fail(self._wrong_turn())
        if len(common) != 1:
            return self._fail("Those squares don't hold two halves of one of your pieces. "
                              "Pick both squares a split piece may be on.")
        pid = common.pop()

        amp = 1 / math.sqrt(2)
        a, b, t = from_square1, from_square2, to_square
        new_branches: Dict[str, QuantumBranch] = {}

        # The merged piece gives up its castling rights in every outcome, so
        # the halves differ only in position and can interfere cleanly.
        rights_mask = chess.BB_SQUARES[a] | chess.BB_SQUARES[b] | chess.BB_SQUARES[t]
        if pid[0].upper() == "K":
            rights_mask |= chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8

        def add(src: QuantumBranch, factor: float, edit, merged: bool = True) -> None:
            child = src.copy()
            child.amplitude = src.amplitude * factor
            edit(child)
            if merged:
                child.board.castling_rights &= ~rights_mask
            new_branches[child.branch_id] = child

        for branch in self.branches.values():
            board = branch.board
            on_a = branch.piece_ids.get(a) == pid
            on_b = branch.piece_ids.get(b) == pid
            if not (on_a or on_b):
                add(branch, 1.0, lambda c: c.push(chess.Move.null()), merged=False)
                continue

            src = a if on_a else b
            leg = self._make_move(board, src, t)
            if leg not in board.legal_moves:
                return self._fail(
                    f"{chess.square_name(src)} → {chess.square_name(t)}: "
                    f"{self._why_illegal(board, leg)} (in some branches).")
            if on_b and board.piece_at(a) is not None:
                return self._fail(
                    f"{chess.square_name(a)} must be empty wherever the piece is on "
                    f"{chess.square_name(b)} (a failed merge lands the piece there).")
            add(branch, amp, lambda c, m=leg: c.push(m))

            def bounce(c: QuantumBranch, src=src) -> None:
                if src != a:
                    c.relocate(src, a)
                c.push(chess.Move.null())
            add(branch, amp if on_a else -amp, bounce)

        for child in new_branches.values():
            if child.board.was_into_check():
                return self._fail("The merge could leave your king in check.")

        # Safety net: interference must conserve probability.  If the halves
        # still differ in some hidden way, refuse rather than corrupt the state.
        sums: Dict[tuple, complex] = defaultdict(complex)
        for child in new_branches.values():
            sums[child.key()] += child.amplitude
        if abs(sum(abs(v) ** 2 for v in sums.values()) - 1) > 1e-9:
            return self._fail("These halves differ in more than their position "
                              "(e.g. castling or en-passant rights), so they can't be "
                              "merged.")

        self.state_manager.set_branches(new_branches)
        self._finish_action(coherent=True)
        landed = sum(br.get_probability() for br in self.branches.values()
                     if br.piece_ids.get(t) == pid)
        if landed < 1 - 1e-9:
            self.last_notes.insert(0,
                f"The piece landed on {chess.square_name(t)} with probability {landed:.0%}; "
                f"the rest bounced to {chess.square_name(a)}.")
        return True

    def measure_square(self, square: chess.Square) -> Tuple[bool, str]:
        """
        Measure what is on a square -- which kind of piece, or empty -- and
        collapse every branch that disagrees.  Uses your turn.
        Returns (success, outcome) where outcome is e.g. "White bishop" or "Empty".
        """
        if not self._start_action():
            return False, ""

        def occupant(branch: QuantumBranch) -> Optional[chess.Piece]:
            return branch.board.piece_at(square)

        outcomes: Dict[Optional[chess.Piece], float] = defaultdict(float)
        for branch in self.branches.values():
            outcomes[occupant(branch)] += branch.get_probability()
        if max(outcomes.values()) > 1 - 1e-9:
            return self._fail("That square's contents are already certain, so there is "
                              "nothing to measure."), ""

        roll, result = random.random(), None
        for result, p in outcomes.items():
            roll -= p
            if roll < 0:
                break
        kept = {}
        for bid, branch in self.branches.items():
            if occupant(branch) == result:
                branch.push(chess.Move.null())
                kept[bid] = branch
        self.state_manager.set_branches(kept)
        self.state_manager._normalize_amplitudes()
        self._finish_action()
        if result is None:
            return True, "Empty"
        color = "White" if result.color == chess.WHITE else "Black"
        return True, f"{color} {PIECE_NAMES[result.piece_type]}"

    # ==================== FREE ACTION ====================

    def apply_phase_shift(self, square: chess.Square, phase_multiplier: complex = -1) -> bool:
        """
        Free action (once per turn, before your main action): multiply the
        amplitude of every branch where your piece is on `square` by
        `phase_multiplier`.  Only meaningful on a piece in superposition, and
        it only shows up when the halves are recombined with quantum_merge.
        """
        self.last_notes = []
        if self.result is not None:
            return self._fail("The game is over.")
        if not self.can_phase_shift():
            return self._fail("You already used your phase shift this turn.")

        color = self.get_current_turn()
        prob = self.state_manager.get_occupancy_probability(square)
        if not (1e-9 < prob < 1 - 1e-9):
            return self._fail("Phase shifts only affect a piece in superposition.")

        targets = [b for b in self.branches.values() if b.board.piece_at(square) is not None]
        if any(b.board.piece_at(square).color != color for b in targets):
            return self._fail(self._wrong_turn())

        for branch in targets:
            branch.amplitude *= phase_multiplier
        self._phase_used_ply = self.ply
        self.last_error = ""
        return True

    # ==================== PIECE TRACKING ====================

    def piece_ids(self) -> List[str]:
        """Every piece that started the game (captured ones included)."""
        return self._initial_id_list

    @staticmethod
    def _location(branch: QuantumBranch, pid: str) -> Location:
        for sq, other in branch.piece_ids.items():
            if other == pid:
                return sq
        return None

    def piece_distribution(self, pid: str) -> Dict[Location, float]:
        """Where a piece may be: {square or None (captured): probability}."""
        dist: Dict[Location, float] = defaultdict(float)
        for branch in self.branches.values():
            dist[self._location(branch, pid)] += branch.get_probability()
        return dict(dist)

    def piece_symbol_at(self, pid: str, square: chess.Square) -> Optional[str]:
        """Current symbol of a piece on a square (differs after promotion)."""
        for branch in self.branches.values():
            if branch.piece_ids.get(square) == pid:
                return branch.board.piece_at(square).symbol()
        return None

    def is_quantum_piece(self, pid: str) -> bool:
        """True unless the piece is on one square (or captured) with certainty."""
        return max(self.piece_distribution(pid).values()) < 1 - 1e-9

    def quantum_pieces(self) -> List[str]:
        return [pid for pid in self.piece_ids() if self.is_quantum_piece(pid)]

    def pieces_on_square(self, square: chess.Square) -> Dict[str, float]:
        """{piece id: probability it is on `square`}."""
        result: Dict[str, float] = defaultdict(float)
        for branch in self.branches.values():
            pid = branch.piece_ids.get(square)
            if pid is not None:
                result[pid] += branch.get_probability()
        return dict(result)

    def piece_correlations(self, pid: str) -> List[dict]:
        """
        Other pieces entangled with `pid`: knowing where one is tells you
        something about the other.  For each, returns the mutual information
        (bits) and, for every location of `pid`, the conditional distribution
        of the other piece.
        """
        out = []
        locs = {bid: self._location(b, pid) for bid, b in self.branches.items()}
        mine = self.piece_distribution(pid)
        for other in self.quantum_pieces():
            if other == pid:
                continue
            joint: Dict[Tuple[Location, Location], float] = defaultdict(float)
            for bid, branch in self.branches.items():
                joint[(locs[bid], self._location(branch, other))] += branch.get_probability()
            theirs = self.piece_distribution(other)
            info = sum(p * math.log2(p / (mine[x] * theirs[y]))
                       for (x, y), p in joint.items() if p > 1e-12)
            if info < 1e-6:
                continue
            conditional: Dict[Location, Dict[Location, float]] = defaultdict(dict)
            for (x, y), p in joint.items():
                conditional[x][y] = p / mine[x]
            out.append({'piece': other, 'information': info,
                        'conditional': dict(conditional)})
        return sorted(out, key=lambda c: -c['information'])

    def branch_summaries(self) -> List[Tuple[complex, Dict[str, Location]]]:
        """Each branch's amplitude and the locations of the quantum pieces in it."""
        qp = self.quantum_pieces()
        rows = [(b.amplitude, {pid: self._location(b, pid) for pid in qp})
                for b in self.branches.values()]
        return sorted(rows, key=lambda r: -abs(r[0]))

    def get_piece_survival_probabilities(self) -> Dict[str, float]:
        """{piece id: probability it is still on the board} for pieces at risk."""
        out = {}
        for pid in self.piece_ids():
            dead = self.piece_distribution(pid).get(None, 0.0)
            if dead > 1e-9:
                out[pid] = 1 - dead
        return out

    # ==================== DISPLAY ====================

    def get_expected_board(self) -> chess.Board:
        """The most probable branch's board."""
        best = max(self.branches.values(), key=lambda b: b.get_probability())
        return best.board.copy()

    def display(self):
        print(self.get_expected_board())
        print(f"\nBranches: {self.get_branch_count()}")
        print(f"Turn: {'White' if self.get_current_turn() else 'Black'}")
        if self.is_quantum_check():
            print(f"⚠️  QUANTUM CHECK ({self.get_check_probability():.0%})")

    def get_fen(self) -> str:
        return self.get_expected_board().fen()
