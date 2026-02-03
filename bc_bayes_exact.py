from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from collections import Counter
import math
from typing import Iterable

import seed_seeker  # compiled cython module

RARITY_ORDER = ["Rare", "Super", "Uber", "Legendary"]

# -----------------------------
# Banner parsing (paste.txt)
# -----------------------------

@dataclass(frozen=True)
class Banner:
    rate_cumsum: List[int]
    units_by_rarity: List[List[str]]
    rerollable_rarities: List[int]

def _parse_units_with_indices(units_str: str) -> List[str]:
    units_str = units_str.strip()
    if not units_str:
        return []
    pat = re.compile(r"(\d+)\s+([^,]+)(?:,|$)")
    found = pat.findall(units_str)
    idx_to_name: Dict[int, str] = {int(i): name.strip() for i, name in found}
    return [idx_to_name[i] for i in sorted(idx_to_name)]

def parse_banner_paste(paste: str) -> Banner:
    lines = [ln.strip() for ln in paste.splitlines() if ln.strip()]
    relevant: Dict[str, Dict[str, Any]] = {}

    line_regex = re.compile(
        r"^(?P<rarity>Rare|Super|Uber|Legendary):\s+"
        r"(?P<rate>[\d.]+)%\s+\(\d+\s+cats\)\s*(?P<units>.*)$"
    )

    for ln in lines:
        m = line_regex.match(ln)
        if not m:
            continue
        rarity = m.group("rarity")
        rate_pct = float(m.group("rate"))
        slots = int(round(rate_pct * 100))
        units = _parse_units_with_indices(m.group("units"))
        relevant[rarity] = {"slots": slots, "units": units}

    slots_list, pools = [], []
    for r in RARITY_ORDER:
        if r not in relevant:
            slots_list.append(0)
            pools.append([])
        else:
            slots_list.append(relevant[r]["slots"])
            pools.append(relevant[r]["units"])

    cumsum = []
    s = 0
    for v in slots_list:
        s += v
        cumsum.append(s)
    cumsum[-1] = 10000
    for i in range(len(cumsum) - 2, -1, -1):
        cumsum[i] = min(cumsum[i], cumsum[i + 1])

    return Banner(rate_cumsum=cumsum, units_by_rarity=pools, rerollable_rarities=[0])

def load_paste(path: str = "paste.txt") -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {path}")
    return p.read_text(encoding="utf-8", errors="replace")


# -----------------------------
# Deterministic simulator (Python) for prediction mixture once posterior small
# -----------------------------

def advance_seed(seed: int) -> int:
    seed &= 0xFFFFFFFF
    seed ^= ((seed << 13) & 0xFFFFFFFF)
    seed ^= (seed >> 17)
    seed ^= ((seed << 15) & 0xFFFFFFFF)
    return seed & 0xFFFFFFFF

def get_rarity(seed: int, rate_cumsum: Sequence[int]) -> int:
    x = seed % 10000
    for i, thr in enumerate(rate_cumsum):
        if x < thr:
            return i
    return len(rate_cumsum) - 1

def get_unit_index(seed: int, n: int, removed_index: int = -1) -> int:
    if n <= 0:
        return -1
    if removed_index < 0:
        return seed % n
    if n <= 1:
        return -1
    idx = seed % (n - 1)
    if idx >= removed_index:
        idx += 1
    return idx

def simulate_next_k_from_seed_before(
    seed_before: int,
    banner_units_int: List[List[int]],
    rate_cumsum: List[int],
    rerollable: List[int],
    observed_len: int,
    k: int,
) -> List[int]:
    seed = seed_before & 0xFFFFFFFF
    last = None
    out: List[int] = []
    reroll_set = set(rerollable)

    for _ in range(observed_len + k):
        seed = advance_seed(seed)
        rarity = get_rarity(seed, rate_cumsum)

        seed = advance_seed(seed)
        pool = banner_units_int[rarity]
        n = len(pool)
        idx = get_unit_index(seed, n, -1)
        unit_id = pool[idx] if idx >= 0 else -1

        if last is not None and rarity in reroll_set and unit_id == last:
            seed = advance_seed(seed)
            idx2 = get_unit_index(seed, n, idx)
            unit_id = pool[idx2] if idx2 >= 0 else -1

        out.append(unit_id)
        last = unit_id

    return out[observed_len:]


# -----------------------------
# Posterior management: over seed_after values (chaining)
# -----------------------------

Posterior = Dict[int, float]  # maps seed_before_next_roll -> probability mass

def posterior_uniform(states: List[int]) -> Posterior:
    if not states:
        return {}
    w = 1.0 / len(states)
    return {s: w for s in states}

def posterior_from_matches_seed_after(matches: List[Tuple[int, int]], prior_over_seed_before: Posterior | None) -> Posterior:
    """
    matches: list of (seed_before, seed_after)
    prior_over_seed_before: posterior from previous iteration over seed_before values.
      If None: treat seed_before prior as uniform over the found matches.

    Returns posterior over seed_after values.
    """
    if not matches:
        return {}

    # If no prior, uniform over matches
    if prior_over_seed_before is None:
        counts = Counter(seed_after for _, seed_after in matches)
        total = sum(counts.values())
        return {s_after: c / total for s_after, c in counts.items()}

    # With prior: mass on a match = prior(seed_before); then aggregate by seed_after
    acc = Counter()
    total_mass = 0.0
    for seed_before, seed_after in matches:
        w = prior_over_seed_before.get(seed_before, 0.0)
        if w > 0:
            acc[seed_after] += w
            total_mass += w

    if total_mass <= 0:
        # fallback: uniform over seed_after in matches
        return posterior_uniform(list({s_after for _, s_after in matches}))

    return {s_after: float(mass / total_mass) for s_after, mass in acc.items()}

def save_posterior(path: str, posterior: Posterior) -> None:
    with open(path, "wb") as f:
        pickle.dump(posterior, f)

def load_posterior(path: str) -> Posterior:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Pickle did not contain a dict posterior")
    # ensure int->float
    out: Posterior = {}
    for k, v in obj.items():
        out[int(k) & 0xFFFFFFFF] = float(v)
    return out


# -----------------------------
# Prediction mixture from posterior
# -----------------------------

def predictive_mixture(
    posterior_seed_before: Posterior,
    id_to_name: Dict[int, str],
    banner_units_int: List[List[int]],
    rate_cumsum: List[int],
    rerollable: List[int],
    observed_len: int,
    k: int,
) -> List[Dict[str, float]]:
    per_pos = [Counter() for _ in range(k)]
    for seed_before, w in posterior_seed_before.items():
        future = simulate_next_k_from_seed_before(seed_before, banner_units_int, rate_cumsum, rerollable, observed_len, k)
        for i, uid in enumerate(future):
            per_pos[i][uid] += w

    out: List[Dict[str, float]] = []
    for c in per_pos:
        total = sum(c.values())
        probs: Dict[str, float] = {}
        for uid, mass in c.items():
            probs[id_to_name.get(uid, f"<id:{uid}>")] = float(mass / total) if total > 0 else 0.0
        out.append(probs)
    return out

def topk(probs: Dict[str, float], k: int = 6) -> List[Tuple[str, float]]:
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:k]


# -----------------------------
# Shannon entropy
# -----------------------------

def entropy_states_bits(posterior: dict[int, float]) -> float:
    """Shannon entropy of the posterior over states (bits)."""
    h = 0.0
    for p in posterior.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return h

def entropy_dist_bits(dist: dict[str, float]) -> float:
    """Shannon entropy of a categorical distribution (bits)."""
    h = 0.0
    for p in dist.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return h

def shannon_entropy_bits(probs: Dict[str, float]) -> float:
    """Shannon entropy in bits."""
    h = 0.0
    for p in probs.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return h

def format_prob(p: float) -> str:
    return f"{p:.4%}"  # 4 decimal % looks nice for small probs

def uber_breakdown(probs: Dict[str, float], uber_names: set[str], top_n: int) -> tuple[float, List[tuple[str, float]]]:
    """Return (P(any_uber), top_ubers_list) for this roll."""
    ubers = [(name, probs.get(name, 0.0)) for name in uber_names]
    ubers = [(n, p) for n, p in ubers if p > 0.0]
    total = sum(p for _, p in ubers)
    ubers.sort(key=lambda x: x[1], reverse=True)
    return total, ubers[:top_n]



# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paste", default="paste.txt")
    ap.add_argument("--posterior-pkl", default="posterior.pkl")
    ap.add_argument("--use-pickle", action="store_true", help="Load prior posterior from pickle (if present).")
    ap.add_argument("--save-pickle", action="store_true", help="Save updated posterior to pickle after this update.")
    ap.add_argument("--start", type=int, default=0, help="Start seed (inclusive) for seeking (only used if no prior pickle).")
    ap.add_argument("--end", type=int, default=1_000_000, help="End seed (exclusive) for seeking (only used if no prior pickle).")
    ap.add_argument("--max-found", type=int, default=0, help="Stop after this many matches (0 = all in range).")
    ap.add_argument("--predict-k", type=int, default=30,
                help="Number of future rolls to predict")

    ap.add_argument("--top-k", type=int, default=6,
                help="Number of top outcomes to display per roll")

    ap.add_argument("--print-pos", type=int, default=10,
                help="How many future roll positions to print")
    ap.add_argument("--show-cat",action="append",default=[],
        help='Print probability of a specific cat at each roll. Repeatable. '
         'Special value "Ubers" prints P(any Uber) + which Ubers.')

    ap.add_argument("--entropy", action="store_true", help="Print Shannon entropy (bits) per roll to visualize uncertainty collapse.")

    ap.add_argument("--uber-top",type=int,default=5, help="When using --show-cat Ubers, show this many top Ubers per roll.")
    ap.add_argument("--observed-file",default="rolls.txt", help="Path to a text file containing observed rolls, comma-separated. " 'Example: "Gold Cat, Pirate Cat"')
    ap.add_argument("--estimate-rolls",action="store_true",help="Estimate how many additional rolls are needed to identify the exact seed, "
         "using posterior entropy H(S) and next-roll entropy H(Y).")


    args = ap.parse_args()

    banner = parse_banner_paste(load_paste(args.paste))

    def load_observed_rolls(path: str) -> List[str]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Observed rolls file not found: {path}. "
                "Expected comma-separated cat names."
            )

        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return []

        # Split on commas, strip whitespace
        rolls = [item.strip() for item in text.split(",") if item.strip()]
        return rolls


    #
    #observed_rolls = ["Viking Cat", "Tin Cat", "Fortune Teller Cat", "Archer Cat", "Bishop Cat"]
    observed_rolls = load_observed_rolls(args.observed_file)
    print(f"Loaded observed rolls ({len(observed_rolls)}): {observed_rolls}")

    # Map names to intsor load from elsewhere (keep it simple: hardcode for now
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    # Uber rarity index is 2 in [Rare, Super, Uber, Legendary]
    uber_names = set(banner.units_by_rarity[2])
    

    def get_id(name: str) -> int:
        if name not in name_to_id:
            new_id = len(name_to_id) + 1
            name_to_id[name] = new_id
            id_to_name[new_id] = name
        return name_to_id[name]

    banner_units_int: List[List[int]] = []
    for pool in banner.units_by_rarity:
        banner_units_int.append([get_id(u) for u in pool])

    observed_int = [get_id(u) for u in observed_rolls]


    # Sanity check observed_rolls for typos and other cats not in campaign
    unknown = [r for r in observed_rolls if r not in name_to_id]
    if unknown:
        raise SystemExit(f"Unknown cat names in observed rolls: {unknown}")

    # Load prior posterior if requested and exists
    prior: Posterior | None = None
    if args.use_pickle and Path(args.posterior_pkl).exists():
        prior = load_posterior(args.posterior_pkl)
        print(f"Loaded prior posterior from {args.posterior_pkl} with {len(prior)} states.")
    else:
        print("No prior posterior loaded (starting fresh).")

    # Exact match seeking:
    # - If prior exists, we seek from each prior seed_before state directly by running seeker over tiny ranges?
    #   Not necessary: you already have specific seed_before candidates, so you can just simulate forward.
    #   BUT: since you requested Cython seeker usage, we’ll use it in two modes:
    #
    # Mode A: no prior -> brute seek in [start,end)
    # Mode B: prior exists -> for each prior seed_before, check exact match by seeking range [seed_before, seed_before+1)
    #
    matches: List[Tuple[int, int]] = []

    if prior is None:
        # Full range search
        matches = seed_seeker.seek_seeds_before_after(
            args.start, args.end,
            banner.rate_cumsum,
            banner_units_int,
            banner.rerollable_rarities,
            observed_int,
            args.max_found
        )
        print(f"Matches found in range [{args.start},{args.end}): {len(matches)}")
    else:
        # Check each prior state as the candidate "seed_before" for this iteration
        # by searching the 1-seed range [s, s+1)
        for s_before, w in sorted(prior.items(), key=lambda kv: -kv[1]):
            m = seed_seeker.seek_seeds_before_after(
                s_before, s_before + 1,
                banner.rate_cumsum,
                banner_units_int,
                banner.rerollable_rarities,
                observed_int,
                1
            )
            if m:
                matches.extend(m)
        print(f"Matches found from prior support: {len(matches)} / {len(prior)}")

    if not matches:
        raise SystemExit("No exact matching seeds found (check banner paste, roll names, version/reroll rules, or search range).")

    # Build posterior over seed_after (this is the new "seed_before_next_roll" support)
    posterior_after = posterior_from_matches_seed_after(matches, prior_over_seed_before=prior)

    # Save if requested
    if args.save_pickle:
        save_posterior(args.posterior_pkl, posterior_after)
        print(f"Saved posterior with {len(posterior_after)} states to {args.posterior_pkl}")

    # Predict next K from the posterior seed_after states
    # IMPORTANT: We are now predicting starting from seed_after values (seed BEFORE next roll),
    # so observed_len=0 for this prediction horizon.
    pred = predictive_mixture(
        posterior_after,
        id_to_name,
        banner_units_int,
        banner.rate_cumsum,
        banner.rerollable_rarities,
        observed_len=0,
        k=args.predict_k,
    )

    #Roll+01 distribution available so compute estimated rolls left
    if args.estimate_rolls:
        Hs = entropy_states_bits(posterior_after)
        Neff = 2 ** Hs  # effective number of states

        Hy = entropy_dist_bits(pred[0]) if pred else 0.0  # Roll+01 entropy
        if Hy > 1e-12:
            est = math.ceil(Hs / Hy)
        else:
            est = float("inf")

        print(f"Estimate: H(S)={Hs:.3f} bits  (≈ {Neff:.2f} effective states)")
        print(f"Estimate: H(Y_next)={Hy:.3f} bits (next-roll uncertainty)")
        if est != float('inf'):
            print(f"Estimated additional rolls to unique seed ≈ {est}")
        else:
            print("Estimated additional rolls to unique seed ≈ inf (next roll already deterministic)")



    def parse_show_cat_args(raw_args):
        """
        Takes repeated --show-cat args, each possibly comma-separated,
        returns a de-duplicated list preserving order.
        """
        seen = set()
        out = []

        for arg in raw_args or []:
            # Split on commas
            for item in arg.split(","):
                name = item.strip()
                if not name:
                    continue
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(name)
        return out

    show_items = parse_show_cat_args(args.show_cat)
    show_items_norm = [s.lower() for s in show_items]


    for i in range(min(args.print_pos, len(pred))):
        roll_probs = pred[i]

        # Main top-k line
        best = topk(roll_probs, args.top_k)
        s = ", ".join([f"{u} ({p:.2%})" for u, p in best])
        line = f"Roll+{i+1:02d}: {s}"

        # Optional entropy
        if args.entropy:
            h = shannon_entropy_bits(roll_probs)
            line += f" | H={h:.3f} bits"

        print(line)

        # Optional show-cat lines
        if show_items:
            for raw, norm in zip(show_items, show_items_norm):
                if norm == "ubers" or norm == "uber":
                    total, top_u = uber_breakdown(roll_probs, uber_names, args.uber_top)
                    if top_u:
                        detail = ", ".join([f"{n} ({format_prob(p)})" for n, p in top_u])
                    else:
                        detail = "(none)"
                    print(f"  P(any Uber)={format_prob(total)} | Top Ubers: {detail}")
                else:
                    p = roll_probs.get(raw, 0.0)
                    print(f"  P({raw})={format_prob(p)}")


if __name__ == "__main__":
    main()

