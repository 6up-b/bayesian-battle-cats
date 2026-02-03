# distutils: language=c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
from libc.stdint cimport uint32_t
from libcpp.vector cimport vector

cdef inline uint32_t advance_seed(uint32_t seed) nogil:
    seed ^= (seed << 13)
    seed ^= (seed >> 17)
    seed ^= (seed << 15)
    return seed

cdef inline int get_rarity(uint32_t seed, int t0, int t1, int t2, int t3) nogil:
    cdef int x = <int>(seed % 10000)
    if x < t0: return 0
    if x < t1: return 1
    if x < t2: return 2
    return 3

cdef inline int get_unit_index(uint32_t seed, int n, int removed_index) nogil:
    cdef int idx
    if n <= 0:
        return -1
    if removed_index < 0:
        return <int>(seed % n)
    if n <= 1:
        return -1
    idx = <int>(seed % (n - 1))
    if idx >= removed_index:
        idx += 1
    return idx

cpdef list seek_seeds_before_after(
    unsigned long long start_seed,
    unsigned long long end_seed,
    list rate_cumsum,                 # length 4
    list units_by_rarity_int,         # [list[int], list[int], list[int], list[int]]
    list rerollable_rarities,         # e.g. [0]
    list observed_int,                # list[int]
    int max_found=0                   # 0 => all in range; else stop after N matches
):
    """
    Returns list of (seed_before, seed_after) for seeds in [start_seed, end_seed)
    that EXACTLY reproduce observed_int.

    seed_after is the RNG state after consuming the observed rolls (including any reroll advances),
    i.e., the state you would use as "seed before the next roll" when chaining sessions.
    """
    cdef int t0 = <int>rate_cumsum[0]
    cdef int t1 = <int>rate_cumsum[1]
    cdef int t2 = <int>rate_cumsum[2]
    cdef int t3 = <int>rate_cumsum[3]

    cdef vector[int] pool0
    cdef vector[int] pool1
    cdef vector[int] pool2
    cdef vector[int] pool3

    cdef int v
    for v in units_by_rarity_int[0]:
        pool0.push_back(<int>v)
    for v in units_by_rarity_int[1]:
        pool1.push_back(<int>v)
    for v in units_by_rarity_int[2]:
        pool2.push_back(<int>v)
    for v in units_by_rarity_int[3]:
        pool3.push_back(<int>v)

    cdef int reroll0 = 0
    cdef int reroll1 = 0
    cdef int reroll2 = 0
    cdef int reroll3 = 0
    for v in rerollable_rarities:
        if v == 0: reroll0 = 1
        elif v == 1: reroll1 = 1
        elif v == 2: reroll2 = 1
        elif v == 3: reroll3 = 1

    cdef vector[int] obs
    for v in observed_int:
        obs.push_back(<int>v)
    cdef int m = <int>obs.size()

    cdef list found = []
    cdef uint32_t s
    cdef uint32_t seed
    cdef uint32_t seed_after
    cdef int i, rarity, n, idx, unit_id, last_id
    cdef bint has_last
    cdef int rerollable
    cdef bint matched

    cdef uint32_t start32 = <uint32_t>(start_seed & 0xFFFFFFFF)
    cdef uint32_t end32 = <uint32_t>(end_seed & 0xFFFFFFFF)

    with nogil:
        s = start32
        while s < end32:
            seed = s
            has_last = False
            last_id = -1
            matched = True

            for i in range(m):
                seed = advance_seed(seed)
                rarity = get_rarity(seed, t0, t1, t2, t3)

                seed = advance_seed(seed)

                if rarity == 0:
                    n = <int>pool0.size()
                    idx = get_unit_index(seed, n, -1)
                    unit_id = pool0[idx] if idx >= 0 else -1
                    rerollable = reroll0
                elif rarity == 1:
                    n = <int>pool1.size()
                    idx = get_unit_index(seed, n, -1)
                    unit_id = pool1[idx] if idx >= 0 else -1
                    rerollable = reroll1
                elif rarity == 2:
                    n = <int>pool2.size()
                    idx = get_unit_index(seed, n, -1)
                    unit_id = pool2[idx] if idx >= 0 else -1
                    rerollable = reroll2
                else:
                    n = <int>pool3.size()
                    idx = get_unit_index(seed, n, -1)
                    unit_id = pool3[idx] if idx >= 0 else -1
                    rerollable = reroll3

                if has_last and rerollable and unit_id == last_id:
                    seed = advance_seed(seed)
                    if rarity == 0:
                        idx = get_unit_index(seed, <int>pool0.size(), idx)
                        unit_id = pool0[idx] if idx >= 0 else -1
                    elif rarity == 1:
                        idx = get_unit_index(seed, <int>pool1.size(), idx)
                        unit_id = pool1[idx] if idx >= 0 else -1
                    elif rarity == 2:
                        idx = get_unit_index(seed, <int>pool2.size(), idx)
                        unit_id = pool2[idx] if idx >= 0 else -1
                    else:
                        idx = get_unit_index(seed, <int>pool3.size(), idx)
                        unit_id = pool3[idx] if idx >= 0 else -1

                if unit_id != obs[i]:
                    matched = False
                    break

                last_id = unit_id
                has_last = True

            if matched:
                seed_after = seed
                with gil:
                    found.append((<unsigned long long>s, <unsigned long long>seed_after))
                    if max_found > 0 and len(found) >= max_found:
                        return found

            s += 1

    return found

