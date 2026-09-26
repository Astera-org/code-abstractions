"""In-house coding benchmark: 20 int->int functions x 10 languages = 200 generations
per model, executed and scored by the same harness that measures factorization --
one apples-to-apples capability axis for the scatter (same recipes, reasoning OFF,
greedy, execution-scored). Difficulty comes from CORRECTNESS, not speed: the runner
call templates and 120s timeouts make efficiency gates language-unfair.

Tiers: 6 easy floor anchors, 8 medium, 6 hard (modular combinatorics, digit-DP,
base-factorisation traps, two invented specs immune to memorisation). All battery
inputs < 2^31 and all outputs < 2^31 (32-bit-language harness templates).
pass = all six battery outputs exact, per (function, language) cell.
"""
MOD = 1_000_000_007


def _digits(n):
    return [int(c) for c in str(n)]


# ---- easy ----
def r_digit_sum(n): return sum(_digits(n))
def r_count_divisors(n): return sum(1 for d in range(1, int(n**0.5) + 1) if n % d == 0
                                    for _ in ([1] if d * d == n else [1, 1]))
def r_int_sqrt(n):
    x = int(n**0.5)
    while (x + 1) * (x + 1) <= n: x += 1
    while x * x > n: x -= 1
    return x
def r_largest_prime_factor(n):
    f, d = 1, 2
    while d * d <= n:
        while n % d == 0: f, n = d, n // d
        d += 1
    return max(f, n) if n > 1 else f
def r_binary_gap(n):
    b = bin(n)[2:]; best = cur = 0; seen = False
    for c in b:
        if c == '1':
            if seen: best = max(best, cur)
            seen, cur = True, 0
        else: cur += 1
    return best
def r_alt_digit_sum(n):
    s, sign = 0, 1
    for d in _digits(n): s += sign * d; sign = -sign
    return s

# ---- medium ----
def r_collatz_steps(n):
    s = 0
    while n != 1: n = n // 2 if n % 2 == 0 else 3 * n + 1; s += 1
    return s
def r_euler_phi(n):
    res, p, m = n, 2, n
    while p * p <= m:
        if m % p == 0:
            while m % p == 0: m //= p
            res -= res // p
        p += 1
    if m > 1: res -= res // m
    return res
def r_josephus3(n):
    pos = 0
    for i in range(2, n + 1): pos = (pos + 3) % i
    return pos + 1
def r_nth_prime(n):
    c, x = 0, 1
    while c < n:
        x += 1
        if all(x % d for d in range(2, int(x**0.5) + 1)): c += 1
    return x
def r_happy_steps(n):
    s = 0
    while n != 1 and n != 4:
        n = sum(d * d for d in _digits(n)); s += 1
    return s
def r_partitions(n):
    P = [1] + [0] * n
    for k in range(1, n + 1):
        for i in range(k, n + 1): P[i] = (P[i] + P[i - k]) % MOD
    return P[n]
def r_binary_multiple_digits(n):
    # digits of the smallest positive multiple of n whose decimal digits are all 0/1
    from collections import deque
    seen = {1 % n}; q = deque([(1 % n, 1)])
    if 1 % n == 0: return 1
    while q:
        r, L = q.popleft()
        for d in (0, 1):
            nr = (r * 10 + d) % n
            if nr == 0: return L + 1
            if nr not in seen: seen.add(nr); q.append((nr, L + 1))
def r_gray_weight(n): return bin(n ^ (n >> 1)).count("1")

# ---- hard ----
def r_derangements(n):
    a, b = 1, 0          # D(0), D(1)
    if n == 0: return 1
    for i in range(2, n + 1): a, b = b, (i - 1) * (a + b) % MOD
    return b if n >= 1 else a
def r_distinct_digit_count(n):
    def ok(x): s = str(x); return len(set(s)) == len(s)
    if n < 100000: return sum(1 for x in range(1, n + 1) if ok(x))
    # combinatorial for large n
    s = str(n); L = len(s); total = 0
    for length in range(1, L):
        t = 9
        for i in range(1, length): t *= 10 - i
        total += t
    used = set()
    for i, ch in enumerate(s):
        d = int(ch)
        for x in range(0 if i else 1, d):
            if x in used: continue
            t = 1
            for j in range(L - i - 1): t *= 9 - i - j
            total += t
        if d in used: return total
        used.add(d)
    return total + 1
def r_fact_zeros_base12(n):
    v2 = v3 = 0; p = 2
    while p <= n: v2 += n // p; p *= 2
    p = 3
    while p <= n: v3 += n // p; p *= 3
    return min(v2 // 2, v3)
def r_squarefree_count(n):
    import math
    lim = int(math.isqrt(n)); mu = [1] * (lim + 1); is_c = [False] * (lim + 1)
    primes = []
    for i in range(2, lim + 1):
        if not is_c[i]: primes.append(i); mu[i] = -1
        for p in primes:
            if i * p > lim: break
            is_c[i * p] = True
            if i % p == 0: mu[i * p] = 0; break
            mu[i * p] = -mu[i]
    return sum(mu[d] * (n // (d * d)) for d in range(1, lim + 1))
def r_verlix(n):
    s = 0
    while n >= 100:
        n = sum(d * (1 + i) for i, d in enumerate(reversed(_digits(n)))); s += 1
    return s * 1000 + n
def r_dromber(n):
    s = 0
    for d in _digits(n):
        s = s * 2 + d if d % 2 == 0 else s + d * d
    return s % 100000


def r_sq_digit_sum(n): return sum(d * d for d in _digits(n))
def r_even_digits(n): return sum(1 for d in _digits(n) if d % 2 == 0)
def r_digit_prod(n):
    import math
    return math.prod(_digits(n))
def r_largest_proper_divisor(n):
    return max(d for d in range(1, n) if n % d == 0) if n > 1 else 1


# the factor-experiment functions are a SUBSET of the benchmark (authors), so the
# capability axis covers exactly the functions the factorization axis is built on
BENCH = [
 ("sq_digit_sum", "returns the sum of the squares of the digits of a non-negative "
  "integer", r_sq_digit_sum, [7, 1234, 5, 90, 406, 2468]),
 ("even_digits", "returns how many of the digits of a non-negative integer are even",
  r_even_digits, [7, 1234, 5, 90, 406, 2468]),
 ("digit_prod", "returns the product of the digits of a non-negative integer",
  r_digit_prod, [7, 1234, 5, 90, 406, 2468]),
 ("largest_proper_divisor", "returns the largest proper divisor of a positive "
  "integer", r_largest_proper_divisor, [7, 1234, 5, 90, 406, 2468]),

 ("digit_sum", "returns the sum of the digits of a non-negative integer",
  r_digit_sum, [7, 1234, 5, 90, 406, 2468]),
 ("count_divisors", "returns the number of positive divisors of a positive integer",
  r_count_divisors, [7, 36, 100, 97, 5040, 1000000]),
 ("int_sqrt", "returns the floor of the square root of a non-negative integer, "
  "computed exactly (beware floating point at large inputs)",
  r_int_sqrt, [0, 15, 16, 2147395599, 999950884, 2147483647]),
 ("largest_prime_factor", "returns the largest prime factor of an integer greater "
  "than 1", r_largest_prime_factor, [2, 13195, 600851475, 97, 1024, 999999937]),
 ("binary_gap", "returns the length of the longest run of zeros strictly between "
  "two ones in the binary representation of a positive integer",
  r_binary_gap, [9, 529, 20, 15, 1041, 32]),
 ("alt_digit_sum", "returns the alternating sum of the digits of a positive integer, "
  "starting with a plus sign on the most significant digit",
  r_alt_digit_sum, [7, 1234, 90, 406, 2468, 918273]),
 ("collatz_steps", "returns the number of Collatz steps (n -> n/2 if even, else "
  "3n+1) needed to reach 1 from a positive integer",
  r_collatz_steps, [1, 7, 27, 97, 871, 6171]),
 ("euler_phi", "returns Euler's totient of a positive integer",
  r_euler_phi, [1, 9, 10, 97, 360, 999999]),
 ("josephus3", "returns the 1-indexed survivor position of the Josephus problem "
  "with n people in a circle, eliminating every 3rd person",
  r_josephus3, [1, 5, 7, 14, 41, 100]),
 ("nth_prime", "returns the n-th prime number, where the 1st prime is 2",
  r_nth_prime, [1, 6, 25, 100, 500, 1000]),
 ("happy_steps", "returns how many times the sum-of-squared-digits map must be "
  "applied to a positive integer before reaching 1 or 4",
  r_happy_steps, [1, 7, 19, 4, 89, 12345]),
 ("partitions", "returns the number of integer partitions of n, modulo 1000000007",
  r_partitions, [1, 5, 10, 50, 100, 1000]),
 ("binary_multiple_digits", "returns the number of decimal digits of the smallest "
  "positive multiple of n whose decimal representation contains only the digits 0 "
  "and 1", r_binary_multiple_digits, [1, 2, 3, 7, 9, 99]),
 ("gray_weight", "returns the number of ones in the binary reflected Gray code of "
  "a non-negative integer", r_gray_weight, [0, 1, 7, 10, 1023, 1000000]),
 ("derangements", "returns the number of derangements of n elements, modulo "
  "1000000007", r_derangements, [1, 2, 4, 10, 100, 100000]),
 ("distinct_digit_count", "returns how many integers from 1 to n have no repeated "
  "decimal digit", r_distinct_digit_count, [10, 100, 1000, 5432, 98765, 1000000000]),
 ("fact_zeros_base12", "returns the number of trailing zeros of n factorial when "
  "written in base 12", r_fact_zeros_base12, [5, 12, 24, 100, 1000, 1000000]),
 ("squarefree_count", "returns how many integers from 1 to n are squarefree (not "
  "divisible by any perfect square greater than 1)",
  r_squarefree_count, [1, 10, 100, 1000, 123456, 1000000]),
 ("verlix", "computes the verlix value of n: repeatedly, while n is at least 100, "
  "replace n by the sum over its decimal digits of digit*(1+i) where i is the "
  "digit's position from the right starting at 0; when n drops below 100, return "
  "steps*1000+n where steps counts the replacements",
  r_verlix, [7, 99, 100, 1234, 98765, 1000000]),
 ("dromber", "computes the dromber value of n: starting with s=0, scan the decimal "
  "digits of n from left to right; for each even digit d set s to 2*s+d, for each "
  "odd digit d set s to s+d*d; return s modulo 100000",
  r_dromber, [7, 24, 135, 2468, 13579, 9182736]),
]


# ---- expert tier (added after the calibration pilot: gemma 0.875 was only
# challenged by multi-phase construction tasks; these live in that band) ----
def r_mountain_count(n):
    def is_mountain(x):
        d = [int(c) for c in str(x)]
        if len(d) < 3: return False
        i = 1
        while i < len(d) and d[i] > d[i - 1]: i += 1
        if i == 1 or i == len(d): return False
        while i < len(d) and d[i] < d[i - 1]: i += 1
        return i == len(d)
    return sum(1 for x in range(1, n + 1) if is_mountain(x))
def r_fib_reps(n):
    fibs = [1, 2]
    while fibs[-1] <= n: fibs.append(fibs[-1] + fibs[-2])
    fibs = [f for f in fibs if f <= n][::-1]
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def go(i, rem):
        if rem == 0: return 1
        if i >= len(fibs) or rem < 0: return 0
        if sum(fibs[i:]) < rem: return 0
        return go(i + 1, rem - fibs[i]) + go(i + 1, rem)
    return go(0, n)
def r_largest_palindrome_leq(n):
    x = n
    while str(x) != str(x)[::-1]: x -= 1
    return x
def r_oddfact_zeros_base15(n):
    # trailing zeros of (2n-1)!! = 1*3*5*...*(2n-1) in base 15 = 3*5
    def v_in_oddfact(p):
        # exponent of odd prime p in product of first n odd numbers
        tot, pk = 0, p
        while pk <= 2 * n - 1:
            # count odd multiples of pk in [1, 2n-1]
            m = (2 * n - 1) // pk
            tot += (m + 1) // 2
            pk *= p
        return tot
    return min(v_in_oddfact(3), v_in_oddfact(5))
def r_trelk(n):
    s, mode = 0, 0
    for d in _digits(n):
        if d == 0: mode = 1 - mode; continue
        s = s + d * d if mode == 0 else s * 2 + d
    return s % 1_000_000
def r_josephus_sqrt(n):
    k = r_int_sqrt(n)
    if k < 1: k = 1
    pos = 0
    for i in range(2, n + 1): pos = (pos + k) % i
    return pos + 1
def r_prime_digitprime_next(n):
    def is_p(x):
        if x < 2: return False
        return all(x % d for d in range(2, int(x**0.5) + 1))
    x = 2 * n + 1
    while not (is_p(x) and is_p(sum(_digits(x)))): x += 1
    return x


BENCH += [
 ("mountain_count", "returns how many integers from 1 to n have decimal digits that "
  "strictly increase and then strictly decrease, with both parts non-empty",
  r_mountain_count, [100, 500, 1000, 12321, 99999, 400000]),
 ("fib_reps", "returns the number of ways to write n as a sum of distinct Fibonacci "
  "numbers from the sequence 1, 2, 3, 5, 8, ... (each used at most once)",
  r_fib_reps, [1, 10, 100, 1000, 10000, 121393]),
 ("largest_palindrome_leq", "returns the largest integer not exceeding n whose "
  "decimal representation is a palindrome",
  r_largest_palindrome_leq, [7, 100, 1000, 12320, 99900, 1000000]),
 ("oddfact_zeros_base15", "returns the number of trailing zeros, in base 15, of the "
  "product 1*3*5*...*(2n-1) of the first n odd numbers",
  r_oddfact_zeros_base15, [3, 10, 50, 100, 1000, 100000]),
 ("trelk", "computes the trelk value of n: starting with s=0 in mode A, scan the "
  "decimal digits left to right; a zero digit switches the mode (A to B or B to A) "
  "and is otherwise ignored; in mode A a digit d sets s to s+d*d, in mode B it sets "
  "s to 2*s+d; return s modulo 1000000",
  r_trelk, [7, 105, 2048, 90210, 1030507, 86420975]),
 ("josephus_sqrt", "returns the 1-indexed survivor of the Josephus problem with n "
  "people where every k-th person is eliminated and k is the floor of the square "
  "root of n", r_josephus_sqrt, [2, 10, 50, 100, 1000, 10000]),
 ("prime_digitprime_next", "returns the smallest prime strictly greater than 2n "
  "whose digit sum is also prime",
  r_prime_digitprime_next, [1, 10, 100, 1234, 50000, 1000000]),
]
