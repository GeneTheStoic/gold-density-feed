"""
gold_density_feed.py - publish the option-implied probability distribution of GLD
as a small JSON feed, with the diagnostics needed to judge how far to trust it.

WHAT THIS IS
    An option chain does not only imply a volatility. Taken together, the prices
    across all strikes imply a full probability distribution for where the underlying
    can finish at expiry. Breeden and Litzenberger showed in 1978 that the second
    derivative of the call price with respect to strike, discounted, is that density.

WHAT IT IS NOT
    The result is a RISK-NEUTRAL distribution of GLD. Two separate restrictions:
      1. Risk-neutral is not real-world. The left tail is systematically fatter than
         outcomes turn out to be, because people pay extra for crash protection.
      2. The underlying is GLD, an ETF, not spot gold and not the XAUUSD a broker
         quotes. Rescaling the percentiles onto a gold chart is a monitoring
         convenience, not an equivalence.

WHY THE FORWARD COMES FROM PUT-CALL PARITY
    GLD is not a non-dividend-paying asset in the pricing sense: it carries an expense
    ratio and its own financing economics. Assuming F = S * exp(rT) would misspecify
    the forward and bias every implied volatility solved from it. Instead the forward
    is read out of the market, from the calls and puts quoted at the same near-the-money
    strikes, and everything is priced in Black-76 on that forward. That also puts
    out-of-the-money puts and calls in one space, so mixing the two sides is consistent
    rather than stitched together afterwards.

WHY THE POLYNOMIAL DEGREE IS CHOSEN RATHER THAN FIXED
    A smile fit does not guarantee a convex call curve, and a curve that is not convex
    produces negative probability. Fixing the degree and clipping the negatives repairs
    the symptom and hides the cause. Here every degree is recovered in full and the
    first arbitrage-consistent one is used, so clipping has almost nothing left to do.
    Whatever it does is published.

WHAT IS REPORTED SO THE READER CAN JUDGE THE RESULT
    negative probability mass before clipping, monotonicity and convexity violations
    of the recovered call curve, the largest put-call implied volatility gap near the
    money, and a sensitivity study across polynomial degree, strike window and tail
    rule. Percentiles are far more stable than third and fourth moments, and the feed
    says so in its own output.

DEPENDENCIES
    yfinance, pandas, numpy.

USAGE
    pip install yfinance pandas numpy
    python gold_density_feed.py                  print the JSON
    python gold_density_feed.py --out feed.json  also write it to a file
    python gold_density_feed.py --debug          show the fit, checks and sensitivity
    python gold_density_feed.py --selftest       check the engine against algebra
"""

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# the gold ETF, whose options are the chain anyone can pull for free
TICKER = "GLD"

# discount rate only. The carry is not assumed: the forward is read from the market
RISK_FREE = 0.04

# liquidity filters
MIN_OI = 10
MIN_PRICE = 0.05
MAX_SPREAD = 0.25          # reject a quote whose bid-ask spread exceeds this of its mid

# the strike window that is fitted, as a fraction of the forward
MONEY_LO, MONEY_HI = 0.80, 1.20

# a solved volatility outside this range is a bad quote, not a market view
IV_FLOOR, IV_CEIL = 0.02, 2.00

# accept a listed expiry inside this window and take the one nearest 30 days
EXPIRY_LO, EXPIRY_HI = 20, 45

# the synthetic strike grid the density is differentiated on
GRID_LO, GRID_HI = 0.55, 1.65
GRID_STEPS = 2200

# the smile fit
FIT_DEGREE = 3
FIT_WIDTH = 0.35
MIN_POINTS = 8
MIN_SPAN = 0.04

# outside the quoted strikes the fit must not be extrapolated. "flat" holds the edge
# volatility, "linvar" extends total variance linearly at the slope of the edge
TAIL_RULE = "flat"

# a recovered call curve that is not convex is not arbitrage-consistent, and the
# density it produces contains negative probability. Rather than fixing a degree and
# repairing whatever comes out, the degree is chosen so the curve is convex and almost
# nothing needs repairing. This is the cheap form of a shape constraint
NEG_MASS_MAX = 1.0

# realized volatility is reported alongside, for reference only
RV_WINDOW = 30

# a contract whose last trade is older than this is still used when it has a live
# two-sided quote, but the median age is reported, because a live bounded spread is a
# stronger freshness test than a trade print
MAX_QUOTE_AGE_DAYS = 7

# the sensitivity study: every combination is recovered and the spread is published.
# FIT_WIDTH is included because a fit weighted hard toward the money can flatten the
# wings, which is exactly the region the recovery exists to measure
SENS_DEGREES = (2, 3, 4)
SENS_WINDOWS = ((0.75, 1.25), (0.80, 1.20), (0.85, 1.15))
SENS_TAILS = ("flat", "linvar")
SENS_WIDTHS = (0.20, 0.35, 1.00)

# the filter study: the liquidity thresholds are judgement, so their effect is measured
SENS_MIN_OI = (0, 10, 50)
SENS_MAX_SPREAD = (0.10, 0.25, 0.50)


class ChainUnusable(Exception):
    """Raised when the chain of the day cannot support a density."""


def finite(value):
    """
    Convert to float and return 0.0 for anything that is not a real number.

    pandas returns NaN for a missing quote, and NaN is truthy in Python, so the
    natural-looking float(row.get("bid") or 0.0) passes NaN through silently.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def norm_cdf(x):
    """Standard normal cumulative distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def b76_call(fwd, strike, t_years, vol, rate):
    """Black-76 price of a European call written on a forward."""
    if t_years <= 0.0 or vol <= 0.0:
        return math.exp(-rate * t_years) * max(fwd - strike, 0.0)
    v = vol * math.sqrt(t_years)
    d1 = (math.log(fwd / strike) + 0.5 * v * v) / v
    d2 = d1 - v
    return math.exp(-rate * t_years) * (fwd * norm_cdf(d1) - strike * norm_cdf(d2))


def b76_put(fwd, strike, t_years, vol, rate):
    """Black-76 price of a European put, by parity on the same forward."""
    call = b76_call(fwd, strike, t_years, vol, rate)
    return call - math.exp(-rate * t_years) * (fwd - strike)


def implied_vol(market_price, fwd, strike, t_years, rate, is_call):
    """
    Solve Black-76 backward for volatility, by bisection.

    An option price rises steadily as the assumed volatility rises, so a bracketed
    search cannot get lost. Returns None when the quote sits below intrinsic value or
    above what the ceiling volatility can produce, both of which mean a bad quote.
    """
    disc = math.exp(-rate * t_years)
    intrinsic = disc * (max(fwd - strike, 0.0) if is_call else max(strike - fwd, 0.0))
    if market_price <= intrinsic + 1e-6:
        return None
    pricer = b76_call if is_call else b76_put
    low, high = IV_FLOOR, IV_CEIL
    if pricer(fwd, strike, t_years, high, rate) < market_price:
        return None
    for _ in range(100):
        mid = 0.5 * (low + high)
        if pricer(fwd, strike, t_years, mid, rate) < market_price:
            low = mid
        else:
            high = mid
        if high - low < 1e-6:
            break
    return 0.5 * (low + high)


def quote(row):
    """
    Return the bid-ask midpoint and the relative spread, or None when unusable.

    A wide spread is the clearest sign that a printed mid is not a price anyone would
    trade at, so the spread travels with the quote and is filtered on directly.
    """
    bid = finite(row.get("bid"))
    ask = finite(row.get("ask"))
    #--- ask below bid is a crossed or locked quote, which is never tradable
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
        return None
    mid = 0.5 * (bid + ask)
    if mid <= 0.0:
        return None
    return mid, (ask - bid) / mid


def quote_age_days(row):
    """Days since the contract last traded, or None when the feed does not say."""
    stamp = row.get("lastTradeDate")
    try:
        ts = pd.Timestamp(stamp)
    except (TypeError, ValueError):
        return None
    if ts is None or pd.isna(ts):
        return None
    now = pd.Timestamp.now(tz=ts.tz) if ts.tzinfo else pd.Timestamp.now()
    return float((now - ts).total_seconds() / 86400.0)


def strike_map(frame):
    """Every strike in one side of the chain that carries a usable two-sided quote."""
    out = {}
    if frame is None or frame.empty:
        return out
    for _, row in frame.iterrows():
        k = finite(row.get("strike"))
        q = quote(row)
        if k > 0.0 and q is not None and q[0] >= MIN_PRICE and q[1] <= MAX_SPREAD:
            out[k] = q[0]
    return out


def forward_from_parity(chain, spot, t_years, rate, debug=False):
    """
    Read the forward out of the market instead of assuming a carry.

    Put-call parity says C - P = exp(-rT) * (F - K) at every strike, so each strike
    quoted on both sides gives an estimate of F. The median over the strikes nearest
    the money is taken, which is robust to one bad quote.

    This matters because GLD is not a non-dividend-paying asset in the pricing sense.
    It carries an expense ratio and its own financing economics, so F = S * exp(rT)
    would be wrong and would bias every volatility solved against it.
    """
    cmap, pmap = strike_map(chain.calls), strike_map(chain.puts)
    common = sorted(set(cmap) & set(pmap), key=lambda k: abs(k - spot))
    if len(common) < 3:
        raise ChainUnusable("fewer than three strikes quoted on both sides")

    disc = math.exp(-rate * t_years)
    estimates = [k + (cmap[k] - pmap[k]) / disc for k in common[:8]]
    fwd = float(np.median(estimates))
    if not math.isfinite(fwd) or fwd <= 0.0:
        raise ChainUnusable("put-call parity produced no usable forward")

    spread = (max(estimates) - min(estimates)) / fwd
    if debug:
        print("  forward %.4f from put-call parity over %d strikes, spread %.4f%%"
              % (fwd, len(estimates), 100.0 * spread))
        print("  net carry %.4f%% a year against spot %.2f, so against a %.1f%% discount"
              " rate the holding cost is about %.2f%%"
              % (100.0 * math.log(fwd / spot) / t_years, spot, 100.0 * rate,
                 100.0 * rate - 100.0 * math.log(fwd / spot) / t_years))
    return fwd, spread


def parity_iv_gap(chain, fwd, t_years, rate):
    """
    Largest disagreement between call-implied and put-implied volatility near the money.

    Mixing out-of-the-money puts on the left with out-of-the-money calls on the right
    only makes sense if the two sides agree where they overlap. This measures that.
    """
    cmap, pmap = strike_map(chain.calls), strike_map(chain.puts)
    gaps = []
    for k in sorted(set(cmap) & set(pmap), key=lambda k: abs(k - fwd))[:10]:
        cv = implied_vol(cmap[k], fwd, k, t_years, rate, True)
        pv = implied_vol(pmap[k], fwd, k, t_years, rate, False)
        if cv is not None and pv is not None:
            gaps.append(abs(cv - pv))
    if not gaps:
        return float("nan"), float("nan")
    return float(max(gaps)), float(np.median(gaps))


def collect_smile(chain, fwd, t_years, rate, money_lo, money_hi, debug=False,
                  min_oi=None, max_spread=None):
    """
    Solve implied volatility for the out-of-the-money contracts of one expiry.

    Out-of-the-money options carry the liquidity and almost all of the time value, so
    puts are read below the forward and calls above it. Because everything is priced
    in Black-76 on the same forward, the two sides land on one curve.
    """
    min_oi = MIN_OI if min_oi is None else min_oi
    max_spread = MAX_SPREAD if max_spread is None else max_spread
    points, ages = [], []
    for frame, is_call in ((chain.puts, False), (chain.calls, True)):
        if frame is None or frame.empty:
            continue
        usable = frame[frame["openInterest"].fillna(0) >= min_oi]
        if usable.empty:
            usable = frame
        for _, row in usable.iterrows():
            strike = finite(row.get("strike"))
            if strike <= 0.0:
                continue
            if is_call and strike < fwd:
                continue
            if not is_call and strike > fwd:
                continue
            moneyness = strike / fwd
            if moneyness < money_lo or moneyness > money_hi:
                continue
            q = quote(row)
            if q is None or q[0] < MIN_PRICE or q[1] > max_spread:
                continue
            vol = implied_vol(q[0], fwd, strike, t_years, rate, is_call)
            if vol is None or vol <= IV_FLOOR or vol >= IV_CEIL:
                continue
            #--- the relative spread travels with the point, so the fit can trust a
            #--- tight quote more than a wide one instead of treating them alike
            points.append((strike, vol, q[1]))
            age = quote_age_days(row)
            if age is not None:
                ages.append(age)
    points.sort()
    if debug and points:
        print("  solved %d out-of-the-money contracts, strikes %.0f to %.0f"
              % (len(points), points[0][0], points[-1][0]))
        if ages:
            print("  median days since last trade among them: %.1f" % float(np.median(ages)))
    return points


def fit_smile(points, fwd, degree=FIT_DEGREE, debug=False, width=FIT_WIDTH):
    """
    Fit a smooth volatility curve in forward log-moneyness, and remember its range.

    Differentiating raw quotes twice turns bid and ask noise into a meaningless
    density, so the quotes are smoothed first. A low-degree polynomial keeps the
    dependencies to numpy and cannot oscillate the way a high-order spline can. It
    does NOT guarantee an arbitrage-free call surface, which is why the recovered
    curve is checked afterwards rather than assumed.
    """
    if len(points) < MIN_POINTS:
        raise ChainUnusable("only %d usable contracts" % len(points))

    strikes = np.array([p[0] for p in points], dtype=float)
    vols = np.array([p[1] for p in points], dtype=float)
    spreads = np.array([p[2] if len(p) > 2 else 0.0 for p in points], dtype=float)
    x = np.log(strikes / fwd)

    span = float(x.max() - x.min())
    if span < MIN_SPAN:
        raise ChainUnusable("the usable strikes span only %.3f in log-moneyness" % span)

    #--- two weights multiplied. The first concentrates the fit near the money, and a
    #--- narrow choice there flattens the wings, so its width is swept in the
    #--- sensitivity study rather than treated as settled. The second trusts a tight
    #--- quote more than a wide one, which is what a bid-ask spread is evidence about
    weights = np.exp(-(x / width) ** 2) / (1.0 + spreads / 0.05)
    for deg in (degree, degree - 1, 1):
        if deg < 1 or len(points) < deg + 3:
            continue
        try:
            coeffs = np.polyfit(x, vols, deg, w=weights)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.all(np.isfinite(coeffs)):
            continue
        if debug:
            print("  smile fitted at degree %d over %d points, span %.3f"
                  % (deg, len(points), span))
        return {"coeffs": coeffs, "xlo": float(x.min()), "xhi": float(x.max()),
                "degree": deg}

    flat = float(np.average(vols, weights=weights))
    if debug:
        print("  smile fit failed, falling back to a flat %.4f" % flat)
    return {"coeffs": np.array([flat]), "xlo": float(x.min()), "xhi": float(x.max()),
            "degree": 0}


def smooth_vol(fit, fwd, strike, t_years, tail=TAIL_RULE):
    """
    Evaluate the fitted smile at one strike, under an explicit tail rule.

    Inside the quoted range the polynomial is used. Outside it the fit must never be
    extrapolated, because a polynomial diverges past its data and invents probability
    mass. Two conventions are offered:
      flat    hold the edge volatility, which makes the far tails lognormal
      linvar  extend total variance linearly at the slope of the edge
    Neither is the market's opinion. The tails outside the quoted strikes are a
    convention, and the sensitivity study measures how much that convention matters.
    """
    x = math.log(strike / fwd)
    lo, hi = fit["xlo"], fit["xhi"]
    if lo <= x <= hi:
        vol = float(np.polyval(fit["coeffs"], x))
    elif tail == "flat":
        vol = float(np.polyval(fit["coeffs"], min(max(x, lo), hi)))
    else:
        edge = hi if x > hi else lo
        step = 1e-4
        v_edge = float(np.polyval(fit["coeffs"], edge))
        v_in = float(np.polyval(fit["coeffs"], edge - step if x > hi else edge + step))
        w_edge = v_edge * v_edge * t_years
        w_in = v_in * v_in * t_years
        slope = (w_edge - w_in) / (step if x > hi else -step)
        w = max(w_edge + slope * (x - edge), 1e-8)
        vol = math.sqrt(w / t_years)
    return min(max(vol, IV_FLOOR), IV_CEIL)


def density_from_smile(fit, fwd, t_years, rate, tail=TAIL_RULE):
    """
    Recover the risk-neutral density, and report what had to be repaired to get it.

    The identity assumes an arbitrage-consistent call curve. A polynomial smile does
    not guarantee one, so three things are measured rather than assumed: whether the
    call curve falls monotonically with strike, whether it is convex, and how much
    negative probability mass the second difference produced. Negative mass is clipped
    because a density cannot be negative, but the amount is published, because a large
    figure means the fit was not arbitrage-consistent.
    """
    grid = np.linspace(fwd * GRID_LO, fwd * GRID_HI, GRID_STEPS)
    step = float(grid[1] - grid[0])
    calls = np.array([b76_call(fwd, k, t_years, smooth_vol(fit, fwd, k, t_years, tail), rate)
                      for k in grid])

    #--- static arbitrage on the recovered call curve, before anything is repaired
    #--- the tolerance is scaled to the size of the curve itself, because an absolute
    #--- threshold counts floating point noise in the far wings, where call prices are
    #--- almost zero, as if it were a real convexity failure
    first = np.diff(calls) / step
    second = (calls[2:] - 2.0 * calls[1:-1] + calls[:-2]) / (step * step)
    tol_first = 1e-6 * max(float(np.max(np.abs(first))), 1e-12)
    tol_second = 1e-6 * max(float(np.max(np.abs(second))), 1e-12)
    mono_bad = int(np.sum(first > tol_first))
    convex_bad = int(np.sum(second < -tol_second))

    #--- no-arbitrage bounds on the call itself: it can never be worth less than its
    #--- discounted intrinsic value, nor more than the discounted forward
    disc = math.exp(-rate * t_years)
    lower = disc * np.maximum(fwd - grid, 0.0)
    upper = disc * fwd
    bound_bad = int(np.sum(calls < lower - tol_second) + np.sum(calls > upper + tol_second))

    strikes = grid[1:-1]
    dens = math.exp(rate * t_years) * second
    gross = float(np.sum(np.abs(dens)) * step)
    neg_mass = float(np.sum(np.abs(dens[dens < 0.0])) * step)
    neg_pct = (100.0 * neg_mass / gross) if gross > 0 else float("nan")

    dens = np.clip(dens, 0.0, None)
    area = float(np.sum(dens) * step)
    if area <= 0.0:
        raise ChainUnusable("the recovered density had no usable mass")
    dens = dens / area
    cdf = np.cumsum(dens) * step
    diag = {"neg_mass_pct": neg_pct, "mono_violations": mono_bad,
            "convex_violations": convex_bad, "bound_violations": bound_bad}
    return strikes, dens, cdf, diag


def choose_fit(points, fwd, t_years, rate, tail=None, debug=False):
    """
    Pick the smile specification whose recovered call curve is arbitrage-consistent.

    Two decisions are searched together, because they interact. The polynomial degree
    sets how much curvature the smile may have. The tail rule sets what happens outside
    the quoted strikes. Holding volatility flat out there is the cruder of the two
    rules: where a sloped polynomial meets a flat extension there is a kink, a kink in
    the volatility curve becomes a non-convex spot in the call curve, and a non-convex
    call curve produces negative probability. Extending total variance linearly at the
    slope of the edge joins smoothly instead, so it is tried first.

    Every combination is recovered in full and the first one that leaves less than
    NEG_MASS_MAX of negative probability mass is taken, richest degree first. Negative
    mass is the gate because it measures the economic damage directly. If nothing
    qualifies the least bad is used, and the figures are published either way.
    """
    attempts = []
    for deg in (FIT_DEGREE, 2, 1):
        for rule in (("linvar", "flat") if tail is None else (tail,)):
            try:
                fit = fit_smile(points, fwd, degree=deg)
                out = density_from_smile(fit, fwd, t_years, rate, rule)
            except (ChainUnusable, ValueError, np.linalg.LinAlgError):
                continue
            diag = out[3]
            fit = dict(fit)
            fit["tail"] = rule
            attempts.append((fit, out, diag))
            clean = diag["neg_mass_pct"] <= NEG_MASS_MAX
            if debug:
                print("  degree %d, %s tail: negative mass %.4f%%, convexity violations %d%s"
                      % (fit["degree"], rule, diag["neg_mass_pct"],
                         diag["convex_violations"], "   accepted" if clean else ""))
            if clean:
                return fit, out
    if not attempts:
        raise ChainUnusable("no specification produced a usable density")
    best = min(attempts, key=lambda a: a[2]["neg_mass_pct"])
    if debug:
        print("  nothing was fully arbitrage-consistent, using degree %d with the %s "
              "tail, the least bad" % (best[0]["degree"], best[0]["tail"]))
    return best[0], best[1]


def percentile(strikes, cdf, q):
    """Read a price level off the cumulative distribution."""
    return float(np.interp(q, cdf, strikes))


def moments(strikes, dens):
    """Mean, standard deviation, skewness and kurtosis on the density's own grid."""
    step = float(strikes[1] - strikes[0])
    mean = float(np.sum(strikes * dens) * step)
    var = float(np.sum((strikes - mean) ** 2 * dens) * step)
    sd = math.sqrt(max(var, 0.0))
    if sd <= 0.0:
        return mean, 0.0, 0.0, 0.0
    skew = float(np.sum((strikes - mean) ** 3 * dens) * step) / sd ** 3
    kurt = float(np.sum((strikes - mean) ** 4 * dens) * step) / sd ** 4
    return mean, sd, skew, kurt


def lognormal_shape(vol, t_years):
    """
    Skewness and kurtosis of the lognormal a flat smile would have produced.

    A lognormal is already right-skewed and already fat-tailed, so comparing the
    recovered shape against zero and three would be misleading.
    """
    s2 = vol * vol * t_years
    e = math.exp(s2)
    skew = (e + 2.0) * math.sqrt(max(e - 1.0, 0.0))
    kurt = math.exp(4.0 * s2) + 2.0 * math.exp(3.0 * s2) + 3.0 * math.exp(2.0 * s2) - 3.0
    return skew, kurt


def realized_volatility(closes, window=RV_WINDOW):
    """Annualized standard deviation of daily log returns."""
    logret = np.log(closes / closes.shift(1)).dropna()
    if len(logret) < window:
        return float("nan")
    return float(logret.tail(window).std(ddof=1) * math.sqrt(252))


def pick_expiry(tk):
    """Choose the listed expiry nearest 30 days inside the accepted window."""
    today = datetime.now(timezone.utc).date()
    best, best_gap = None, 10 ** 9
    for expiry in tk.options:
        days = (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days
        if days < EXPIRY_LO or days > EXPIRY_HI:
            continue
        if abs(days - 30) < best_gap:
            best, best_gap = (expiry, days), abs(days - 30)
    return best


def sensitivity(chain, fwd, t_years, rate, debug=False):
    """
    Recover the density again under every reasonable alternative choice, and report
    how far the answer moves.

    Three decisions in this pipeline are conventions rather than measurements: the
    polynomial degree, the strike window that is fitted, and the rule applied outside
    the quoted strikes. A number that survives all of them is a market observation. A
    number that swings is an artifact of the method, and the reader is entitled to
    know which is which before using it.
    """
    rows = []
    for deg in SENS_DEGREES:
        for lo, hi in SENS_WINDOWS:
          for width in SENS_WIDTHS:
            for tail in SENS_TAILS:
                try:
                    pts = collect_smile(chain, fwd, t_years, rate, lo, hi)
                    f = fit_smile(pts, fwd, degree=deg, width=width)
                    st, de, cd, dg = density_from_smile(f, fwd, t_years, rate, tail)
                    _, sd, sk, ku = moments(st, de)
                    rows.append({
                        "p05": percentile(st, cd, 0.05) / fwd,
                        "p25": percentile(st, cd, 0.25) / fwd,
                        "p50": percentile(st, cd, 0.50) / fwd,
                        "p75": percentile(st, cd, 0.75) / fwd,
                        "p95": percentile(st, cd, 0.95) / fwd,
                        "sd": sd / fwd, "skew": sk, "kurt": ku,
                        "neg": dg["neg_mass_pct"]})
                except (ChainUnusable, ValueError, np.linalg.LinAlgError):
                    continue
    if not rows:
        return {}
    out = {"runs": len(rows)}
    for key in ("p05", "p25", "p50", "p75", "p95", "sd", "skew", "kurt", "neg"):
        vals = [row[key] for row in rows]
        out[key + "_min"] = float(min(vals))
        out[key + "_max"] = float(max(vals))
    if debug:
        print("  sensitivity over %d combinations of degree, window and tail rule:"
              % len(rows))
        for key, label in (("p05", "5th pct "), ("p25", "25th pct"), ("p50", "median  "),
                           ("p75", "75th pct"), ("p95", "95th pct")):
            lo, hi = out[key + "_min"], out[key + "_max"]
            print("    %s  %.4f to %.4f of forward, spread %.2f%%"
                  % (label, lo, hi, 100.0 * (hi - lo)))
        print("    skewness  %+.3f to %+.3f" % (out["skew_min"], out["skew_max"]))
        print("    kurtosis   %.3f to %.3f" % (out["kurt_min"], out["kurt_max"]))
        print("    negative mass before clipping %.3f%% to %.3f%%"
              % (out["neg_min"], out["neg_max"]))
    return out


def filter_sensitivity(chain, fwd, t_years, rate, degree, tail, debug=False):
    """
    Vary the liquidity thresholds and report what they cost.

    Open interest and maximum spread are judgement, not physics. Loosening them lets
    in quotes nobody would trade at; tightening them throws away the wings, which is
    where the tail of the distribution lives. Measuring the effect is the only honest
    way to present a threshold.
    """
    rows = []
    for oi in SENS_MIN_OI:
        for sp in SENS_MAX_SPREAD:
            try:
                pts = collect_smile(chain, fwd, t_years, rate, MONEY_LO, MONEY_HI,
                                    min_oi=oi, max_spread=sp)
                f = fit_smile(pts, fwd, degree=degree)
                st, de, cd, _ = density_from_smile(f, fwd, t_years, rate, tail)
                rows.append({"n": len(pts),
                             "p05": percentile(st, cd, 0.05) / fwd,
                             "p50": percentile(st, cd, 0.50) / fwd,
                             "p95": percentile(st, cd, 0.95) / fwd})
            except (ChainUnusable, ValueError, np.linalg.LinAlgError):
                continue
    if not rows:
        return {}
    out = {"runs": len(rows)}
    for key in ("n", "p05", "p50", "p95"):
        vals = [r[key] for r in rows]
        out[key + "_min"] = min(vals)
        out[key + "_max"] = max(vals)
    if debug:
        print("  filter study over %d threshold combinations:" % len(rows))
        print("    contracts kept %d to %d" % (out["n_min"], out["n_max"]))
        print("    median  %.4f to %.4f of forward" % (out["p50_min"], out["p50_max"]))
        print("    5th pct %.4f to %.4f, 95th pct %.4f to %.4f"
              % (out["p05_min"], out["p05_max"], out["p95_min"], out["p95_max"]))
    return out


def build_feed(debug=False):
    """Pull the chain, recover the density, run every check, assemble the feed."""
    tk = yf.Ticker(TICKER)
    history = tk.history(period="6mo", interval="1d")
    if history.empty:
        raise RuntimeError("no price history returned for %s" % TICKER)
    spot = float(history["Close"].iloc[-1])
    rv = realized_volatility(history["Close"])

    chosen = pick_expiry(tk)
    if chosen is None:
        raise ChainUnusable("no listed expiry inside the accepted window")
    expiry, days = chosen
    t_years = days / 365.0
    if debug:
        print("  spot %.2f, expiry %s (%d days)" % (spot, expiry, days))

    chain = tk.option_chain(expiry)
    fwd, fwd_spread = forward_from_parity(chain, spot, t_years, RISK_FREE, debug)
    gap_max, gap_med = parity_iv_gap(chain, fwd, t_years, RISK_FREE)
    if debug and math.isfinite(gap_max):
        print("  put-call implied volatility gap near the money: median %.4f, worst %.4f"
              % (gap_med, gap_max))

    points = collect_smile(chain, fwd, t_years, RISK_FREE, MONEY_LO, MONEY_HI, debug)
    fit, recovered = choose_fit(points, fwd, t_years, RISK_FREE, None, debug)
    strikes, dens, cdf, diag = recovered
    chosen_tail = fit["tail"]

    levels = {q: percentile(strikes, cdf, q) for q in (0.05, 0.25, 0.50, 0.75, 0.95)}
    step = float(strikes[1] - strikes[0])
    area = float(np.sum(dens) * step)
    mean, sd, skew, kurt = moments(strikes, dens)
    atm_vol = smooth_vol(fit, fwd, fwd, t_years, chosen_tail)
    bench_skew, bench_kurt = lognormal_shape(atm_vol, t_years)
    prob_above = float(1.0 - np.interp(spot, strikes, cdf))
    sens = sensitivity(chain, fwd, t_years, RISK_FREE, debug)
    filt = filter_sensitivity(chain, fwd, t_years, RISK_FREE, fit["degree"],
                              chosen_tail, debug)

    if debug:
        print("  density integrates to %.6f" % area)
        print("  mean %.4f against the forward %.4f  (%.4f%%)"
              % (mean, fwd, 100.0 * (mean / fwd - 1.0)))
        print("  negative mass before clipping %.4f%% of gross" % diag["neg_mass_pct"])
        print("  monotonicity violations %d, convexity violations %d, bound violations %d"
              % (diag["mono_violations"], diag["convex_violations"],
                 diag["bound_violations"]))
        print("  one standard deviation %.2f, flat-smile Black-76 says %.2f"
              % (sd, fwd * atm_vol * math.sqrt(t_years)))
        print("  skewness %+.4f against a lognormal %+.4f" % (skew, bench_skew))
        print("  kurtosis %.4f against a lognormal %.4f" % (kurt, bench_kurt))
        print("  90 percent band %.2f to %.2f" % (levels[0.05], levels[0.95]))

    return {
        "symbol": TICKER,
        "underlying_note": "GLD ETF, not spot gold and not broker XAUUSD",
        "spot": round(spot, 2),
        "forward": round(fwd, 4),
        "forward_source": "put-call parity",
        "net_carry_pct": round(100.0 * math.log(fwd / spot) / t_years, 4),
        "expiry": expiry,
        "days_to_expiry": days,
        "atm_iv": round(atm_vol, 6),
        "rv_30d_gld": round(rv, 6) if math.isfinite(rv) else None,
        "r05": round(levels[0.05] / fwd, 6),
        "r25": round(levels[0.25] / fwd, 6),
        "r50": round(levels[0.50] / fwd, 6),
        "r75": round(levels[0.75] / fwd, 6),
        "r95": round(levels[0.95] / fwd, 6),
        "prob_above_spot_gld": round(prob_above, 6),
        "implied_move_1sd_pct": round(100.0 * sd / fwd, 4),
        "bs_move_1sd_pct": round(100.0 * atm_vol * math.sqrt(t_years), 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "bench_skewness": round(bench_skew, 4),
        "bench_kurtosis": round(bench_kurt, 4),
        "density_area": round(area, 6),
        "neg_mass_pct": round(diag["neg_mass_pct"], 6),
        "mono_violations": diag["mono_violations"],
        "convex_violations": diag["convex_violations"],
        "bound_violations": diag["bound_violations"],
        "parity_iv_gap_max": round(gap_max, 6) if math.isfinite(gap_max) else None,
        "forward_spread_pct": round(100.0 * fwd_spread, 4),
        "contracts_used": len(points),
        "fit_degree": fit["degree"],
        "tail_rule": chosen_tail,
        "sens_runs": sens.get("runs", 0),
        "sens_p05_min": round(sens.get("p05_min", float("nan")), 6),
        "sens_p05_max": round(sens.get("p05_max", float("nan")), 6),
        "sens_p50_min": round(sens.get("p50_min", float("nan")), 6),
        "sens_p50_max": round(sens.get("p50_max", float("nan")), 6),
        "sens_p95_min": round(sens.get("p95_min", float("nan")), 6),
        "sens_p95_max": round(sens.get("p95_max", float("nan")), 6),
        "sens_kurt_min": round(sens.get("kurt_min", float("nan")), 4),
        "sens_kurt_max": round(sens.get("kurt_max", float("nan")), 4),
        "sens_neg_mass_min": round(sens.get("neg_min", float("nan")), 4),
        "sens_neg_mass_max": round(sens.get("neg_max", float("nan")), 4),
        "filter_runs": filt.get("runs", 0),
        "filter_contracts_min": filt.get("n_min", 0),
        "filter_contracts_max": filt.get("n_max", 0),
        "filter_p50_min": round(filt.get("p50_min", float("nan")), 6),
        "filter_p50_max": round(filt.get("p50_max", float("nan")), 6),
        "filter_p05_min": round(filt.get("p05_min", float("nan")), 6),
        "filter_p05_max": round(filt.get("p05_max", float("nan")), 6),
        "reliable": "median and the 25 to 75 band",
        "less_reliable": "skewness, kurtosis and the 5 and 95 tails",
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "GLD option chain, risk-neutral density by Breeden-Litzenberger",
    }


def _inv_erf(y):
    """Inverse error function by bisection, so the self test needs no scipy."""
    lo, hi = -6.0, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erf(mid) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def selftest(fwd=415.0, vol=0.25, days=29, rate=RISK_FREE, seed=7):
    """
    Two checks. The first proves the numerical engine, the second probes its
    robustness to the thing that actually breaks it in the market: quote noise.

    Check one, the flat smile. One constant volatility at every strike is Black-76
    with no skew, whose terminal distribution is lognormal with a mean of exactly the
    forward and percentiles that can be written down in closed form.

    Check two, the noisy smile. The same flat smile is perturbed with errors of the
    size a bid-ask spread produces, refitted and recovered again. This is the test the
    flat-smile check cannot do: it shows how much of the recovered shape survives
    realistic noise, and how much negative mass appears when the fit stops being
    arbitrage-consistent.
    """
    t = days / 365.0
    print("self test 1: flat smile at %.4f, %d days, against the analytic lognormal"
          % (vol, days))
    fit = {"coeffs": np.array([vol]), "xlo": -1.0, "xhi": 1.0, "degree": 0}
    st, de, cd, diag = density_from_smile(fit, fwd, t, rate, "flat")
    mean, sd, skew, kurt = moments(st, de)
    step = float(st[1] - st[0])
    print("  density integrates to        %.8f   (exactly 1)" % (np.sum(de) * step))
    print("  mean                         %10.4f   forward %10.4f" % (mean, fwd))
    for q in (0.05, 0.25, 0.50, 0.75, 0.95):
        z = math.sqrt(2.0) * _inv_erf(2.0 * q - 1.0)
        exact = fwd * math.exp(-0.5 * vol * vol * t + vol * math.sqrt(t) * z)
        got = percentile(st, cd, q)
        print("  p%02d  recovered %10.4f   exact %10.4f   error %+.5f%%"
              % (q * 100, got, exact, 100.0 * (got / exact - 1.0)))
    bs, bk = lognormal_shape(vol, t)
    print("  skewness  recovered %+.5f   exact %+.5f" % (skew, bs))
    print("  kurtosis  recovered  %.5f   exact  %.5f" % (kurt, bk))
    print("  negative mass %.6f%%, monotonicity violations %d, convexity violations %d"
          % (diag["neg_mass_pct"], diag["mono_violations"], diag["convex_violations"]))

    print()
    print("self test 2: the same smile with bid-ask noise, refitted and recovered")
    print("            and then thinned, to show what sparse strikes cost")
    rng = np.random.default_rng(seed)
    strikes = np.linspace(fwd * 0.80, fwd * 1.20, 40)
    for noise in (0.000, 0.005, 0.015):
        pts = [(float(k), float(max(vol + rng.normal(0.0, noise), IV_FLOOR + 1e-4)))
               for k in strikes]
        f = fit_smile(pts, fwd, FIT_DEGREE)
        st, de, cd, diag = density_from_smile(f, fwd, t, rate, "flat")
        _, sd, skew, kurt = moments(st, de)
        print("  noise %.1f vol points: kurtosis %.3f (exact %.3f), skewness %+.3f "
              "(exact %+.3f), negative mass %.4f%%, convexity violations %d"
              % (noise * 100, kurt, bk, skew, bs, diag["neg_mass_pct"],
                 diag["convex_violations"]))

    #--- sparse strikes are the other way a real chain differs from the ideal one: an
    #--- overnight or illiquid expiry may quote a handful of widely spaced strikes,
    #--- and a curve through few bunched points is not a smile
    for keep in (40, 20, 10, 8):
        sub = np.linspace(fwd * 0.80, fwd * 1.20, keep)
        pts = [(float(k), float(max(vol + rng.normal(0.0, 0.005), IV_FLOOR + 1e-4)), 0.02)
               for k in sub]
        try:
            f = fit_smile(pts, fwd, FIT_DEGREE)
            st, de, cd, diag = density_from_smile(f, fwd, t, rate, "flat")
            _, sd, skew, kurt = moments(st, de)
            print("  %2d strikes: kurtosis %.3f (exact %.3f), negative mass %.4f%%, "
                  "convexity violations %d"
                  % (keep, kurt, bk, diag["neg_mass_pct"], diag["convex_violations"]))
        except ChainUnusable as why:
            print("  %2d strikes: refused, %s" % (keep, why))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Publish the option-implied distribution of GLD as JSON")
    parser.add_argument("--out", help="also write the JSON to this file")
    parser.add_argument("--debug", action="store_true", help="show the fit and the checks")
    parser.add_argument("--selftest", action="store_true",
                        help="check the engine against algebra and against noise, then exit")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    if args.debug:
        print("recovering the risk-neutral density from market prices:")

    try:
        feed = build_feed(args.debug)
    except ChainUnusable as reason:
        # a thin chain is a normal event on a quiet night. Leaving the last good feed
        # in place is correct, and exiting zero keeps the scheduled job green
        print("chain unusable today: %s" % reason)
        print("the previous feed is left untouched")
        return 0

    text = json.dumps(feed, indent=2)
    print(("\n" if args.debug else "") + text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print("\nwritten to %s" % args.out)

    print("\nGLD, expiry %s in %d days, %d contracts used, fit degree %d"
          % (feed["expiry"], feed["days_to_expiry"], feed["contracts_used"],
             feed["fit_degree"]))
    print("forward %.4f from put-call parity, net carry %+.2f%% a year"
          % (feed["forward"], feed["net_carry_pct"]))
    print("90 percent band: %.1f%% to %.1f%% of the forward"
          % (feed["r05"] * 100, feed["r95"] * 100))
    print("negative mass %.4f%%, monotonicity %d, convexity %d"
          % (feed["neg_mass_pct"], feed["mono_violations"], feed["convex_violations"]))
    print("median is stable across %d specifications: %.4f to %.4f"
          % (feed["sens_runs"], feed["sens_p50_min"], feed["sens_p50_max"]))
    print("kurtosis is not: %.2f to %.2f across the same specifications"
          % (feed["sens_kurt_min"], feed["sens_kurt_max"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
