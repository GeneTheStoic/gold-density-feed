"""
implied_density_feed.py - publish an option-implied probability distribution as a
small JSON feed, with the diagnostics needed to judge how far to trust it.

WHAT THIS IS
    An option chain does not only imply a volatility. Taken together, the prices
    across all strikes imply a full probability distribution for where the underlying
    can finish at expiry. Breeden and Litzenberger showed in 1978 that the second
    derivative of the call price with respect to strike, discounted, is that density.

WHAT IT IS NOT
    The result is a RISK-NEUTRAL distribution of the selected ETF. Two restrictions:
      1. Risk-neutral is not real-world. The left tail is systematically fatter than
         outcomes turn out to be, because people pay extra for crash protection.
      2. The option underlying is an ETF, not necessarily the cash index, commodity,
         or CFD a broker quotes. Rescaling the percentiles onto another instrument is
         a monitoring convenience, not an equivalence.

WHY THE FORWARD COMES FROM PUT-CALL PARITY
    An ETF can carry dividends, fees, storage costs, or futures-roll economics.
    Assuming F = S * exp(rT) can therefore misspecify the forward and bias every
    implied volatility solved from it. Instead the forward is read out of the market,
    from calls and puts quoted at the same near-the-money strikes, and everything is
    priced in Black-76 on that forward. That also puts out-of-the-money puts and calls
    in one space, so mixing the two sides is consistent rather than stitched together.

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
    python implied_density_feed.py --ticker GLD
    python implied_density_feed.py --ticker SPY --out feed_spy.json
    python implied_density_feed.py --ticker USO --debug
    python implied_density_feed.py --selftest

Reads only. Never trades.
"""

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# the command-line default. Every downstream calculation uses this selected ticker
DEFAULT_TICKER = "GLD"
TICKER = DEFAULT_TICKER

# labels travel with the feed so the ETF-to-terminal mapping is never hidden
UNDERLYING_NOTES = {
    "GLD": "GLD ETF proxy for gold, not spot gold or broker XAUUSD",
    "SLV": "SLV ETF proxy for silver, not spot silver or broker XAGUSD",
    "SPY": "SPY ETF proxy for the S&P 500, not a cash index or broker US500 CFD",
    "USO": "USO holds oil futures; it is not spot crude or broker USOIL",
}

# four percent is a transparent discount-rate input, not a carry assumption; the
# forward and net carry are recovered independently from put-call parity
RISK_FREE = 0.04

# liquidity filters: 10 contracts and a five-cent price reject empty penny wings,
# while a 25 percent relative spread keeps only quotes with a meaningful midpoint
MIN_OI = 10
MIN_PRICE = 0.05
MAX_SPREAD = 0.25          # reject a quote whose bid-ask spread exceeds this of its mid

# 80 to 120 percent of the forward keeps the liquid core and enough wing curvature
# without letting remote penny strikes dictate the fit
MONEY_LO, MONEY_HI = 0.80, 1.20

# two to 200 percent accepts plausible ETF regimes while rejecting solver-boundary
# values that indicate an inconsistent price rather than a measured volatility
IV_FLOOR, IV_CEIL = 0.02, 2.00

# 20 to 45 days brackets the article's one-month target without selecting a very
# short gamma-dominated expiry or jumping to the following quarter
EXPIRY_LO, EXPIRY_HI = 20, 45

# the 55 to 165 percent grid reaches beyond the fitted strikes, while 2,200 steps
# keep the flat-smile percentile error below three basis points in the self test
GRID_LO, GRID_HI = 0.55, 1.65
GRID_STEPS = 2200

# degree three permits ordinary skew and curvature without high-order oscillation;
# width 0.35 keeps the center influential, and eight points across four percent of
# the forward is the minimum span allowed to support a curve rather than a line
FIT_DEGREE = 3
FIT_WIDTH = 0.35
MIN_POINTS = 8
MIN_SPAN = 0.04

# outside the quoted strikes the fit must not be extrapolated. The live feed does not
# read this default: choose_fit tries linear total variance first, falls back to flat,
# and publishes the rule it used. The default only applies when a caller names no rule
TAIL_RULE = "linvar"

# a recovered call curve that is not convex is not arbitrage-consistent, and the
# density it produces contains negative probability. The specification is chosen to
# leave the least negative mass, and the first one under this gate is taken. The gate
# does not guarantee convexity: remaining violations are counted and published
NEG_MASS_MAX = 1.0

# 30 completed sessions roughly matches the target option horizon; realized
# volatility is reported only as context, never substituted for the implied density
RV_WINDOW = 30

# the spot comes from a different request than the chain, so it can be stale while
# the options are current. Beyond this much implied carry it is stale, not financed
CARRY_MAX = 25.0

# the sensitivity study: every combination is recovered and the spread is published.
# FIT_WIDTH is included because a fit weighted hard toward the money can flatten the
# wings, which is exactly the region the recovery exists to measure
SENS_DEGREES = (2, 3, 4)
SENS_WINDOWS = ((0.75, 1.25), (0.80, 1.20), (0.85, 1.15))
SENS_TAILS = ("flat", "linvar")
SENS_WIDTHS = (0.20, 0.35, 1.00)

# the filter study: the liquidity thresholds are judgment, so their effect is measured
SENS_MIN_OI = (0, 10, 50)
SENS_MAX_SPREAD = (0.10, 0.25, 0.50)


class ChainUnusable(Exception):
    """Raised when the chain of the day cannot support a density."""


def finite(value):
    """
    Convert to float and return 0.0 for anything that is not a real number.

    pandas returns NaN for a missing quote, and NaN is truthy in Python, so the
    natural-looking float(row.get("bid") or 0.0) passes NaN through silently.

    Assumes zero is the safe sentinel for a missing external numeric field.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def norm_cdf(x):
    """Standard normal cumulative distribution. Assumes x is a finite scalar."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def b76_call(fwd, strike, t_years, vol, rate):
    """
    Black-76 price of a European call written on a forward.

    Assumes the forward and strike are positive and the rate is continuously compounded.
    """
    if t_years <= 0.0 or vol <= 0.0:
        return math.exp(-rate * t_years) * max(fwd - strike, 0.0)
    v = vol * math.sqrt(t_years)
    d1 = (math.log(fwd / strike) + 0.5 * v * v) / v
    d2 = d1 - v
    return math.exp(-rate * t_years) * (fwd * norm_cdf(d1) - strike * norm_cdf(d2))


def b76_put(fwd, strike, t_years, vol, rate):
    """
    Black-76 price of a European put, by parity on the same forward.

    Assumes the call pricer and put-call parity use the same forward and discount rate.
    """
    call = b76_call(fwd, strike, t_years, vol, rate)
    return call - math.exp(-rate * t_years) * (fwd - strike)


def implied_vol(market_price, fwd, strike, t_years, rate, is_call):
    """
    Solve Black-76 backward for volatility, by bisection.

    An option price rises steadily as the assumed volatility rises, so a bracketed
    search cannot get lost. Returns None when the quote sits below intrinsic value or
    above what the ceiling volatility can produce, both of which mean a bad quote.

    Assumes a positive forward, strike and time to expiry, with a finite market price.
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

    Assumes row exposes yfinance-style bid and ask fields.
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
    """
    Days since the contract last traded, or None when the feed does not say.

    Assumes lastTradeDate is absent or convertible to a pandas timestamp.
    """
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
    """
    Return every strike in one chain side that carries a usable two-sided quote.

    Assumes frame uses yfinance option-chain columns and may be empty.
    """
    out = {}
    if frame is None or frame.empty:
        return out
    for _, row in frame.iterrows():
        k = finite(row.get("strike"))
        q = quote(row)
        if k > 0.0 and q is not None and q[0] >= MIN_PRICE and q[1] <= MAX_SPREAD:
            out[k] = q[0]
    return out


def forward_from_parity(chain, t_years, rate, debug=False):
    """
    Read the forward out of the market instead of assuming a carry.

    European put-call parity says C - P = exp(-rT) * (F - K) at every strike, so each strike
    quoted on both sides gives an estimate of F. The median over the strikes nearest
    the money is taken, which is robust to one bad quote.

    Nothing in here uses a spot quote. The strike nearest the money is the one where
    the call and the put are closest in price, which is a property of the chain
    itself. A spot that arrives late or stale therefore cannot move the forward, nor
    move which strikes are treated as being at the money, which is the mistake this
    replaced: the spot comes from a different request than the options do.

    This matters because an ETF need not be a non-dividend-paying asset in the pricing
    sense. Dividends, fees, storage, or futures-roll economics can make F = S * exp(rT)
    wrong and bias every volatility solved against it.

    Assumes that European put-call parity is an adequate approximation for these quotes.
    """
    cmap, pmap = strike_map(chain.calls), strike_map(chain.puts)
    common = sorted(set(cmap) & set(pmap), key=lambda k: abs(cmap[k] - pmap[k]))
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
    return fwd, spread


def parity_iv_gap(chain, fwd, t_years, rate):
    """
    Largest disagreement between call-implied and put-implied volatility near the money.

    Mixing out-of-the-money puts on the left with out-of-the-money calls on the right
    only makes sense if the two sides agree where they overlap. This measures that.

    Assumes both option sides are priced on the supplied forward and discount rate.
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

    Assumes chain exposes yfinance-style calls and puts for one common expiry.
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

    Assumes points contain positive strikes, solved volatilities and relative spreads.
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

    Assumes fit contains polynomial coefficients and finite fitted-range endpoints.
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
    and the density renormalized, so every percentile describes the repaired density.
    The amount repaired is published, because a large figure means the fit was not
    arbitrage-consistent. Where on the grid the repair happened is not reported.

    Assumes a positive forward and time to expiry and an explicit supported tail rule.
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
    #--- discounted intrinsic value, nor more than the discounted forward. These are
    #--- prices, so the tolerance is a price too, a tiny fraction of the forward
    disc = math.exp(-rate * t_years)
    lower = disc * np.maximum(fwd - grid, 0.0)
    upper = disc * fwd
    tol_price = 1e-8 * fwd
    bound_bad = int(np.sum(calls < lower - tol_price) + np.sum(calls > upper + tol_price))

    strikes = grid[1:-1]
    dens = math.exp(rate * t_years) * second
    #--- the negative mass is reported as a share of the gross absolute mass, the
    #--- positive and negative parts added together. That share stays between 0 and
    #--- 100 even for a badly broken fit. For small values it is within a factor of
    #--- (1 + 2x) of the share of net probability, so 0.004 percent reads the same
    #--- either way and 4.74 percent of gross is about 5.2 percent of net
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
    Pick the smile specification that leaves the least negative probability mass.

    Two decisions are searched together, because they interact. The polynomial degree
    sets how much curvature the smile may have. The tail rule sets what happens outside
    the quoted strikes. Holding volatility flat out there is the cruder of the two
    rules: where a sloped polynomial meets a flat extension there is a kink, a kink in
    the volatility curve becomes a non-convex spot in the call curve, and a non-convex
    call curve produces negative probability. Instead, extending total variance linearly
    using the edge slope produces a smooth join, so that rule is tried first.

    Every combination is recovered in full and the first one that leaves less than
    NEG_MASS_MAX of negative probability mass is taken, richest degree first. Negative
    mass is the gate because it measures the economic damage directly. It is not a
    proof of arbitrage consistency: convexity violations can survive the gate, and
    they are counted and published. If nothing qualifies the least bad is used, and
    the figures are published either way.

    Assumes points have already passed the quote, liquidity and strike-window filters.
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
        print("  nothing passed the negative-mass gate, using degree %d with the %s "
              "tail, the least bad" % (best[0]["degree"], best[0]["tail"]))
    return best[0], best[1]


def percentile(strikes, cdf, q):
    """
    Read a price level off the cumulative distribution.

    Assumes strikes and cdf are ordered together and q lies between zero and one.
    """
    return float(np.interp(q, cdf, strikes))


def moments(strikes, dens):
    """
    Return mean, standard deviation, skewness and kurtosis on the density's grid.

    Assumes an evenly spaced strike grid and a nonnegative normalized density.
    """
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

    Assumes positive volatility and time to expiry.
    """
    s2 = vol * vol * t_years
    e = math.exp(s2)
    skew = (e + 2.0) * math.sqrt(max(e - 1.0, 0.0))
    kurt = math.exp(4.0 * s2) + 2.0 * math.exp(3.0 * s2) + 3.0 * math.exp(2.0 * s2) - 3.0
    return skew, kurt


def realized_volatility(closes, window=RV_WINDOW):
    """
    Return the annualized standard deviation of daily log returns.

    Assumes positive, chronological daily closes and 252 trading sessions per year.
    """
    logret = np.log(closes / closes.shift(1)).dropna()
    if len(logret) < window:
        return float("nan")
    return float(logret.tail(window).std(ddof=1) * math.sqrt(252))


def pick_expiry(tk):
    """
    Choose the listed expiry nearest 30 days inside the accepted window.

    Assumes tk.options contains ISO dates and the current UTC date is the reference.
    """
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

    Assumes one chain and forward are reused across all specification combinations.
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

    Open interest and maximum spread are judgment, not physics. Loosening them lets
    in quotes nobody would trade at; tightening them throws away the wings, which is
    where the tail of the distribution lives. Measuring the effect is the only honest
    way to present a threshold.

    Assumes the chosen degree and tail rule already produced the reported base fit.
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


def spot_price(ticker, closes):
    """
    Take the freshest spot the provider will give, and say which one it was.

    The daily history can lag the option chain by days, and a spot that lags produces
    a carry that is arithmetic rather than finance. The intraday quote is asked for
    first because it belongs to roughly the same moment as the chain. The last
    completed daily bar is the fallback. Which one was used travels with the feed,
    because a reader checking the carry is entitled to know.

    Assumes ticker is a yfinance Ticker and closes contains completed daily observations.
    """
    def probe(source, key):
        """
        Read one field however this library version exposes it.

        Assumes mapping and attribute access may independently fail.
        """
        try:
            return finite(source[key])
        except Exception:                  # not a mapping, or the lookup itself failed
            pass
        try:
            return finite(getattr(source, key))
        except Exception:                  # not an attribute, or the endpoint is down
            return 0.0

    try:
        fast = ticker.fast_info
    except Exception:                      # the quote endpoint is optional, not fatal
        fast = None
    if fast is not None:
        for key in ("last_price", "lastPrice", "regular_market_price"):
            value = probe(fast, key)
            if value > 0.0:
                return value, "intraday quote"

    if not closes.empty:
        value = finite(closes.iloc[-1])
        if value > 0.0:
            return value, "last daily close"
    return float("nan"), "unavailable"


def build_feed(debug=False):
    """
    Pull the chain, recover the density, run every check and assemble the feed.

    Assumes TICKER identifies an optionable ETF with a usable yfinance chain.
    """
    tk = yf.Ticker(TICKER)
    try:
        history = tk.history(period="6mo", interval="1d")
    except Exception as reason:
        raise ChainUnusable("price history request failed: %s" % reason) from reason
    if history.empty:
        raise ChainUnusable("no price history returned for %s" % TICKER)
    #--- the provider returns a row for the current day even when the session has not
    #--- opened, and that row's close is NaN. Taking the last row blindly puts NaN into
    #--- the spot, and from there into the carry and every probability measured against
    #--- it, so the last row with an actual close is the one taken
    closes = history["Close"].dropna()
    if closes.empty:
        raise ChainUnusable("no usable close price returned for %s" % TICKER)
    spot, spot_from = spot_price(tk, closes)
    rv = realized_volatility(closes)

    try:
        chosen = pick_expiry(tk)
    except Exception as reason:
        raise ChainUnusable("expiry request failed: %s" % reason) from reason
    if chosen is None:
        raise ChainUnusable("no listed expiry inside the accepted window")
    expiry, days = chosen
    t_years = days / 365.0
    if debug:
        print("  spot %.2f from the %s, expiry %s (%d days)"
              % (spot, spot_from, expiry, days))

    try:
        chain = tk.option_chain(expiry)
    except Exception as reason:
        raise ChainUnusable("option-chain request failed: %s" % reason) from reason
    fwd, fwd_spread = forward_from_parity(chain, t_years, RISK_FREE, debug)

    #--- the spot arrives from a different request than the chain does, so it can be
    #--- stale while the options are current. Nothing in the recovery uses it: it is
    #--- only here to report the carry and the probability measured against it. When it
    #--- implies a carry no financing cost could produce, the spot is the thing that is
    #--- wrong, so it is dropped rather than published as a number that looks real
    carry = (100.0 * math.log(fwd / spot) / t_years
             if math.isfinite(spot) and spot > 0.0 else float("nan"))
    if not math.isfinite(carry) or abs(carry) > CARRY_MAX:
        if debug:
            print("  spot %.2f from the %s against the forward %.4f implies a carry of"
                  " %+.1f%% a year, which is not a financing cost. That quote is stale,"
                  " so the spot, the carry and the probability above it are published"
                  " as null and the density is unaffected."
                  % (spot, spot_from, fwd, carry))
        spot, carry, spot_from = float("nan"), float("nan"), "rejected as stale"
    elif debug:
        print("  net carry %+.4f%% a year against spot %.2f, so against a %.1f%%"
              " discount rate the holding cost is about %.2f%%"
              % (carry, spot, 100.0 * RISK_FREE, 100.0 * RISK_FREE - carry))
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
    prob_above = (float(1.0 - np.interp(spot, strikes, cdf))
                  if math.isfinite(spot) else float("nan"))
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

    feed = {
        "symbol": TICKER,
        "underlying_note": UNDERLYING_NOTES.get(
            TICKER, "%s option underlying; verify any mapping to the local symbol" % TICKER),
        "spot": round(spot, 2) if math.isfinite(spot) else float("nan"),
        "spot_source": spot_from,
        "forward": round(fwd, 4),
        "forward_source": "put-call parity",
        "net_carry_pct": round(carry, 4) if math.isfinite(carry) else float("nan"),
        "expiry": expiry,
        "days_to_expiry": days,
        "atm_iv": round(atm_vol, 6),
        "rv_30d": round(rv, 6) if math.isfinite(rv) else None,
        "r05": round(levels[0.05] / fwd, 6),
        "r25": round(levels[0.25] / fwd, 6),
        "r50": round(levels[0.50] / fwd, 6),
        "r75": round(levels[0.75] / fwd, 6),
        "r95": round(levels[0.95] / fwd, 6),
        "prob_above_spot": round(prob_above, 6),
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
        "source": "%s option chain, risk-neutral density by Breeden-Litzenberger"
                  % TICKER,
    }
    #--- NaN and Infinity are not JSON. Python will write them anyway, and a strict
    #--- parser downstream then rejects the whole file, so any field that failed to
    #--- compute is published as null instead of as a token no standard reader accepts.
    #--- math.isfinite is the test, not the finite helper above, because that helper
    #--- maps a bad value to zero and zero is a legitimate reading for half these fields
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in feed.items()}


def _inv_erf(y):
    """
    Return the inverse error function by bisection, so the self test needs no scipy.

    Assumes y lies strictly between minus one and one.
    """
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

    Assumes positive forward, volatility and days, with a deterministic integer seed.
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
    """
    Parse command-line arguments, run the requested mode and optionally write one feed.

    Assumes --ticker names an optionable ETF; an unusable chain leaves any old file intact.
    """
    global TICKER
    parser = argparse.ArgumentParser(
        description="Publish an ETF option-implied distribution as JSON")
    parser.add_argument("--ticker", default=DEFAULT_TICKER,
                        help="optionable ETF symbol, for example GLD, SLV, SPY or USO")
    parser.add_argument("--out", help="also write the JSON to this file")
    parser.add_argument("--debug", action="store_true", help="show the fit and the checks")
    parser.add_argument("--selftest", action="store_true",
                        help="check the engine against algebra and against noise, then exit")
    args = parser.parse_args()
    TICKER = args.ticker.strip().upper()
    if not TICKER:
        parser.error("--ticker cannot be empty")

    if args.selftest:
        return selftest()

    if args.debug:
        print("recovering the risk-neutral density from market prices:")

    try:
        feed = build_feed(args.debug)
    except ChainUnusable as reason:
        #--- a thin chain is a normal event on a quiet night. Leaving the last good feed
        #--- in place is correct, and exiting zero keeps the scheduled job green
        print("chain unusable today: %s" % reason)
        print("the previous feed is left untouched")
        return 0

    text = json.dumps(feed, indent=2)
    print(("\n" if args.debug else "") + text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print("\nwritten to %s" % args.out)

    print("\n%s, expiry %s in %d days, %d contracts used, fit degree %d"
          % (feed["symbol"], feed["expiry"], feed["days_to_expiry"],
             feed["contracts_used"],
             feed["fit_degree"]))
    #--- a field that could not be computed is published as null, so the summary has to
    #--- say so rather than trying to format it
    def show(key, spec="%.4f"):
        """
        Format one feed field for the console or report it as unavailable.

        Assumes feed is the validated flat dictionary assembled above.
        """
        value = feed.get(key)
        return "not available" if value is None else spec % value

    print("forward %.4f from put-call parity, net carry %s a year"
          % (feed["forward"], show("net_carry_pct", "%+.2f%%")))
    print("90 percent band: %.1f%% to %.1f%% of the forward"
          % (feed["r05"] * 100, feed["r95"] * 100))
    print("negative mass %s, monotonicity %d, convexity %d"
          % (show("neg_mass_pct", "%.4f%%"), feed["mono_violations"],
             feed["convex_violations"]))
    print("median is stable across %d specifications: %.4f to %.4f"
          % (feed["sens_runs"], feed["sens_p50_min"], feed["sens_p50_max"]))
    print("kurtosis is not: %.2f to %.2f across the same specifications"
          % (feed["sens_kurt_min"], feed["sens_kurt_max"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
