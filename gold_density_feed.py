#!/usr/bin/env python3
"""
gold_density_feed.py - publish the option-implied probability distribution of gold
as a small JSON feed.

WHAT THIS IS
    An option chain does not only imply a volatility. Taken together, the prices
    across all strikes imply a full probability distribution for where the underlying
    can finish at expiry. Breeden and Litzenberger showed in 1978 that the second
    derivative of the call price with respect to strike, discounted, is that density.
    This script recovers it from a free public chain and publishes its percentiles.

WHAT IT IS NOT
    The result is a RISK-NEUTRAL distribution, not a forecast. It is the distribution
    implied by prices in a world indifferent to risk, and its left tail is
    systematically fatter than outcomes actually turn out to be, because people pay
    extra for crash protection. It answers "what is the market charging as if the odds
    were", never "what will happen".

WHY THE OUTPUT IS RELATIVE, NOT IN DOLLARS
    The chain is quoted on GLD, the gold ETF, near 400 dollars. A MetaTrader chart
    quotes XAUUSD in the thousands. Publishing dollar levels would be useless there,
    so the feed publishes multiplicative moves (0.94, 1.00, 1.06 and so on) that any
    terminal can apply to its own gold price.

METHOD
    1. take the listed expiry nearest 30 days
    2. solve implied volatility for the liquid out-of-the-money contracts inside a
       sensible band of strikes, ignoring the far wings that trade in pennies
    3. fit a smooth curve through those volatilities in log-moneyness space
       (differentiating raw quotes twice would amplify noise into nonsense)
    4. hold that curve flat beyond the last fitted strike, because a polynomial
       extrapolated past its data diverges and invents probability in the tails
    5. reprice a dense grid of synthetic calls from the smoothed curve
    6. take the second difference across strikes to obtain the density
    7. clip small negatives, normalize, integrate to a distribution, read percentiles

DEPENDENCIES
    yfinance, pandas, numpy. Nothing else, on purpose: the smoothing is a plain
    weighted polynomial fit in numpy rather than a spline library.

USAGE
    pip install yfinance pandas numpy
    python gold_density_feed.py                  print the JSON
    python gold_density_feed.py --out feed.json  also write it to a file
    python gold_density_feed.py --debug          show the fit and the validation checks
"""

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "GLD"           # gold ETF: liquid options, no dividend, quoted in dollars
RISK_FREE = 0.04         # short-rate approximation
MIN_OI = 10              # ignore contracts with almost no open interest
MIN_PRICE = 0.05         # ignore contracts trading in pennies, their prices are noise
MONEY_LO, MONEY_HI = 0.80, 1.20    # fit only strikes inside this fraction of spot
IV_FLOOR, IV_CEIL = 0.02, 2.00     # sanity bounds on any solved volatility
EXPIRY_LO, EXPIRY_HI = 20, 45      # accept an expiry in this window, nearest 30 wins
GRID_LO, GRID_HI = 0.55, 1.65      # dense strike grid, as a fraction of spot
GRID_STEPS = 2200        # resolution of that grid
FIT_DEGREE = 3           # polynomial degree for the smile fit in log-moneyness
FIT_WIDTH = 0.35         # weighting width of that fit, in log-moneyness
RV_WINDOW = 30           # sessions of realized volatility, for the comparison


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(spot, strike, t_years, vol, rate) -> float:
    """Black-Scholes price of a European call on a non-dividend-paying asset."""
    if t_years <= 0 or vol <= 0:
        return max(spot - strike, 0.0)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * math.sqrt(t_years))
    d2 = d1 - vol * math.sqrt(t_years)
    return spot * norm_cdf(d1) - strike * math.exp(-rate * t_years) * norm_cdf(d2)


def bs_put(spot, strike, t_years, vol, rate) -> float:
    """Black-Scholes price of a European put, by put-call parity."""
    call = bs_call(spot, strike, t_years, vol, rate)
    return call - spot + strike * math.exp(-rate * t_years)


def implied_vol(market_price, spot, strike, t_years, rate, is_call):
    """Solve Black-Scholes backward for volatility, by bisection."""
    intrinsic = max((spot - strike) if is_call else (strike - spot), 0.0)
    if market_price <= intrinsic + 1e-6:
        return None
    pricer = bs_call if is_call else bs_put
    low, high = IV_FLOOR, IV_CEIL
    if pricer(spot, strike, t_years, high, rate) < market_price:
        return None
    for _ in range(100):
        mid = 0.5 * (low + high)
        if pricer(spot, strike, t_years, mid, rate) < market_price:
            low = mid
        else:
            high = mid
        if high - low < 1e-6:
            break
    return 0.5 * (low + high)


def mid_price(row):
    """Prefer the bid/ask midpoint; fall back to the last trade."""
    bid = float(row.get("bid") or 0.0)
    ask = float(row.get("ask") or 0.0)
    if bid > 0 and ask > 0 and ask >= bid:
        return 0.5 * (bid + ask)
    last = float(row.get("lastPrice") or 0.0)
    return last if last > 0 else None


def pick_expiry(tk):
    """Choose the listed expiry nearest 30 days inside the accepted window."""
    today = datetime.now(timezone.utc).date()
    best, best_gap = None, 10 ** 9
    for expiry in tk.options:
        days = (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days
        if EXPIRY_LO <= days <= EXPIRY_HI and abs(days - 30) < best_gap:
            best, best_gap = (expiry, days), abs(days - 30)
    return best


def collect_smile(chain, spot, t_years, debug=False):
    """
    Solve implied volatility for the usable out-of-the-money contracts of one expiry.

    Out-of-the-money options carry the liquidity and almost all of the time value, so
    we read puts below the spot and calls above it, which is the market convention.
    Three filters decide what counts as usable: open interest, a live bid, and a price
    above a few cents. The far wings are excluded entirely. They trade in pennies, one
    tick of price granularity moves their implied volatility by whole points, and a
    curve fitted through them invents probability that nobody is actually pricing.
    """
    points = []
    for frame, is_call in ((chain.puts, False), (chain.calls, True)):
        if frame is None or frame.empty:
            continue
        usable = frame[frame["openInterest"].fillna(0) >= MIN_OI]
        if usable.empty:
            usable = frame
        for _, row in usable.iterrows():
            strike = float(row["strike"])
            if is_call and strike < spot:
                continue
            if (not is_call) and strike > spot:
                continue
            moneyness = strike / spot
            if moneyness < MONEY_LO or moneyness > MONEY_HI:
                continue
            if float(row.get("bid") or 0.0) <= 0.0:
                continue
            price = mid_price(row)
            if price is None or price < MIN_PRICE:
                continue
            vol = implied_vol(price, spot, strike, t_years, RISK_FREE, is_call)
            if vol is None or not (IV_FLOOR < vol < IV_CEIL):
                continue
            points.append((strike, vol))
    points.sort()
    if debug and points:
        print(f"  kept {len(points)} contracts, strikes {points[0][0]:.0f} to "
              f"{points[-1][0]:.0f} ({100.0 * points[0][0] / spot:.0f}% to "
              f"{100.0 * points[-1][0] / spot:.0f}% of spot)")
    return points


def fit_smile(points, spot):
    """
    Fit a smooth volatility curve in log-moneyness, and remember its valid range.

    Differentiating raw market quotes twice turns bid and ask noise into a meaningless
    density, so the quotes are smoothed first. A low-degree polynomial is enough for a
    single expiry, keeps the dependencies to numpy, and cannot oscillate the way a
    high-order spline can.

    The fitted range is recorded because the curve must never be extrapolated. Past the
    last fitted strike a polynomial diverges and invents probability mass in the tails,
    which shows up immediately as an impossible kurtosis.
    """
    if len(points) < FIT_DEGREE + 3:
        return None
    strikes = np.array([p[0] for p in points], dtype=float)
    vols = np.array([p[1] for p in points], dtype=float)
    x = np.log(strikes / spot)
    weights = np.exp(-((x / FIT_WIDTH) ** 2))
    coeffs = np.polyfit(x, vols, FIT_DEGREE, w=weights)
    return {"coeffs": coeffs, "xlo": float(x.min()), "xhi": float(x.max())}


def smooth_vol(fit, spot, strike):
    """
    Evaluate the fitted smile at one strike.

    Inside the fitted range the polynomial is used. Outside it, the volatility of the
    nearest fitted strike is held constant, which keeps the far tails lognormal
    instead of letting the fit run away.
    """
    x = math.log(strike / spot)
    x = min(max(x, fit["xlo"]), fit["xhi"])
    vol = float(np.polyval(fit["coeffs"], x))
    return min(max(vol, IV_FLOOR), IV_CEIL)


def density_from_smile(fit, spot, t_years, rate):
    """
    Recover the risk-neutral density by the Breeden-Litzenberger identity.

    Price a dense grid of synthetic calls from the smoothed smile, take the second
    difference across strikes, and undo the discounting. Small negative values are an
    artifact of the fit rather than real probabilities, so they are clipped and the
    result is renormalized to integrate to one.
    """
    grid = np.linspace(spot * GRID_LO, spot * GRID_HI, GRID_STEPS)
    step = grid[1] - grid[0]
    calls = np.array([bs_call(spot, k, t_years, smooth_vol(fit, spot, k), rate)
                      for k in grid])
    second = (calls[2:] - 2.0 * calls[1:-1] + calls[:-2]) / (step * step)
    strikes = grid[1:-1]
    dens = np.exp(rate * t_years) * second
    dens = np.clip(dens, 0.0, None)
    area = float(np.sum(dens) * step)
    if area <= 0:
        return None, None, None
    dens = dens / area
    cdf = np.cumsum(dens) * step
    return strikes, dens, cdf


def percentile(strikes, cdf, q):
    """Read a price level off the cumulative distribution."""
    return float(np.interp(q, cdf, strikes))


def realized_volatility(closes: pd.Series, window: int = RV_WINDOW) -> float:
    """Annualized standard deviation of daily log returns."""
    logret = np.log(closes / closes.shift(1)).dropna()
    if len(logret) < window:
        return float("nan")
    return float(logret.tail(window).std(ddof=1) * math.sqrt(252.0))


def build_feed(debug=False) -> dict:
    tk = yf.Ticker(TICKER)

    history = tk.history(period="6mo", interval="1d")
    if history.empty:
        raise RuntimeError(f"no price history returned for {TICKER}")
    spot = float(history["Close"].iloc[-1])
    rv = realized_volatility(history["Close"])

    chosen = pick_expiry(tk)
    if chosen is None:
        raise RuntimeError("no listed expiry inside the accepted window")
    expiry, days = chosen
    t_years = days / 365.0
    if debug:
        print(f"  spot {spot:.2f}, expiry {expiry} ({days} days)")

    chain = tk.option_chain(expiry)
    points = collect_smile(chain, spot, t_years, debug)
    fit = fit_smile(points, spot)
    if fit is None:
        raise RuntimeError("not enough usable contracts to fit the smile")

    strikes, dens, cdf = density_from_smile(fit, spot, t_years, RISK_FREE)
    if strikes is None:
        raise RuntimeError("the recovered density had no usable mass")

    levels = {q: percentile(strikes, cdf, q) for q in (0.05, 0.25, 0.50, 0.75, 0.95)}
    step = strikes[1] - strikes[0]
    mean = float(np.sum(strikes * dens) * step)
    var = float(np.sum(((strikes - mean) ** 2) * dens) * step)
    sd = math.sqrt(max(var, 0.0))
    skew = float(np.sum(((strikes - mean) ** 3) * dens) * step) / (sd ** 3) if sd > 0 else 0.0
    kurt = float(np.sum(((strikes - mean) ** 4) * dens) * step) / (sd ** 4) if sd > 0 else 0.0
    prob_above = float(1.0 - np.interp(spot, strikes, cdf))

    atm_vol = smooth_vol(fit, spot, spot)
    bs_move = spot * atm_vol * math.sqrt(t_years)

    #--- benchmark: the same recovery run on a flat smile at the at-the-money
    #--- volatility. That is a lognormal, and it is the honest reference point.
    #--- In price terms a lognormal is already right-skewed, so the market's
    #--- skewness has to be read against this number rather than against zero.
    flat = {"coeffs": np.array([atm_vol]), "xlo": -1.0, "xhi": 1.0}
    b_strikes, b_dens, _ = density_from_smile(flat, spot, t_years, RISK_FREE)
    b_step = b_strikes[1] - b_strikes[0]
    b_mean = float(np.sum(b_strikes * b_dens) * b_step)
    b_sd = math.sqrt(max(float(np.sum(((b_strikes - b_mean) ** 2) * b_dens) * b_step), 0.0))
    b_skew = float(np.sum(((b_strikes - b_mean) ** 3) * b_dens) * b_step) / (b_sd ** 3)
    b_kurt = float(np.sum(((b_strikes - b_mean) ** 4) * b_dens) * b_step) / (b_sd ** 4)

    if debug:
        print(f"  density integrates to {float(np.sum(dens) * step):.4f}")
        print(f"  median {levels[0.50]:.2f} against spot {spot:.2f}")
        print(f"  one standard deviation {sd:.2f}, Black-Scholes says {bs_move:.2f}")
        print(f"  skewness {skew:+.3f}, kurtosis {kurt:.3f}")
        print(f"  lognormal benchmark: skewness {b_skew:+.3f}, kurtosis {b_kurt:.3f}")
        print(f"  90 percent band {levels[0.05]:.2f} to {levels[0.95]:.2f}")
        for m in (0.85, 0.95, 1.00, 1.05, 1.15):
            print(f"    fitted volatility at {m * 100:5.0f}% of spot: "
                  f"{100.0 * smooth_vol(fit, spot, spot * m):5.2f}%")

    return {
        "symbol": TICKER,
        "spot": round(spot, 4),
        "expiry": expiry,
        "days_to_expiry": days,
        "atm_iv": round(atm_vol, 6),
        "rv_30d_gld": None if math.isnan(rv) else round(rv, 6),
        "r05": round(levels[0.05] / spot, 6),
        "r25": round(levels[0.25] / spot, 6),
        "r50": round(levels[0.50] / spot, 6),
        "r75": round(levels[0.75] / spot, 6),
        "r95": round(levels[0.95] / spot, 6),
        "prob_above_spot": round(prob_above, 6),
        "implied_move_1sd_pct": round(100.0 * sd / spot, 4),
        "bs_move_1sd_pct": round(100.0 * bs_move / spot, 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "bench_skewness": round(b_skew, 4),
        "bench_kurtosis": round(b_kurt, 4),
        "density_area": round(float(np.sum(dens) * step), 6),
        "contracts_used": len(points),
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "option chain, risk-neutral density by Breeden-Litzenberger",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the option-implied distribution as JSON")
    parser.add_argument("--out", help="also write the JSON to this file")
    parser.add_argument("--debug", action="store_true", help="show the fit and the checks")
    args = parser.parse_args()

    if args.debug:
        print("recovering the risk-neutral density from market prices:")
    feed = build_feed(args.debug)
    text = json.dumps(feed, indent=2)
    print(("\n" if args.debug else "") + text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"\nwritten to {args.out}")

    print(f"\nexpiry {feed['expiry']} in {feed['days_to_expiry']} days, "
          f"{feed['contracts_used']} contracts used")
    print(f"90 percent band: {feed['r05'] * 100:.1f}% to {feed['r95'] * 100:.1f}% of spot")
    print(f"implied one standard deviation move: {feed['implied_move_1sd_pct']:.2f}%")
    print(f"probability of finishing above spot: {feed['prob_above_spot'] * 100:.1f}%")
    print(f"skewness {feed['skewness']:+.3f} against a lognormal benchmark of "
          f"{feed['bench_skewness']:+.3f}")
    print(f"kurtosis {feed['kurtosis']:.2f} against a lognormal benchmark of "
          f"{feed['bench_kurtosis']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
