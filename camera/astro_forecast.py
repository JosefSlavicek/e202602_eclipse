#!/usr/bin/env python3
"""Night forecast check for M31 astrophotography.

For each place and each dark hour of the night it prints:
  - low / mid / high cloud cover, rain, wind, gusts, relative humidity,
    temperature minus dew point, wind at 250 hPa (jet stream),
    from several weather models (Open-Meteo API, free, no key),
  - seeing and transparency (7Timer! ASTRO API, GFS based, 3-hourly, ~3 days),
  - Sun altitude, Moon altitude / illumination / distance from M31, M31 altitude
    (computed locally, accuracy about 0.5 degree, good enough for planning).

Values outside the checklist limits are marked with "!" (and red on a terminal).
At the end of each night it prints how many hours pass all weather checks per model.

Times are Europe/Prague local time.

Examples:
  python3 astro_forecast.py                      # tonight, default places
  python3 astro_forecast.py --nights 3
  python3 astro_forecast.py --date 2026-09-20 --step 2
  python3 astro_forecast.py --point 50.0832,17.2309,1491,Praded --point 50.2,17.1

Needs: Python 3.6+, requests.
"""

import argparse
import calendar
import sys
import time
from datetime import date, datetime, timedelta
from math import acos, asin, atan2, cos, degrees, radians, sin, tan

import requests

# name, latitude, longitude, elevation [m]
PLACES = [
    ("Praděd", 50.0832, 17.2309, 1491),
    ("Červenohorské sedlo", 50.1250, 17.1547, 1013),
    ("Dlouhé stráně - horní nádrž", 50.0750, 17.1594, 1350),
]

# Open-Meteo model id, short label
MODELS = [
    ("ecmwf_ifs", "ECMWF"),
    ("icon_d2", "ICON-D2"),
    ("icon_eu", "ICON-EU"),
    ("meteofrance_seamless", "MeteoFr"),
    ("gfs_seamless", "GFS"),
]

# Variables a model does not provide are taken from this model.
# ECMWF 9 km has no pressure level data (jet stream), the 25 km version has.
FALLBACK_MODELS = {"ecmwf_ifs": "ecmwf_ifs025"}

# Checklist limits
MAX_CLOUD = 10          # % for each layer
MAX_RAIN = 0.0          # mm per hour
MAX_WIND = 15           # km/h
MAX_GUST = 25           # km/h
MAX_RH = 80             # %
MIN_DEW_SPREAD = 3      # °C, temperature minus dew point
MAX_JET = 20            # m/s at 250 hPa
MAX_SEEING_CLASS = 6    # 7Timer class 6 = 1.5-2 arcsec
MAX_TRANSP_CLASS = 3    # 7Timer class 3 = 0.4-0.5 mag/airmass
MAX_MOON_ILLUM = 10     # %, marked only when the Moon is above the horizon
MIN_M31_ALT = 30        # degrees

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
SEVEN_TIMER_URL = "http://www.7timer.info/bin/astro.php"

HOURLY = [
    "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "precipitation",
    "wind_speed_10m", "wind_gusts_10m", "relative_humidity_2m",
    "temperature_2m", "dew_point_2m", "wind_speed_250hPa",
]

SEEING_ARCSEC = {1: "<0.5", 2: "0.5-0.75", 3: "0.75-1", 4: "1-1.25",
                 5: "1.25-1.5", 6: "1.5-2", 7: "2-2.5", 8: ">2.5"}
TRANSP_MAG = {1: "<0.3", 2: "0.3-0.4", 3: "0.4-0.5", 4: "0.5-0.6",
              5: "0.6-0.7", 6: "0.7-0.85", 7: "0.85-1", 8: ">1"}


def _dew_spread(r):
    if r["temperature_2m"] is None or r["dew_point_2m"] is None:
        return None
    return r["temperature_2m"] - r["dew_point_2m"]


def _jet(r):
    v = r["wind_speed_250hPa"]
    return None if v is None else v / 3.6


# header, unit, width, format, value from a model row, "bad" test
CHECKS = [
    ("low", "%", 5, "{:.0f}", lambda r: r["cloud_cover_low"], lambda v: v > MAX_CLOUD),
    ("mid", "%", 5, "{:.0f}", lambda r: r["cloud_cover_mid"], lambda v: v > MAX_CLOUD),
    ("high", "%", 5, "{:.0f}", lambda r: r["cloud_cover_high"], lambda v: v > MAX_CLOUD),
    ("rain", "mm", 6, "{:.1f}", lambda r: r["precipitation"], lambda v: v > MAX_RAIN),
    ("wind", "km/h", 6, "{:.0f}", lambda r: r["wind_speed_10m"], lambda v: v > MAX_WIND),
    ("gust", "km/h", 6, "{:.0f}", lambda r: r["wind_gusts_10m"], lambda v: v > MAX_GUST),
    ("RH", "%", 5, "{:.0f}", lambda r: r["relative_humidity_2m"], lambda v: v > MAX_RH),
    ("T-Td", "°C", 6, "{:.1f}", _dew_spread, lambda v: v < MIN_DEW_SPREAD),
    ("jet", "m/s", 5, "{:.0f}", _jet, lambda v: v > MAX_JET),
]

USE_COLOR = False


def mark(text, bad):
    """Append '!' to a bad value and color it red on a terminal."""
    if not bad:
        return text
    text += "!"
    return "\033[31m" + text + "\033[0m" if USE_COLOR else text


def cell(text, bad, width):
    # pad before coloring, so escape codes do not break the alignment
    padded = (text + ("!" if bad else " ")).rjust(width)
    if bad and USE_COLOR:
        return "\033[31m" + padded + "\033[0m"
    return padded


# ---------------------------------------------------------------- time

def prague_offset_hours(ts):
    """UTC offset of Europe/Prague (EU rule: CEST from last Sunday of March
    01:00 UTC to last Sunday of October 01:00 UTC)."""
    utc = datetime.utcfromtimestamp(ts)

    def last_sunday_1utc(month):
        d = datetime(utc.year, month, 31, 1)
        return d - timedelta(days=(d.weekday() + 1) % 7)

    return 2 if last_sunday_1utc(3) <= utc < last_sunday_1utc(10) else 1


def zone_name(ts):
    return "CEST" if prague_offset_hours(ts) == 2 else "CET"


def local_dt(ts):
    return datetime.utcfromtimestamp(ts + prague_offset_hours(ts) * 3600)


def local_noon_ts(d):
    guess = calendar.timegm(datetime(d.year, d.month, d.day, 12).timetuple())
    return guess - prague_offset_hours(guess) * 3600


# ---------------------------------------------------------------- astronomy

def sind(x):
    return sin(radians(x))


def cosd(x):
    return cos(radians(x))


def tand(x):
    return tan(radians(x))


def _days_since_j2000(ts):
    return ts / 86400.0 + 2440587.5 - 2451545.0


def _ecl_to_eq(lam, beta, eps):
    ra = degrees(atan2(sind(lam) * cosd(eps) - tand(beta) * sind(eps), cosd(lam)))
    dec = degrees(asin(sind(beta) * cosd(eps) + cosd(beta) * sind(eps) * sind(lam)))
    return ra % 360, dec


def _eq_to_ecl(ra, dec, eps):
    lam = degrees(atan2(sind(ra) * cosd(eps) + tand(dec) * sind(eps), cosd(ra)))
    beta = degrees(asin(sind(dec) * cosd(eps) - cosd(dec) * sind(eps) * sind(ra)))
    return lam % 360, beta


def _separation(ra1, dec1, ra2, dec2):
    c = sind(dec1) * sind(dec2) + cosd(dec1) * cosd(dec2) * cosd(ra1 - ra2)
    return degrees(acos(max(-1.0, min(1.0, c))))


def _altitude(ra, dec, ts, lat, lon):
    lst = 280.46061837 + 360.98564736629 * _days_since_j2000(ts) + lon
    ha = lst - ra
    return degrees(asin(sind(lat) * sind(dec) + cosd(lat) * cosd(dec) * cosd(ha)))


def sun_radec(ts):
    d = _days_since_j2000(ts)
    L = 280.460 + 0.9856474 * d
    g = 357.528 + 0.9856003 * d
    lam = L + 1.915 * sind(g) + 0.020 * sind(2 * g)
    eps = 23.439 - 0.0000004 * d
    return _ecl_to_eq(lam, 0.0, eps)


def sun_altitude(ts, lat, lon):
    ra, dec = sun_radec(ts)
    return _altitude(ra, dec, ts, lat, lon)


def sky(ts, lat, lon):
    """Sun altitude, Moon altitude / illumination / distance from M31, M31 altitude.
    Low precision formulas from the Astronomical Almanac."""
    d = _days_since_j2000(ts)
    T = d / 36525.0
    eps = 23.439 - 0.0000004 * d

    lam = (218.32 + 481267.881 * T
           + 6.29 * sind(135.0 + 477198.87 * T) - 1.27 * sind(259.3 - 413335.36 * T)
           + 0.66 * sind(235.7 + 890534.22 * T) + 0.21 * sind(269.9 + 954397.74 * T)
           - 0.19 * sind(357.5 + 35999.05 * T) - 0.11 * sind(186.5 + 966404.03 * T))
    beta = (5.13 * sind(93.3 + 483202.02 * T) + 0.28 * sind(228.2 + 960400.89 * T)
            - 0.28 * sind(318.3 + 6003.15 * T) - 0.17 * sind(217.6 - 407332.21 * T))
    parallax = (0.9508 + 0.0518 * cosd(135.0 + 477198.87 * T)
                + 0.0095 * cosd(259.3 - 413335.36 * T) + 0.0078 * cosd(235.7 + 890534.22 * T)
                + 0.0028 * cosd(269.9 + 954397.74 * T))
    moon_ra, moon_dec = _ecl_to_eq(lam, beta, eps)
    moon_alt = _altitude(moon_ra, moon_dec, ts, lat, lon)
    moon_alt -= parallax * cosd(moon_alt)  # geocentric -> topocentric

    sun_ra, sun_dec = sun_radec(ts)
    elongation = _separation(sun_ra, sun_dec, moon_ra, moon_dec)
    illum = (1 - cosd(elongation)) / 2 * 100

    # M31 at J2000, precessed to the equinox of date
    m31_lam, m31_beta = _eq_to_ecl(10.6847, 41.2691, 23.4393)
    m31_lam += 50.29 / 3600 * d / 365.25
    m31_ra, m31_dec = _ecl_to_eq(m31_lam, m31_beta, eps)

    return {
        "sun_alt": _altitude(sun_ra, sun_dec, ts, lat, lon),
        "moon_alt": moon_alt,
        "moon_illum": illum,
        "moon_m31": _separation(moon_ra, moon_dec, m31_ra, m31_dec),
        "m31_alt": _altitude(m31_ra, m31_dec, ts, lat, lon),
    }


# ---------------------------------------------------------------- data sources

def fetch_open_meteo(lat, lon, elev, days):
    """Returns {model: {ts: {variable: value}}}."""
    models = [m for m, _ in MODELS]
    models += [f for m, f in FALLBACK_MODELS.items() if m in models and f not in models]
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(HOURLY),
        "models": ",".join(models),
        "timeformat": "unixtime", "timezone": "GMT",
        "forecast_days": days,
    }
    if elev is not None:
        params["elevation"] = elev
    r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    times = hourly["time"]
    empty = [None] * len(times)

    def get_series(model, var):
        # with a single model Open-Meteo leaves out the model suffix
        key = var + "_" + model if len(models) > 1 else var
        return hourly.get(key, empty)

    data = {}
    for model, _ in MODELS:
        series = {}
        for var in HOURLY:
            values = get_series(model, var)
            if all(v is None for v in values) and model in FALLBACK_MODELS:
                values = get_series(FALLBACK_MODELS[model], var)
            series[var] = values
        data[model] = {ts: {var: series[var][i] for var in HOURLY}
                       for i, ts in enumerate(times)}
    return data


def fetch_7timer(lat, lon):
    """Returns {ts: (seeing class, transparency class)}, 3-hourly."""
    params = {"lon": round(lon, 3), "lat": round(lat, 3), "ac": 0,
              "unit": "metric", "output": "json", "tzshift": 0}
    r = requests.get(SEVEN_TIMER_URL, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    init = calendar.timegm(datetime.strptime(j["init"], "%Y%m%d%H").timetuple())
    return {init + p["timepoint"] * 3600: (p.get("seeing"), p.get("transparency"))
            for p in j["dataseries"]}


def nearest(series, ts, max_diff=5400):
    if not series:
        return None
    t = min(series, key=lambda x: abs(x - ts))
    return series[t] if abs(t - ts) <= max_diff else None


# ---------------------------------------------------------------- output

def fmt_sky(s, seven):
    parts = ["Sun {:.0f}°".format(s["sun_alt"])]

    if s["moon_alt"] <= 0:
        parts.append("Moon below horizon ({:.0f} % lit)".format(s["moon_illum"]))
    else:
        bad = s["moon_illum"] > MAX_MOON_ILLUM
        parts.append(mark("Moon {:.0f}° ({:.0f} % lit, {:.0f}° from M31)".format(
            s["moon_alt"], s["moon_illum"], s["moon_m31"]), bad))

    parts.append(mark("M31 {:.0f}°".format(s["m31_alt"]), s["m31_alt"] < MIN_M31_ALT))

    seeing, transp = seven if seven else (None, None)
    if seeing in SEEING_ARCSEC:
        parts.append(mark('seeing {}" (7T {})'.format(SEEING_ARCSEC[seeing], seeing),
                          seeing > MAX_SEEING_CLASS))
    else:
        parts.append("seeing -")
    if transp in TRANSP_MAG:
        parts.append(mark("transp. {} mag (7T {})".format(TRANSP_MAG[transp], transp),
                          transp > MAX_TRANSP_CLASS))
    else:
        parts.append("transp. -")
    return "   ".join(parts)


def dark_window(start_ts, end_ts, lat, lon, sun_limit):
    dark = [t for t in range(start_ts, end_ts, 300) if sun_altitude(t, lat, lon) < sun_limit]
    return (dark[0], dark[-1]) if dark else None


def print_night(name, lat, lon, elev, night_date, weather, seven, args):
    start_ts = local_noon_ts(night_date)
    end_ts = local_noon_ts(night_date + timedelta(days=1))

    window = dark_window(start_ts, end_ts, lat, lon, args.sun_limit)
    title = "{}  ({:.4f}, {:.4f}, {} m)   night {} -> {}".format(
        name, lat, lon, elev if elev is not None else "?", night_date, night_date + timedelta(days=1))
    print("=" * len(title))
    print(title)
    if window is None:
        print("  Sun never gets below {}° this night. Try --sun-limit -12.\n".format(args.sun_limit))
        return
    # the night of the October switch starts in CEST and ends in CET
    zones = [zone_name(window[0]), zone_name(window[1])]
    zone = zones[0] if zones[0] == zones[1] else "/".join(zones)
    print("  Sun below {}°: {} - {}   (times in {})".format(
        args.sun_limit, local_dt(window[0]).strftime("%H:%M"),
        local_dt(window[1]).strftime("%H:%M"), zone))

    any_model = weather[MODELS[0][0]]
    hours = [ts for ts in sorted(any_model)
             if start_ts <= ts < end_ts and sun_altitude(ts, lat, lon) < args.sun_limit]
    hours = hours[::args.step]
    if not hours:
        print("  No full hour inside the dark window.\n")
        return

    print()
    print("       {:<9}".format("") + "".join((h + " ").rjust(w) for h, _, w, _, _, _ in CHECKS))
    print("{:<7}{:<9}".format(zone if len(zone) < 7 else "time", "model") + "".join((u + " ").rjust(w) for _, u, w, _, _, _ in CHECKS))

    passed = {m: 0 for m, _ in MODELS}
    with_data = {m: 0 for m, _ in MODELS}

    for ts in hours:
        for i, (model, label) in enumerate(MODELS):
            row = weather[model].get(ts)
            cells = []
            ok, complete = True, True
            for _, _, width, fmt, get, is_bad in CHECKS:
                v = get(row) if row else None
                if v is None:
                    complete = False
                    cells.append(cell("-", False, width))
                else:
                    bad = is_bad(v)
                    ok = ok and not bad
                    cells.append(cell(fmt.format(v), bad, width))
            if complete:
                with_data[model] += 1
                passed[model] += ok
            time_col = local_dt(ts).strftime("%H:%M") if i == 0 else ""
            print("{:<7}{:<9}".format(time_col, label) + "".join(cells))
        print("       " + fmt_sky(sky(ts, lat, lon), nearest(seven, ts)))
        print()

    summary = []
    for model, label in MODELS:
        if with_data[model]:
            summary.append("{} {}/{}".format(label, passed[model], with_data[model]))
        else:
            summary.append("{} no data".format(label))
    print("  Hours passing all weather checks:  " + "   ".join(summary))
    print()


def parse_point(text):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("expected lat,lon[,elevation[,name]]")
    lat, lon = float(parts[0]), float(parts[1])
    elev = float(parts[2]) if len(parts) > 2 and parts[2] else None
    name = parts[3] if len(parts) > 3 else "{:.4f}, {:.4f}".format(lat, lon)
    return name, lat, lon, elev


def main():
    global USE_COLOR

    ap = argparse.ArgumentParser(description="Night forecast check for M31 astrophotography.")
    ap.add_argument("--date", help="local date of the evening, YYYY-MM-DD (default: tonight)")
    ap.add_argument("--nights", type=int, default=1, help="number of nights (default 1)")
    ap.add_argument("--step", type=int, default=1, help="print every N-th hour (default 1)")
    ap.add_argument("--point", type=parse_point, action="append",
                    help="lat,lon[,elevation[,name]]; can repeat; replaces the default places")
    ap.add_argument("--sun-limit", type=float, default=-18,
                    help="Sun altitude that counts as dark (default -18, astronomical night)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    USE_COLOR = sys.stdout.isatty() and not args.no_color

    now_local = local_dt(time.time())
    if args.date:
        first_night = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        # after midnight and before 6:00, "tonight" is the night that is running now
        first_night = (now_local - timedelta(hours=6)).date()
    if first_night < now_local.date() - timedelta(days=1):
        ap.error("date is in the past")

    today_utc = datetime.utcnow().date()
    days = (first_night + timedelta(days=args.nights) - today_utc).days + 1
    if days > 16:
        ap.error("Open-Meteo forecast goes only 16 days ahead")
    days = max(days, 1)

    places = args.point or PLACES
    print("Limits: clouds <= {}% per layer, rain {} mm, wind <= {} km/h, gusts <= {} km/h, "
          "RH <= {}%, T-Td >= {} °C, jet <= {} m/s,".format(
              MAX_CLOUD, MAX_RAIN, MAX_WIND, MAX_GUST, MAX_RH, MIN_DEW_SPREAD, MAX_JET))
    print('        seeing <= {}" (7T {}), transparency <= {} mag (7T {}), '
          "Moon up and > {}% lit, M31 >= {}°".format(
              SEEING_ARCSEC[MAX_SEEING_CLASS].split("-")[-1], MAX_SEEING_CLASS,
              TRANSP_MAG[MAX_TRANSP_CLASS].split("-")[-1], MAX_TRANSP_CLASS,
              MAX_MOON_ILLUM, MIN_M31_ALT))
    print("Times are Prague local time (CEST = UTC+2 in summer, CET = UTC+1 in winter). ICON-D2 covers ~2 days, 7Timer ~3 days. "
          "Model wind is for a smoothed terrain, real wind on summits is usually stronger.")
    print()

    for name, lat, lon, elev in places:
        try:
            weather = fetch_open_meteo(lat, lon, elev, days)
        except (requests.RequestException, ValueError, KeyError) as e:
            print("{}: Open-Meteo request failed: {}\n".format(name, e))
            continue
        try:
            seven = fetch_7timer(lat, lon)
        except (requests.RequestException, ValueError, KeyError) as e:
            print("{}: 7Timer request failed ({}), seeing/transparency will be missing".format(name, e))
            seven = {}

        for n in range(args.nights):
            print_night(name, lat, lon, elev, first_night + timedelta(days=n), weather, seven, args)


if __name__ == "__main__":
    main()
