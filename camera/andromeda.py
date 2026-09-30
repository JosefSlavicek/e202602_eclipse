import argparse
import time

import gphoto2 as gp

# Timed shutter-speed presets exposed by the Nikon Z 6, copied from
# camera_config_20260426.txt (/main/capturesettings/shutterspeed, choices 0..52).
# Kept embedded so this script is standalone and does not read the text file.
# These are the only exact, hardware-timed exposures the camera offers; the
# longest is 30 s. Anything longer must be done in Bulb mode.
SHUTTERSPEED_PRESETS = [
    '0.0001s', '0.0002s', '0.0003s', '0.0004s', '0.0005s', '0.0006s', '0.0008s',
    '0.0010s', '0.0012s', '0.0015s', '0.0020s', '0.0025s', '0.0031s', '0.0040s',
    '0.0050s', '0.0062s', '0.0080s', '0.0100s', '0.0125s', '0.0166s', '0.0200s',
    '0.0250s', '0.0333s', '0.0400s', '0.0500s', '0.0666s', '0.0769s', '0.1000s',
    '0.1250s', '0.1666s', '0.2000s', '0.2500s', '0.3333s', '0.4000s', '0.5000s',
    '0.6250s', '0.7692s', '1.0000s', '1.3000s', '1.6000s', '2.0000s', '2.5000s',
    '3.0000s', '4.0000s', '5.0000s', '6.0000s', '8.0000s', '10.0000s', '13.0000s',
    '15.0000s', '20.0000s', '25.0000s', '30.0000s',
]

# (seconds, choice string), sorted ascending by seconds.
_PRESETS = sorted((float(s[:-1]), s) for s in SHUTTERSPEED_PRESETS)
MAX_TIMED_SECONDS = _PRESETS[-1][0]  # 30.0


# Exposure ladder: after every LADDER_EVERY main frames (except after the last
# one) shoot a run of shorter exposures, each LADDER_FACTOR x shorter than the
# one before, always ending at exactly LADDER_FLOOR seconds.
LADDER_EVERY = 20
LADDER_FACTOR = 10.0
LADDER_FLOOR = 0.01


def nearest_shorter_preset(seconds):
    """Largest preset that is <= seconds. Falls back to the shortest preset
    if the request is below everything we can do."""
    candidates = [(sec, name) for sec, name in _PRESETS if sec <= seconds + 1e-9]
    if candidates:
        return candidates[-1]
    return _PRESETS[0]


def shutter_setting(seconds):
    """The camera 'shutterspeed' value a request of `seconds` really uses."""
    if seconds > MAX_TIMED_SECONDS:
        return 'Bulb'
    return nearest_shorter_preset(seconds)[1]


def ladder(main_exposure):
    """Exposure times of one ladder after main frames of `main_exposure`.

    main/F, main/F^2, ... while longer than the floor, then the floor itself.
    A step that would give the same camera setting as the shot before it
    (including the main exposure) is dropped. Empty if the main exposure is
    already at or below the floor.
    """
    if main_exposure <= LADDER_FLOOR:
        return []
    times = []
    t = main_exposure / LADDER_FACTOR
    while t > LADDER_FLOOR * (1 + 1e-9):
        times.append(t)
        t /= LADDER_FACTOR
    times.append(LADDER_FLOOR)

    out, prev = [], shutter_setting(main_exposure)
    for t in times:
        setting = shutter_setting(t)
        if setting == 'Bulb' or setting != prev:
            out.append(t)
        prev = setting
    return out


def shot_plan(main_exposure, repeats):
    """[(seconds, label)] for the whole run: main frames with ladders between
    each block of LADDER_EVERY, none after the last main frame."""
    plan = []
    steps = ladder(main_exposure)
    for i in range(1, repeats + 1):
        plan.append((main_exposure, f'main {i}/{repeats}'))
        if i % LADDER_EVERY == 0 and i < repeats:
            plan += [(t, f'ladder {k}/{len(steps)}')
                     for k, t in enumerate(steps, 1)]
    return plan


def describe(seconds):
    """How a request is carried out, for printing."""
    setting = shutter_setting(seconds)
    if setting == 'Bulb':
        return f'Bulb {seconds:.3f}s'
    if abs(nearest_shorter_preset(seconds)[0] - seconds) > 1e-6:
        return f'{seconds:.4g}s -> preset {setting}'
    return f'preset {setting}'


def main():
    parser = argparse.ArgumentParser(
        description='Capture a sequence of exposures on the Nikon Z 6. '
                    'Exposures <= 30 s use the nearest shorter hardware-timed '
                    'preset; longer exposures use Bulb mode. After every '
                    f'{LADDER_EVERY} main frames (except after the last) a '
                    f'ladder of exposures {LADDER_FACTOR:g}x shorter each, '
                    f'down to {LADDER_FLOOR:g}s, is shot.')
    parser.add_argument('exposure', type=float,
                        help='Exposure time in seconds (e.g. 20, 30, 120, 180).')
    parser.add_argument('repeats', type=int,
                        help='Number of main-exposure shots to take (ladder '
                             'shots come on top).')
    parser.add_argument('download', type=int, choices=(0, 1),
                        help='1 = download each finished image to the current '
                             'folder (kept on the card too); 0 = do not download.')
    args = parser.parse_args()

    if args.exposure <= 0:
        parser.error('exposure must be positive')
    if args.repeats <= 0:
        parser.error('repeats must be positive')

    plan = shot_plan(args.exposure, args.repeats)

    err, camera = gp.gp_camera_new()
    assert err == gp.GP_OK, err
    context = gp.gp_context_new()
    err = gp.gp_camera_init(camera, context)
    assert err == gp.GP_OK, err

    err, config = gp.gp_camera_get_config(camera, context)
    assert err == gp.GP_OK, err

    def set_config(name, value):
        err, child = gp.gp_widget_get_child_by_name(config, name)
        assert err == gp.GP_OK, (err, name)
        err = gp.gp_widget_set_value(child, value)
        assert err == gp.GP_OK, (err, name, value)
        err = gp.gp_camera_set_config(camera, config, context)
        assert err == gp.GP_OK, (err, name, value)
        print(f"Set {name} = {value}")

    def drain_until_file(timeout_ms=10000):
        """Wait for the camera to report the captured file is written.
        Returns (folder, name) or None on timeout."""
        waited = 0
        while waited < timeout_ms:
            err, ev_type, ev_data = gp.gp_camera_wait_for_event(camera, 200, context)
            assert err == gp.GP_OK, err
            waited += 200
            if ev_type == gp.GP_EVENT_FILE_ADDED:
                print(f"File saved: {ev_data.folder}/{ev_data.name}")
                return ev_data.folder, ev_data.name
        print("Warning: timed out waiting for file-added event")
        return None

    def download(folder, name):
        """Copy the image off the card to the current folder; card keeps it."""
        err, cam_file = gp.gp_file_new()
        assert err == gp.GP_OK, err
        err = gp.gp_camera_file_get(camera, folder, name,
                                    gp.GP_FILE_TYPE_NORMAL, cam_file, context)
        assert err == gp.GP_OK, (err, folder, name)
        err = gp.gp_file_save(cam_file, name)
        assert err == gp.GP_OK, (err, name)
        print(f"Downloaded: {name}")

    try:
        set_config('controlmode', '0')
        set_config('bracketing', 'Off')
        set_config('longexpnr', 'Off')
        set_config('capturetarget', 'Memory card')
        set_config('capturemode', 'Single Shot')

        steps = ladder(args.exposure)
        print(f"\nMain exposure: {describe(args.exposure)}, "
              f"{args.repeats} shots")
        if steps:
            print(f"Ladder after every {LADDER_EVERY} main shots: "
                  + ', '.join(describe(t) for t in steps))
        n_ladders = (len(plan) - args.repeats) // len(steps) if steps else 0
        print(f"{len(plan)} shots in total ({n_ladders} ladders)")

        current = None            # 'shutterspeed' now set, to skip re-setting
        for n, (seconds, label) in enumerate(plan, 1):
            print(f"\n--- Shot {n}/{len(plan)}: {label}, {describe(seconds)} ---")
            t0 = time.time()

            setting = shutter_setting(seconds)
            if setting != current:
                set_config('shutterspeed', setting)
                current = setting

            if setting == 'Bulb':
                # Drive the shutter ourselves: open, hold, close, collect.
                set_config('bulb', 1)
                time.sleep(seconds)
                set_config('bulb', 0)
                location = drain_until_file()
            else:
                err, path = gp.gp_camera_capture(camera, gp.GP_CAPTURE_IMAGE, context)
                assert err == gp.GP_OK, err
                print(f"Captured: {path.folder}/{path.name}")
                location = (path.folder, path.name)

            if args.download and location is not None:
                download(*location)

            print(f"Shot took {time.time() - t0:.2f}s")

    finally:
        gp.gp_camera_exit(camera, context)


if __name__ == '__main__':
    main()
