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


def nearest_shorter_preset(seconds):
    """Largest preset that is <= seconds. Falls back to the shortest preset
    if the request is below everything we can do."""
    candidates = [(sec, name) for sec, name in _PRESETS if sec <= seconds + 1e-9]
    if candidates:
        return candidates[-1]
    return _PRESETS[0]


def main():
    parser = argparse.ArgumentParser(
        description='Capture a sequence of exposures on the Nikon Z 6. '
                    'Exposures <= 30 s use the nearest shorter hardware-timed '
                    'preset; longer exposures use Bulb mode.')
    parser.add_argument('exposure', type=float,
                        help='Exposure time in seconds (e.g. 20, 30, 120, 180).')
    parser.add_argument('repeats', type=int,
                        help='Number of shots to take.')
    parser.add_argument('download', type=int, choices=(0, 1),
                        help='1 = download each finished image to the current '
                             'folder (kept on the card too); 0 = do not download.')
    args = parser.parse_args()

    if args.exposure <= 0:
        parser.error('exposure must be positive')
    if args.repeats <= 0:
        parser.error('repeats must be positive')

    use_bulb = args.exposure > MAX_TIMED_SECONDS

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

        if use_bulb:
            set_config('shutterspeed', 'Bulb')
            print(f"\nBulb mode: {args.exposure:.3f}s per shot, "
                  f"{args.repeats} shots")
        else:
            sec, name = nearest_shorter_preset(args.exposure)
            if abs(sec - args.exposure) > 1e-6:
                print(f"Requested {args.exposure:.4f}s -> using nearest shorter "
                      f"preset {name} ({sec:.4f}s)")
            set_config('shutterspeed', name)
            print(f"\nTimed mode: {name} per shot, {args.repeats} shots")

        for i in range(args.repeats):
            print(f"\n--- Shot {i + 1}/{args.repeats} ---")
            t0 = time.time()

            if use_bulb:
                # Drive the shutter ourselves: open, hold, close, collect.
                set_config('bulb', 1)
                time.sleep(args.exposure)
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
