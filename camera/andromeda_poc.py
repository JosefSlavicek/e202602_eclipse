import time

import gphoto2 as gp

SHUTTER_SPEED = '30.0000s'
N_SHOTS = 10


def main():
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

    try:
        set_config('controlmode', '0')
        set_config('bracketing', 'Off')
        set_config('longexpnr', 'Off')
        set_config('capturetarget', 'Memory card')
        set_config('capturemode', 'Single Shot')
        set_config('shutterspeed', SHUTTER_SPEED)

        for i in range(N_SHOTS):
            print(f"\n--- Shot {i + 1}/{N_SHOTS} ---")
            t0 = time.time()

            err, camera_file_path = gp.gp_camera_capture(camera, gp.GP_CAPTURE_IMAGE, context)
            assert err == gp.GP_OK, err
            print(f"Captured in {time.time() - t0:.2f}s: "
                  f"{camera_file_path.folder}/{camera_file_path.name}")

    finally:
        gp.gp_camera_exit(camera, context)


if __name__ == '__main__':
    main()
