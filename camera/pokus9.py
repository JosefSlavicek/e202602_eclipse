import gphoto2 as gp
import time

last_vals = dict()
t0 = []

def unlock_and_burst():
    error, camera = gp.gp_camera_new()
    context = gp.gp_context_new()
    gp.gp_camera_init(camera, context)
    error, config = gp.gp_camera_get_config(camera, context)
    assert error == gp.GP_OK

    def set_config(name, value):
        if name in last_vals and last_vals[name] == value:
            print(f"{name} already with value {value}")
            return True
        error, child = gp.gp_widget_get_child_by_name(config, name)
        assert error == gp.GP_OK, (error, name)
        gp.gp_widget_set_value(child, value)
        gp.gp_camera_set_config(camera, config, context)
        last_vals[name] = value
        print(f"Set {name} to {value}")
        return True

    def do_burst(shutterspeed, aebracketingstep, aebracketingpattern, repeat_style, wait_for_user):
        set_config('shutterspeed', shutterspeed)
        set_config('aebracketingstep', aebracketingstep)
        set_config('bracketing', 'On')
        set_config('aebracketingpattern', aebracketingpattern)

        # 5. Finalize setup
        set_config('capturetarget', 'Memory card')
        set_config('capturemode', 'Burst')

        if repeat_style == 'no_repeat':
            bracket_count0 = int(aebracketingpattern[0])
            n_iter = 1
        elif repeat_style == 'double_in_two_runs':
            bracket_count0 = int(aebracketingpattern[0])
            n_iter = 2
        elif repeat_style == 'double_in_single_run':
            bracket_count0 = 2 * int(aebracketingpattern[0])
            n_iter = 1
        else:
            raise ValueError(repeat_style)
        if wait_for_user:
            input('Pres enter to continue')
            if len(t0) <= 0:
                t0.append(time.time())
        for _ in range(n_iter):
            bracket_count = bracket_count0
            print(f"Firing {bracket_count} shots...")
            for i in range(bracket_count):
                gp.gp_camera_trigger_capture(camera, context)
            print("Waiting for files to write to card...")
            while bracket_count > 0:
                err, ev_type, ev_data = gp.gp_camera_wait_for_event(camera, 100, context)
                if ev_type == gp.GP_EVENT_FILE_ADDED:
                    print(f"File saved: {ev_data.name}")
                    bracket_count -= 1

    try:
        set_config('controlmode', '0')

        do_burst('0.0020s',   '1 EV', '9 images (normal, 4 unders and 4 overs)', 'double_in_single_run', True)
        do_burst('0.0015s',   '1 EV', '9 images (normal, 4 unders and 4 overs)', 'double_in_single_run', False)#
        do_burst('0.2500s',   '1 EV', '5 images (normal, 2 unders and 2 overs)', 'no_repeat', False)
        do_burst('0.0012s',   '1 EV', '9 images (normal, 4 unders and 4 overs)', 'double_in_single_run', False)

        do_burst('0.1666s',   '1 EV', '5 images (normal, 2 unders and 2 overs)', 'double_in_two_runs', False)
        do_burst('0.2000s',   '1 EV', '5 images (normal, 2 unders and 2 overs)', 'double_in_two_runs', False)

        do_burst('0.2500s',   '1 EV', '5 images (normal, 2 unders and 2 overs)', 'double_in_two_runs', False)
        do_burst('0.1666s',   '1 EV', '5 images (normal, 2 unders and 2 overs)', 'double_in_two_runs', False)
        do_burst('0.2000s',   '1 EV', '5 images (normal, 2 unders and 2 overs)', 'double_in_two_runs', False)

        do_burst('0.2500s',   '1 EV', '5 images (normal, 2 unders and 2 overs)', 'double_in_two_runs', False)
        do_burst('0.2000s',   '1 EV', '5 images (normal, 2 unders and 2 overs)', 'no_repeat', False)

        do_burst('0.0012s',   '1 EV', '9 images (normal, 4 unders and 4 overs)', 'double_in_single_run', False)
        do_burst('0.1666s',   '1 EV', '5 images (normal, 2 unders and 2 overs)', 'no_repeat', False)
        do_burst('0.0015s',   '1 EV', '9 images (normal, 4 unders and 4 overs)', 'double_in_single_run', False)
        do_burst('0.0020s',   '1 EV', '9 images (normal, 4 unders and 4 overs)', 'double_in_single_run', False)#

        print(f"Done. Check the card. {time.time()-t0[0]:.2f} s")
        print()

    finally:
        gp.gp_camera_exit(camera, context)

if __name__ == "__main__":
    unlock_and_burst()
