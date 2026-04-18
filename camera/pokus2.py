import gphoto2 as gp
import time

def unlock_and_burst():
    t0 = time.time()
    error, camera = gp.gp_camera_new()
    context = gp.gp_context_new()
    gp.gp_camera_init(camera, context)

    def set_config(name, value):
        error, config = gp.gp_camera_get_config(camera, context)
        error, child = gp.gp_widget_get_child_by_name(config, name)
        if error == gp.GP_OK:
            gp.gp_widget_set_value(child, value)
            gp.gp_camera_set_config(camera, config, context)
            print(f"Set {name} to {value}")
            return True
        print(f"Failed to set {name}")
        return False

    def do_burst(shutterspeed, aebracketingstep, aebracketingpattern):
        set_config('shutterspeed', shutterspeed)
        set_config('aebracketingstep', aebracketingstep)
        set_config('bracketing', 'On')
        set_config('aebracketingpattern', aebracketingpattern)

        # 5. Finalize setup
        set_config('capturetarget', 'Memory card')
        set_config('capturemode', 'Burst')

        bracket_count = int(aebracketingpattern[0])
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
        for _ in range(10):
            set_config('controlmode', '0')
            do_burst('0.0008s', '2/3 EV', '9 images (normal, 4 unders and 4 overs)')
            do_burst('0.05s',   '2/3 EV', '9 images (normal, 4 unders and 4 overs)')
            do_burst('0.5s',   '1 EV', '2 images (normal and over)')
            print(f"Done. Check the card. {time.time()-t0:.2f} s")

    finally:
        gp.gp_camera_exit(camera, context)

if __name__ == "__main__":
    unlock_and_burst()
