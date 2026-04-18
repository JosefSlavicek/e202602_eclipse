import gphoto2 as gp
import time

def unlock_and_burst():
    error, camera = gp.gp_camera_new()
    context = gp.gp_context_new()
    gp.gp_camera_init(camera, context)

    def set_config(name, value):
        error, config = gp.gp_camera_get_config(camera, context)
        error, child = gp.gp_widget_get_child_by_name(config, name)
        if error == gp.GP_OK:
            gp.gp_widget_set_value(child, value)
            gp.gp_camera_set_config(camera, config, context)
            return True
        return False

    try:
        print("1. Resetting Control Mode...")
        set_config('controlmode', '0') # Hand control back to camera/logic
        
        print("2. Setting Target to Card...")
        set_config('capturetarget', 'Memory card')

        print("3. Enabling AE Bracketing via d0c2...")

        print("4. Setting Drive Mode to Burst...")
        set_config('capturemode', 'Burst')

        # FINAL SYNC: Sometimes the camera needs a moment to 'digest' config changes
        time.sleep(1)
        bracket_count = 9 
        print(f"Firing {bracket_count} shots...")
        
        for i in range(bracket_count):
            gp.gp_camera_trigger_capture(camera, context)
            # We don't sleep here; we want the commands to hit the buffer ASAP

        # 3. The Event Loop (Corrected Unpacking)
        print("Waiting for files to write to card...")
        start_time = time.time()
        while bracket_count > 0:
            # Note the three variables here: err, ev_type, ev_data
            err, ev_type, ev_data = gp.gp_camera_wait_for_event(camera, 100, context)
            
            if ev_type == gp.GP_EVENT_FILE_ADDED:
                print(f"File saved: {ev_data.name}")
                bracket_count -= 1
###            elif ev_type == gp.GP_EVENT_CAPTURE_COMPLETE:
###                # Some Nikon firmware sends this after the sequence
###                print('')
###                break
        # Instead of a high-level capture, we use a simple trigger.
        # This acts like a 'tap' on the shutter button.
        # Wait for the mechanical burst to finish
        time.sleep(2)
        print("Done. Check the card.")

    finally:
        gp.gp_camera_exit(camera, context)

if __name__ == "__main__":
    unlock_and_burst()
