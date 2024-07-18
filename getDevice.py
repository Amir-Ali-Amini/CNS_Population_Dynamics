# =====================================================================================================
# =====================================================================================================
# =====================================================================================================
# ====================================  AMIR ALI AMINI ================================================
# ====================================    610399102    ================================================
# =====================================================================================================
# =====================================================================================================

import torch
import warnings
def get_device(module = None ,change_default_device = False, force_cpu = False, print_details = True):
    module_is_available = module != None 
    setting_module = module

    # find the available device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    # print the available device
    print_details and print(f"Your device is : {device}" if not force_cpu else f"{device} is available but it is forced to use CPU")

    # set module device to available device or force it to use cpu
    if module_is_available:
        try:
            if not force_cpu:
                setting_module = setting_module.to(device)
                print_details and print(f"Your module device has changed to : {device}")  
            else: 
                setting_module = setting_module.to("cpu")
                print_details and print(f"Your module device has changed to : CPU")  
        except:
            warnings.warn("an error occurred with setting device to module...")

    # change torch default device or force it to use cpu
    if change_default_device :  
        try:
            torch.set_default_device("cpu" if force_cpu else device)
            print_details and print(f"Your default device has changed to : {'CPU' if force_cpu else device}")  
        except:
            warnings.warn("an error occurred with setting torch default device...")

    # return device and new module instance to replace
    return (device if not force_cpu else "cpu" , setting_module)