import sys
import time
from tap import Tap
import os
import keyboard
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
from datetime import datetime as dt

FS = 48000

def expand_data(data: list[list[str]], column = 0) -> list[list[str]]:
    if column == len(data[0]):
        numbers = [f"{x:03d}" for x in range(len(data))]
        result = [[a] + b for a,b in zip(numbers, data)]
        return result
    
    idx = -1-column
    output = []
    for item in data:
        split_cell = item[idx]
        parts = split_cell.split('/')
        for i in range(len(parts)):
            new_item = item[:]
            new_item[idx] = parts[i]
            output.append(new_item)
    
    return expand_data(output, column+1)
        


def read_list_file(list_file: str) -> tuple[list[str], list[list[str]]]:
    with open(list_file, 'r') as f:
        lines = f.readlines()

    all_data = []
    for line in lines:
        data = line.split('\t')
        data = [x.strip() for x in data if len(x.strip()) > 0]
        all_data.append(data)

    headers = ["Id"] + all_data[0]
    data = all_data[1:]
    data = expand_data(data)
    return headers, data


def print_settings_delta(header, current_settings, new_settings):
    id = new_settings[0]
    print(f"\n========================= Starting Capture [{id}] =========================\n")
    print("All Settings:\n")
    for i in range(1, len(header)):
        name = header[i]
        b = new_settings[i]
        print(f"{name}: {b}")

    print("\n\nChanges:\n")
    for i in range(1, len(header)):
        name = header[i]
        a = current_settings[i]
        b = new_settings[i]
        if a != b:
            print(f"{name}: {b}")


def run_capture(input_data: np.ndarray) -> np.ndarray:
    seconds = len(input_data) / FS
    data = sd.playrec(input_data, blocking=False)
    start = dt.utcnow()
    while True:
        elapsed = (dt.utcnow() - start).seconds
        if elapsed > (seconds-1):
            break

        cb = sd.get_status()
        if cb.output_underflow or cb.output_underflow or cb.input_underflow or cb.input_overflow:
            print("An error occurred during capture. Aborting")
            return np.zeros(10, np.float32)
        
        print(f"\r{int(elapsed)}/{int(seconds)} sec", end='')
        time.sleep(0.2)
    sd.wait()
    print("")
    return data


def should_run() -> bool:
    print("\nPress Enter to run this capture, or X to skip it")
    while True:
        try:
            if keyboard.is_pressed("x"):
                print("Skipping this capture")
                time.sleep(0.5)
                return False
            if keyboard.is_pressed("enter"):
                print("Running capture now...")
                time.sleep(0.5)
                return True
        except:
            break  # if user pressed a key other than the given key the loop will break


def setup_audio():
    apis = sd.query_hostapis()
    asio_api = None
    for i in range(len(apis)):
        if 'ASIO' in apis[i]['name']:
            asio_api = apis[i]
            break

    if len(asio_api) is None:
        print("No ASIO devices found, aborting!")
        sys.exit()

    devices = sd.query_devices()
    asio_devices = []
    for i in range(len(devices)):
        if i in asio_api["devices"]:
            asio_devices.append(devices[i])

    print(sd.DeviceList(asio_devices))
    idx = input("Select your device: ")
    device = asio_devices[int(idx)]
    input_channel = int(input(f"Input Channel (0-{device['max_input_channels']-1}): "))
    output_channel = int(input(f"Output Channel (0-{device['max_output_channels']-1}): "))
    
    sd.default.device = device["index"]
    sd.default.samplerate = FS
    sd.default.dtype = 'float32'
    sd.default.channels = (1, 1)
    sd.default.blocksize = 1024
    asio_in = sd.AsioSettings(channel_selectors=[input_channel])
    asio_out = sd.AsioSettings(channel_selectors=[output_channel])
    sd.default.extra_settings = asio_in, asio_out
    print("Device Ready!")


def capture(input_file: str, list_file: str, output_dir: str, start_index: int = 0):
    setup_audio()
    sr, input_data = wavfile.read(input_file)
    assert sr == FS, "Samplerate of input file must be 48Khz!"

    header, data = read_list_file(list_file)
    with open(output_dir + "/output_list.txt", 'w') as f:
        f.write(",".join(header) + "\n")
        f.writelines([",".join(str(q) for q in x)+"\n" for x in data])
    
    current_settings = ['' for _ in header]
    for i in range(len(data)):
        if i < start_index:
            continue

        new_settings = data[i]
        print_settings_delta(header, current_settings, new_settings)
        current_settings = new_settings
        id = current_settings[0]
        output_file = os.path.join(output_dir, f"Capture-{id}.wav")
        if should_run():
            capture_data = run_capture(input_data)
            wavfile.write(output_file, FS, capture_data)
            print("Capture Completed")


class CaptureArgs(Tap):
    input_file: str
    list_file: str
    output_dir: str
    start: int | None = None


if __name__ == '__main__':
    args = CaptureArgs(underscores_to_dashes=True).parse_args()
    capture(args.input_file, args.list_file, args.output_dir, args.start or 0)
