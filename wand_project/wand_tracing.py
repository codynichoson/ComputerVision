# coding: utf-8

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

# Create and set logger
# logger = createConsoleLogger(LoggerLevel.Debug)
# setGlobalLogger(logger)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(), device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

# Optimal parameters for registration
# set True if you need
need_bigdepth = False
need_color_depth_map = False

bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
color_depth_map = np.zeros((424, 512),  np.int32).ravel() if need_color_depth_map else None

trace = []

while True:
    frames = listener.waitForNewFrame()

    color = frames["color"]
    ir = frames["ir"]
    depth = frames["depth"]

    registration.apply(color, depth, undistorted, registered, bigdepth=bigdepth, color_depth_map=color_depth_map)

    ir_array = ir.asarray() / 65535.
    mask = np.zeros(ir_array.shape)

    width = int(ir_array.shape[1])
    height = int(ir_array.shape[0])
    dim = (width, height)

    color_array = cv2.resize(color.asarray(), (int(1920 / 3), int(1080 / 3)))
    color_crop = color_array[:, 103:537]
    color_align = cv2.resize(color_crop, dim) # new color array size -> 360x434

    # Find brightest pixel in IR image
    max = np.max(ir_array)

    if max > 0.95:
        max_indices = np.where(ir_array == max)
        max_index = [max_indices[0][0], max_indices[1][0]]
        trace.append(max_index)
        if len(trace) == 100: # Remove first trace element once it hits 100 pixels in length
            trace.pop(0)   

    elif max < 0.95 and len(trace) > 0:
        trace.pop(0)

    elif max < 0.95:
        pass

    # Draw points in mask frame
    for point in trace:
        size = 3
        for i in range(point[0]-size, point[0]+size):
            for j in range(point[1]-size, point[1]+size):
                ir_array[i][j] = 255
                mask[i][j] = 255
                color_align[i-10][j+10] = (0,0,255,1)

    # Find template from trace
    # temp_upper = np.min([sublist[0] for sublist in trace])
    # temp_lower = np.max([sublist[0] for sublist in trace])
    # temp_right = np.max([sublist[1] for sublist in trace])
    # temp_left = np.min([sublist[1] for sublist in trace])
    # template = mask[temp_upper:temp_lower, temp_left:temp_right]
    # print(template.shape)

    #mask = cv2.circle(mask, (max_index[1], max_index[0]), 5, (255, 0, 0), 2)

    ir_and_mask = np.concatenate((ir_array, mask), axis=1)
    cv2.imshow("Wand Stuff", ir_and_mask)
    #cv2.imshow("image", mask)
    #cv2.imshow("template", template)
    

    cv2.imshow("color", color_align)

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)