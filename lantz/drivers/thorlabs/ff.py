# -*- coding: utf-8 -*-

import os

from lantz import Feat, Action
from lantz.foreign import LibraryDriver
from ctypes import c_char_p, c_buffer
from time import sleep


class FF(LibraryDriver):

    lib_path = os.path.join(os.environ['PROGRAMFILES'], 'Thorlabs\\Kinesis')
    lib_name = 'Thorlabs.MotionControl.FilterFlipper.dll'

    LIBRARY_NAME = os.path.join(lib_path, lib_name)
    LIBRARY_PREFIX = ''

    COM_DELAY = 0.2

    def __init__(self, serial_no, *args, **kwargs):
        """
        serial_no: unique device identifier; found on the label of the device
        """
        self.prev_path = os.environ['PATH']
        os.environ['PATH'] = '{}{}{}'.format(self.lib_path, os.pathsep, self.prev_path)
        super(FF, self).__init__(*args, **kwargs)
        os.environ['PATH'] = self.prev_path
        self.serial_number = serial_no
        self.sn_ptr = c_char_p(str(serial_no).encode('ascii'))

        val = self.lib.TLI_BuildDeviceList()
        print(val)
        list_size = self.lib.TLI_GetDeviceListSize()
        self.list_device_serials()
        if list_size == 0:

            print('No devices available, sorry!')

        else:
            retval = self.lib.FF_Open(self.sn_ptr)
            if retval:
                raise RuntimeError('Could not initialize device: error {}'.format(retval))
        return

    def list_device_serials(self):
        # Devices found, next look for S/Ns by device type
        TL_cBufSize = 255
        devID = 37
        sBuf = c_buffer(TL_cBufSize)

        if self.lib.TLI_GetDeviceListByTypeExt(sBuf, TL_cBufSize, devID) != 0:

            print("No devices of type {0} found".format(devID))
            return

        serial_number_list = sBuf.value.decode().rsplit(",")[0:-1]
        print(serial_number_list)

        if not(self.serial_number in serial_number_list):

            print("No device with S/N {0} found".format(self.serial_number))
            return

    def initialize(self):
        self.lib.FF_StartPolling(self.sn_ptr, 200);
        sleep(1)
        return

    def finalize(self):
        self.lib.FF_StopPolling(self.sn_ptr)
        self.lib.FF_Close(self.sn_ptr)
        return

    @Feat()
    def position(self):
        return self.lib.FF_GetPosition(self.sn_ptr)

    @position.setter
    def position(self, value):
        self.lib.FF_MoveToPosition(self.sn_ptr, value)
        sleep(1)
        return

class CameraFlipper(FF):

    @Action()
    def camera_on(self):
        self.position = 1
        return

    @Action()
    def camera_off(self):
        self.position = 2
        return

class CollectionFlipper(FF):

    # verify same convention used in your setup
    @Action()
    def single_mode(self):
        self.position = 1
        return

    @Action()
    def multi_mode(self):
        self.position = 2
        return

if __name__ == '__main__':
    import logging
    import sys
    from lantz.log import log_to_screen
    import numpy as np
    log_to_screen(logging.CRITICAL)
    serial_no = sys.argv[1]
    with FF(serial_no) as inst:
        print(inst.position)
        inst.position = 2
        print(inst.position)
        sleep(1)
        inst.position = 1
