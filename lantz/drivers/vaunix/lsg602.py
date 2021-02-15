# -*- coding: utf-8 -*-
"""
    lantz.drivers.vaunix.LSG602
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Implementation of Vauniv LSG602 Lab Brick signal generator

    Author: Jonathan Karsch
    Date: 10/12/2018
"""

import numpy as np
from lantz import Action, Feat, DictFeat, ureg
from lantz.messagebased import MessageBasedDriver
from lantz.driver import Driver
from collections import OrderedDict
from ctypes import c_int, c_double, byref, c_char_p, CDLL
CDLL_file = "C:/users/Dickens/code/lantz/lantz/drivers/vaunix/LSG 64Bit SDK/vnx_fsynth.dll"
va = CDLL(CDLL_file)
va.fnLSG_GetNumDevices()
DEVID = 1

class LSG602(Driver):

        DEFAULTS = {
            'COMMON': {
                'write_termination': '\r\n',
                'read_termination': '\r\n',
            }
        }

        @Action()
        def initialize(self):
            va.fnLSG_InitDevice(DEVID)

        @Action()
        def finalize(self):
            va.fnLSG_CloseDevice(DEVID)
        
        @Feat()
        def status(self):
            return(va.fnLSG_GetDeviceStatus(DEVID))
        
        # Frequency is specified in 100 KHz increments, where: frequency = Frequency (Hz) / 100,000
        # For example, to specify an output frequency of 1.5 GHz, frequency = 15,000.
        @Feat
        def frequency(self):
            f10e5 = va.fnLSG_GetFrequency(DEVID)
            return(f10e5*100000)
        
        @frequency.setter
        def frequency(self,value):
            print("frequency")
            fset = int(value/100000)
            print(fset)
            va.fnLSG_SetFrequency(DEVID,fset)
        
        '''
        The powerlevel is encoded as the number of .25dB increments,
        with a resolution of .5dB. To set a power level of +5 dBm,
        for example, powerlevel would be 20. To set a power level of -20 dBm, powerlevel would be -80.
        '''
        @Feat
        def power(self):
            powerlevel = va.fnLSG_GetPowerLevel(DEVID)
            power = (40-powerlevel)/4
            return(power)
        
        @power.setter
        def power(self,value):
            powerlevel = int(4*value)
            va.fnLSG_SetPowerLevel(DEVID,powerlevel)

        @Feat
        def RFon(self):
            on = va.fnLSG_GetRF_On(DEVID)
            return(on)
        
        @RFon.setter
        def RFon(self,value):
            va.fnLSG_SetRFOn(DEVID,value)

        