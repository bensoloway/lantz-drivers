#import class Sequence and OutoutState for advanced sequence building
from lantz.drivers.swabian.pulsestreamer.sequence import  Sequence, OutputState

try:
    from tinyrpc import RPCClient
    from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
    from tinyrpc.transports.http import HttpPostClientTransport
except Exception as e:
    print(str(e))
    assert False, \
"""
Failed to import JSON-RPC library. Ensure that you have it installed by typing
> pip install tinyrpc or pip install tinyrpc --upgrade (ensure suppport of Python3)
> pip install gevent-websocket
> pip install requests
in your terminal.
"""

# binary and base64 conversion
import struct
import base64
import six
import numpy as np

#stream binary
import socket
import struct

from lantz.drivers.swabian.pulsestreamer.enums import ClockSource, TriggerStart, TriggerRearm
from lantz.drivers.swabian.pulsestreamer.version import __CLIENT_VERSION__, _compare_version_number

class PulseStreamer():
    """
    Simple python wrapper for a PulseStreamer 8/2
    that describes sequences in the form of sequence steps as (time, [0,1,3], 0.8, -0.4),
    where time is an integer in ns (clock ticks),
    [0,1,3] is a list numbering the channels that should be high
    the last two numbers specify the analog outputs in volt.
    For advanced sequence creation use the method createSequence() and the functionality of
    the class Sequence described in the documentation of the Pulse Streamer 8/2.
    """
    REPEAT_INFINITELY=-1

    def __init__(self, ip_hostname='pulsestreamer'):
        print("Connect to Pulse Streamer via JSON-RPC.")
        print("IP / Hostname:", ip_hostname)
        self.ip_address = ip_hostname
        url = 'http://'+ip_hostname+':8050/json-rpc'
        try:
            client = RPCClient(JSONRPCProtocol(), HttpPostClientTransport(url, timeout=20))
            self.proxy = client.get_proxy()
            try:
                self.proxy.getSerial()
            except:
                try:
                    self.proxy.isRunning()
                    assert False, "Pulse Streamer class not compatible with current firmware. Please update your firmware." \
                        "For detailed information visit https://www.swabianinstruments.com/pulse-streamer-8-2-firmware/ " \
                        "or contact support@swabianinstruments.com"
                except AssertionError:
                    raise
                except:
                    assert False, "No Pulse Streamer found at IP/Host-address: "+ip_hostname
        except AssertionError:
            raise
        except:
            assert False, "No Pulse Streamer found at IP/Host-address: "+ip_hostname
        firmware_version=self.proxy.getFirmwareVersion()
        self.__version_1_1_or_higher=True if _compare_version_number(firmware_version, '1.1.0') >=0 else False
        print("Pulse Streamer 8/2 firmware: v" + firmware_version)
        print("Client software: v" + __CLIENT_VERSION__)
        if(_compare_version_number(firmware_version) >= 0):
            print("The Pulse Streamer 8/2 firmware is more up to date than your client software. We recommend to update your client software.")
            print("For detailed information visit https://www.swabianinstruments.com/support/downloads/ or contact support@swabianinstruments.com")
        elif(_compare_version_number('.'.join(__CLIENT_VERSION__.split('.')[0:2]+['0']),'.'.join(firmware_version.split('.')[0:2]+['0'])) > 0):
            print("Your client software is more up to date than the Pulse Streamer 8/2 firmware. We recommend to update the firmware of your Pulse Streamer 8/2.")
            print("For detailed information visit https://www.swabianinstruments.com/support/downloads/ or contact support@swabianinstruments.com")

    def reset(self):
        return self.proxy.reset()

    def reboot(self):
        return self.proxy.reboot()
        
    def constant(self, state=OutputState.ZERO()):
        if isinstance(state, OutputState):
            state=state.getData()
            output=(0, state[0], state[1], state[2])
        else:
            output = self.convert_sequence_step((0, state[0], state[1], state[2]))
        self.proxy.constant(output)

    def forceFinal(self):
        return self.proxy.forceFinal()
    
    def createOutputState(self, digi, A0=0.0, A1=0.0):
        output=OutputState(digi=digi,A0=A0, A1=A1)
        return output

    def createSequence(self): #ToDo parameter for safe communication/sequence creation
        seq = Sequence()
        return seq
    
    def stream(self, seq, n_runs=REPEAT_INFINITELY, final=OutputState.ZERO()):
        if (self.__version_1_1_or_higher):
            #print("Debug information: Stream binary")
            #return self.__stream_binary(seq, n_runs, final)
            self.__stream_binary(seq, n_runs, final)
        else:
            #print("Debug information: Stream JSON")
            self.__stream_json(seq, n_runs, final)

    def __stream_json(self, seq, n_runs=REPEAT_INFINITELY, final=OutputState.ZERO()):
        if isinstance(final, OutputState):
            state=final.getData()
            final =(0, state[0], state[1], state[2])
        else:
            final = self.convert_sequence_step((0, final[0], final[1], final[2]))

        if six.PY2:
            if isinstance(seq, Sequence):
                s = self.enc(seq.getData())
            else:
                s = self.enc(seq)
        else:
            if isinstance(seq, Sequence):
                s = self.enc(seq.getData()).decode("utf-8")
            else:
                s = self.enc(seq).decode("utf-8")
        
        self.proxy.stream(s, n_runs, final)

    def __stream_binary(self, seq, n_runs=REPEAT_INFINITELY, final=OutputState.ZERO()):
        HOST = self.ip_address #'192.168.0.101' #''127.0.0.1'  # The server's hostname or IP address
        PORT = 21328        # The port used by the server
        MAGIC_TOKEN = 0x53504953
        COMMAND=0
        
        if isinstance(final, OutputState):
            state=final.getData()
            final =(0, state[0], state[1], state[2])
        else:
            final = self.convert_sequence_step((0, final[0], final[1], final[2]))

        if isinstance(seq, Sequence):
            s_enc = self.enc_binary(seq.getData())
        else:
            s_enc = self.enc_binary(seq)
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        pulses=len(s_enc)//4
        pad=32-((pulses*9)%32)
        #pad_list=pad*[0x0]
        CommandHeader=[MAGIC_TOKEN, 1, COMMAND, 0x0, pulses*9+32+pad, 0]
        StreamHighLevelHeader=[n_runs, pulses, final[2], final[3], final[1], 0x0, 0x0, 0x0, 0]
        s.sendall(struct.pack('<4I2Q', *CommandHeader))
        s.sendall(struct.pack('<qQ2h4BQ', *StreamHighLevelHeader))
        fmt = '<' + len(s_enc)//4*'IBhh'
        s.sendall(struct.pack(fmt, *s_enc))
        s.sendall(b'\0' * pad)
        data = s.recv(32)
        s.close()

        return_value=struct.unpack('<4I2Q', data)
        return return_value

    def isStreaming(self):
        return self.proxy.isStreaming()

    def hasFinished(self):
        return self.proxy.hasFinished()

    def hasSequence(self):
        return self.proxy.hasSequence()

    def startNow(self):
        return self.proxy.startNow()

    def getTemperature(self):
        return self.proxy.getTemperature()

    def getUnderflow(self):
        return self.proxy.getUnderflow()

    def getDebugRegister(self):
        return self.proxy.getDebugRegister()

    def selectClock(self, source):
        if not isinstance(source, ClockSource):
            raise TypeError("source must be an instance of ClockSource Enum")
        else:
            return self.proxy.selectClock(source.value)

    def getClock(self):
        return ClockSource(self.proxy.getClock())

    def setSquareWave125MHz(self, channels=[]):
        return self.proxy.setSquareWave125MHz(self.chans_to_mask(channels))

    def getFirmwareVersion(self):
        return self.proxy.getFirmwareVersion()

    def getHardwareVersion(self):
        return self.proxy.getHardwareVersion()

    def getSupplyState(self):
        return self.proxy.getSupplyState()

    def getSerial(self):
        if (self.__version_1_1_or_higher):
            return self.proxy.getSerial()
        else:
            return self.proxy.getSerial(1)
    
    def getFPGAID(self):
        if (self.__version_1_1_or_higher):
            return self.proxy.getFPGAID()
        else:
            return self.proxy.getSerial(0)
    
    def setTrigger(self, start, rearm=TriggerRearm.AUTO):
        if not isinstance(start, TriggerStart):
            raise TypeError("start must be an instance of TriggerStart Enum")
        else:
            if not isinstance(rearm, TriggerRearm):
                raise TypeError("rearm must be an instance of TriggerRearm Enum")
            else:
                return self.proxy.setTrigger(start.value, rearm.value)

    def getTriggerStart(self):
        return TriggerStart(self.proxy.getTriggerStart())
    
    def getTriggerRearm(self):
        return TriggerRearm(self.proxy.getTriggerRearm())
    
    def setHostname(self, hostname):
        return self.proxy.setHostname(hostname)

    def getHostname(self):
        return self.proxy.getHostname()

    def setNetworkConfiguration(self, dhcp, ip='', netmask='', gateway='', testmode=True):
        return self.proxy.setNetworkConfiguration(dhcp, ip, netmask, gateway, testmode)

    def getNetworkConfiguration(self, permanent=False):
        assert permanent in [True, False]
        return self.proxy.getNetworkConfiguration(permanent)
        
    def applyNetworkConfiguration(self):
        return self.proxy.applyNetworkConfiguration()

    def rearm(self):
        return self.proxy.rearm()

    # def setAnalogCalibration(self, dc_offset_a0=0, dc_offset_a1=0, slope_a0=1, slope_a1=1):
    #     HOST = self.ip_address  # The server's hostname or IP address
    #     PORT = 21328        # The port used by the server
    #     MAGIC_TOKEN = 0x53504953
    #     COMMAND=0x101

    #     send_list=[float(dc_offset_a0), float(dc_offset_a1), float(slope_a0), float(slope_a1)]
    #     #print(send_list)
        
    #     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     s.connect((HOST, PORT))
    #     CommandHeader=[MAGIC_TOKEN, 1, COMMAND, 0x0, 32, 0]
    #     s.sendall(struct.pack('<4I2Q', *CommandHeader))
    #     s.sendall(struct.pack('<4d', *send_list))
    #     data = s.recv(32)
    #     s.close()

    #     return_value=struct.unpack('<4I2Q', data)
    #     #print(return_value) #DEBUG
    #     return return_value[2]

    # def getAnalogCalibration(self):
    #     HOST = self.ip_address  # The server's hostname or IP address
    #     PORT = 21328        # The port used by the server
    #     MAGIC_TOKEN = 0x53504953
    #     COMMAND=0x102

    #     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     s.connect((HOST, PORT))
    #     CommandHeader=[MAGIC_TOKEN, 1, COMMAND, 0x0, 0, 0]
    #     s.sendall(struct.pack('<4I2Q', *CommandHeader))
    #     data = s.recv(32)
    #     cmd_answer=struct.unpack('<4I2Q', data)
    #     data =s.recv(cmd_answer[4])
    #     cmd_data=struct.unpack('<4d', data)
    #     s.close()
    #     #print(cmd_answer) #DEBUG
    #     #print(cmd_data) #DEBUG
    #     assert cmd_answer[2] ==0
    #     key_list=['dc_offset_a0', 'dc_offset_a1', 'slope_a0', 'slope_a1']
        
    #     return dict( zip(key_list,list(cmd_data) ))

    def setAnalogCalibration(self, dc_offset_a0=0, dc_offset_a1=0, slope_a0=1, slope_a1=1):
        return self.proxy.setAnalogCalibration(dc_offset_a0, dc_offset_a1, slope_a0, slope_a1)
    
    def getAnalogCalibration(self):
        return self.proxy.getAnalogCalibration()

    def dec(self, b64):
        sdec = base64.b64decode(b64)
        import struct
        fmt = '>' + len(sdec)//3//3*'IBhh'
        s = struct.Struct(fmt)
        res = s.unpack(sdec)          
        return res
        
    def enc(self, seq):
        """
        Convert a human readable python sequence to a base64 encoded string and split sequence steps with duration exceeding limit of unsigned int
        """
        convert_list = self.enc_binary(seq)

        fmt = '>' + len(convert_list)//4*'IBhh'
        s=struct.pack(fmt, *convert_list)  
        return base64.b64encode(s)

    def enc_binary(self, seq):
        """
        Convert a human readable python sequence to a base64 encoded string and split sequence steps with duration exceeding limit of unsigned int
        """
        s = b''
        convert_list = []
        if type(seq[0][1])== list:
            for sequence_step in seq:
                if sequence_step[0] > 0xffffffff:
                    mod=sequence_step[0]%0xffffffff
                    count=sequence_step[0]//0xffffffff
                    for i in range(count):
                        sequence_step = (0xffffffff, sequence_step[1], sequence_step[2], sequence_step[3])
                        convert_list.extend(self.convert_sequence_step(sequence_step))
                    else:
                        sequence_step = (mod, sequence_step[1], sequence_step[2], sequence_step[3])
                convert_list.extend(self.convert_sequence_step(sequence_step))    
        else:
            for sequence_step in seq:
                if sequence_step[0] > 0xffffffff:
                    mod=sequence_step[0]%0xffffffff
                    count=sequence_step[0]//0xffffffff
                    for i in range(count):
                        sequence_step = (0xffffffff, sequence_step[1], sequence_step[2], sequence_step[3])
                        convert_list.extend(sequence_step)
                    else:
                        sequence_step = (mod, sequence_step[1], sequence_step[2], sequence_step[3])
                convert_list.extend(sequence_step)

        assert len(convert_list)//4<=2e6, "The resulting sequence length exceeds the limit of two million sequnence steps"

        return convert_list
    
    def convert_sequence_step(self, sequence_step):
        t, chans, a0, a1 = sequence_step
        assert (abs(a0)<=1 and abs(a1)<=1), "Pulse Streamer 8/2 supports "\
                "analog voltage range of +/-1V" #check hardware
        assert t>=0
        return (t, self.chans_to_mask(chans), int(round(0x7fff*a0)), int(round(0x7fff*a1)))
        #return (t, self.chans_to_mask(chans), a0, a1)

    def chans_to_mask(self, chans):
        mask = 0
        for chan in chans:
            assert chan in range(8),"Pulse Streamer 8/2 supports "\
            "up to eight digital channels"
            mask |= 1<<chan
        return mask
        
"""---------Test-Code-------------------------------"""

if __name__ == '__main__':
    pulser = PulseStreamer(ip_hostname='pulsestreamer')

    print("Serial number:", pulser.getSerial())
    print("Firmware Version:", pulser.getFirmwareVersion())
