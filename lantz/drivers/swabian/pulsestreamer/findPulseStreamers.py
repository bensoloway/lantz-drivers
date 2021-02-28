import os
import re
import socket
import struct
import time
import json

__all__ = ['findPulseStreamers']


def getIpAddresses():
    """Get all my local IPs"""

    if os.name == 'posix':
        return getIpAddresses_posix()
    else:
        return getIpAddresses_windows()


def getIpAddresses_posix():
    """List all addresses, POSIX implementation"""
    import subprocess

    addr_list = []
    try:
        # Try to obtain list of local IPs using ip command with JSON output.
        # > ip -j -p -4 address show scope global
        import json
        s = subprocess.Popen(["ip", "-j", "-p", "-4", "address", "show", "scope", "global"], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        s.wait()
        assert s.returncode == 0
        data = json.loads(s.stdout.read())
        for iface in data:
            for addr in iface['addr_info']:
                addr_list.append(addr['local'])
        return addr_list
    except:
        # Attempt failed
        pass
    
    try:
        # If "ip" command has failed try calling it without JSON output. 
        # Older versions of iptools do not support "-j" key.
        # May not work on all systems  due to possible difference in the output formatting.
        # > ip -4 address show scope global
        s = subprocess.Popen(["ip", "-4", "address", "show", "scope", "global"], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        s.wait()
        assert s.returncode == 0
        for line in s.stdout.readlines():
            line = line.decode().strip()
            if 'inet' in line:
                ip = line.split()[1].split('/')[0]
                addr_list.append(ip)
        return addr_list
    except:
        # Attempt failed
        pass
        
    try:
        # If everything else has failed then try using obsolete command "ifconfig"
        s = subprocess.Popen(["ifconfig"], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        s.wait()
        assert s.returncode == 0
        for line in s.stdout.readlines():
            line = line.decode().strip()
            if line.startswith('inet '):
                addr = line.split()[1]
                addr_list.append(addr)
        return addr_list
    except:
        raise Exception("Unable to determine IP address(es) of the local machine. Make sure you have iptools installed and 'ip' command is available.")


def getIpAddresses_windows():
    """List all IP addresses, Windows implementation"""
    addrList = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    return [item[4][0] for item in addrList]



class DiscoverDevices():
    multicast_addr = ('239.255.255.82', 12345)
    broadcast_addr = ('255.255.255.255', 12345)

    def __init__(self, bind_ip=None):
        self.src_addr = (bind_ip, 0)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow socket to share the address use with other programs if any
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Make socket nonblocking
        self.sock.settimeout(0.001)
        # Allow socket to send broadcasts
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # Set multicast TTL to 1
        ttl = struct.pack('b', 1)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        

        # Tell from what IP to send packets
        if bind_ip:
            self.sock.bind(self.src_addr)
        
        msg = b'PS-DISCOVERY'
        # Send query to broadcast        
        self.sock.sendto(msg, self.broadcast_addr)
        # Send query to multicast
        self.sock.sendto(msg, self.multicast_addr)
        
    def __del__(self):
        self.sock.close()
        
    
    def devices(self):
        try:
            while True:
                msg, addr = self.sock.recvfrom(256)
                if msg.startswith(b'PS-DISCOVERY-REPLY'):
                    msg = msg.replace(b'PS-DISCOVERY-REPLY', b'')
                    yield (addr[0], msg)
                    
        except socket.timeout:
            return


def serial_cmpi(str1, str2):
    """Return True if serial strings are equivalent."""
    str1 = str1.replace(':','').replace('-','').lower()
    str2 = str2.replace(':','').replace('-','').lower()
    return str1 == str2


def findPulseStreamers(search_serial='', _recv_callback=None):
    """
    Find Pulse Streamer devices connected to the network. 

    Specific serial number can be queried by specifying "search_serial" parameter.
    """

    if search_serial != '':
        mac_regex = re.compile(r"((?:[\da-fA-F]{2}[:-]?){5}[\da-fA-F]{2})")
        if mac_regex.match(search_serial) is None:
            raise ValueError(
                'Parameter "search_serial" has incorrect value "{}".'
                ' Must have the following format:'
                ' "12:34:56:78:9A:BC" or be an empty string.'.format(search_serial)
            )

    queries = list()

    for addr in getIpAddresses():
        queries.append(DiscoverDevices(bind_ip=addr))
    
    # wait for replies to arrive
    time.sleep(0.5)
    
    # Read replies collected by every discovery method
    addr_list = []
    info_list = []
    for q in queries:
        for dev in q.devices():
            if dev[0] not in addr_list:
                if callable(_recv_callback):
                    _recv_callback(dev[0], dev[1])
                addr_list.append(dev[0])
                info_list.append(json.loads(dev[1]))
    
    ps_list = list(zip(addr_list, info_list))

    if len(search_serial)>0:
        # Filter out not matching devices if specific serial number was requested
        return [d for d in ps_list if serial_cmpi(d[1]['serial'], search_serial)]
    else:
        return ps_list


def test():
    print('Searching all devices.')
    print(findPulseStreamers())
    
    specific_serials = [
        '00:26:32:f0:3b:1a',
        '00-26-32-f0-3b-1b', 
        '00:26:32:f0:3b:1b', 
        '002632f03b1b', 
        'WS-26-32-f0-G-WR'
    ]
    for ser in specific_serials:
        print('Searching specific serial:', ser)
        print(findPulseStreamers(ser))


if __name__ == "__main__":
    print(findPulseStreamers())
    
