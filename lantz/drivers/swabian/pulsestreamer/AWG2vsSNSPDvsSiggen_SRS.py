import numpy as np
import time
from itertools import count
import pyqtgraph as pg
from lantz import Q_

from spyre import Spyrelet, Task, Element
from spyre.repository import Repository
from spyre.plotting import LinePlotWidget
from spyre.widgets.task import TaskWidget
from spyre.widgets.rangespace import Rangespace
from spyre.widgets.param_widget import ParamWidget
from spyre.widgets.repository_widget import RepositoryWidget

from lantz.drivers.ni.daqmx import Device
from lantz.drivers.ni.daqmx import CounterInputTask, CountEdgesChannel
from lantz.drivers.ni.daqmx import AnalogInputTask, VoltageInputChannel

from lantz.drivers.swabian.pulsestreamer.PulseSequencesIQ_v2 import Pulses
from lantz.drivers.stanford import DG645
from lantz.drivers.agilent import AG33522A
from lantz.drivers.stanford import SG396
from lantz.drivers.agilent import E8257C

class AWGvsSNSPDvsSiggenSpyrelet(Spyrelet):

    requires = {
        'daq': Device,
        "awg": AG33522A,
        "pulses": Pulses,
        "siggen": SG396
    }

    @Task()
    def sweep(self, **kwargs):
        self.dataset.clear()
        params = self.run_parameters.widget.get()
        samples = params['samples']
        delay = params['interpoint delay']
        for sweep in range(params['sweeps']):
            for f in params['Frequency'].array:
                self.siggen.frequency = f
                t0 = time.time()
                prev = np.mean(self.task.read(samples_per_channel=samples))
                prev_t = t0
                time.sleep(delay)
                now = time.time()
                current = np.mean(self.task.read(samples_per_channel=samples))
                count_rate = (current - prev) / (now - prev_t)
                values = {
                    'sweep_idx': sweep,
                    'v': f,
                    'y': count_rate*self.ratio,
                }
                self.sweep.acquire(values)
        return


    @sweep.initializer
    def initialize(self):
        params = self.run_parameters.widget.get()
        inp_ch = params['counter 1']
        if inp_ch in self.daq.counter_input_channels:
            self.task = CounterInputTask('TaskVsTime_CI')
            channel = CountEdgesChannel(inp_ch)
            self.task.add_channel(channel)
        else:
            # should never get here
            pass
        self.setup_pulses()
        volts = Q_(1, "V")
        self.awg.lower_limit[1] = -3. * volts
        self.awg.high_limit[1] = 3. * volts
        self.awg.limit[1] = "ON"
        self.awg.function[1] = "dc"
        self.awg.output[1] = "ON"
        self.awg.offset[1] = params["voltage"]
        self.siggen.rf_amplitude = params["power"]
        self.siggen.rf_toggle = True
        self.task.start()
        return

    def setup_pulses(self):
        params = self.run_parameters.widget.get()
        self.pulses.laser_time = int(params["excitation_time"].to("ns").magnitude)
        self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
        self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
        self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
        seq = self.pulses.EOM()
        self.ratio = self.pulses.total_time/self.pulses.readout_time
        self.pulses.stream(seq)

    @sweep.finalizer
    def finalize(self):
        self.task.clear()
        self.awg.output[1] = "OFF"
        self.siggen.rf_toggle = False
        return

    def setup_pulses(self):
        params = self.run_parameters.widget.get()
        self.pulses.laser_time = int(params["excitation_time"].to("ns").magnitude)
        self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
        self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
        self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
        self.polarize_time = 0
        self.settle = 0
        self.reset = 0
        seq = self.pulses.CODMR()
        self.pulses.stream(seq)

    @Element(name='Run parameters')
    def run_parameters(self):
        params = [
            ('counter 1', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr0',
            }),
            ('Frequency', {
                'type': range,
                'units': "Hz",
                'default': {'func': 'linspace',
                            'start': 5e9,
                            'stop': 8e9,
                            'num': 301},
            }),
            ('voltage', {
                'type': float,
                'default': 0,
                'suffix': ' V',
                'units': "V"
            }),
            ('power', {
                'type': float,
                'default': -10,
            }),
            ('sweeps', {
                'type': int,
                'default': 10,
                'positive': True,
            }),
            ('samples', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ('interpoint delay', {
                'type': float,
                'default': 1,
                'nonnegative': True,
            }),
            ('latest window', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ("excitation_time", {
                'type': float,
                'default': 600e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("readout_time", {
                'type': float,
                'default': 150e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("delay_time", {
                'type': float,
                'default': 2e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("reset_time", {
                'type': float,
                'default': 5e-6,
                'suffix': ' s',
                'units': "s"
            }),
        ]
        w = ParamWidget(params)
        return w

    @Element()
    def latest_history(self):
        p = LinePlotWidget()
        p.plot('channel 1', symbol=None)
        return p

    @latest_history.on(sweep.acquired)
    def latest_history_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        w.set('channel 1', xs=latest_data.v, ys=latest_data.y)
        return

    @Element()
    def averaged(self):
        p = LinePlotWidget()
        p.plot('channel 1', symbol=None)
        return p

    @averaged.on(sweep.acquired)
    def averaged_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        ys = grouped.y
        ys_averaged = ys.mean()
        w.set('channel 1', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys.std())
        return

    @Element()
    def save(self):
        w = RepositoryWidget(self)
        return w

    # @save.on(sweep.running)
    def save_requested(self, ev):
        w = ev.widget
        state, = ev.event_args
        if state:
            return
        print(self.dataset.data)
        w.save_dataset(self.dataset)
        return

class LODMRSpyrelet(Spyrelet):

    requires = {
        'daq': Device,
        "awg": AG33522A,
        "pulses": Pulses,
        "siggen": SG396
    }

    @Task()
    def sweep(self, **kwargs):
        self.dataset.clear()
        params = self.run_parameters.widget.get()
        for sweep in range(params['sweeps']):
            for f in params['Frequency'].array:
                self.siggen.frequency = f
                ctrs_start = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
                time.sleep(params['interpoint delay'])
                ctrs_end = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
                dctrs = ctrs_end - ctrs_start
                ctrs_rates = dctrs[:,0] / dctrs[:,1]
                values = {
                    'sweep_idx': sweep,
                    'v': f,
                    'x': ctrs_rates[0],
                    'y': ctrs_rates[1],
                }
                self.sweep.acquire(values)
        return


    @sweep.initializer
    def initialize(self):
        params = self.run_parameters.widget.get()
        ctrs = [params['counter 1'], params['counter 2']]
        if len(set(ctrs)) != len(ctrs):
            raise RuntimeError('counter channels 1 and 2 must be different')
        self.ctr_tasks = list()
        for idx, ctr in enumerate(ctrs):
            task = CounterInputTask('counter ch {}'.format(idx))
            ch = CountEdgesChannel(ctr)
            task.add_channel(ch)
            task.start()
            self.ctr_tasks.append(task)
        self.setup_pulses()
        volts = Q_(1, "V")
        self.awg.lower_limit[1] = -3. * volts
        self.awg.high_limit[1] = 3. * volts
        self.awg.limit[1] = "ON"
        self.awg.function[1] = "dc"
        self.awg.output[1] = "ON"
        self.awg.offset[1] = params["voltage"]
        self.siggen.rf_amplitude = params["power"]
        self.siggen.rf_toggle = True
        return


    @sweep.finalizer
    def finalize(self):
        for ctr_task in self.ctr_tasks:
            ctr_task.clear()
        self.awg.output[1] = "OFF"
        self.siggen.rf_toggle = False
        return

    def setup_pulses(self):
        params = self.run_parameters.widget.get()
        self.measure_num = params["Measure #"]
        self.pulses.laser_time = int(params["excitation_time"].to("ns").magnitude)
        self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
        self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
        self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
        seq = self.pulses.L_CODMR(measure = self.measure_num)
        self.pulses.stream(seq)

    @Element(name='Run parameters')
    def run_parameters(self):
        params = [
            ('counter 1', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr0',
            }),
            ('counter 2', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr1',
            }),
            ('Frequency', {
                'type': range,
                'units': "Hz",
                'default': {'func': 'linspace',
                            'start': 6.5e9,
                            'stop': 7e9,
                            'num': 501},
            }),
            ('voltage', {
                'type': float,
                'default': 0,
                'suffix': ' V',
                'units': "V"
            }),
            ('Measure #', {
                'type': int,
                'default': 0,
            }),
            ('power', {
                'type': float,
                'default': 16,
            }),
            ('sweeps', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ('samples', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ('interpoint delay', {
                'type': float,
                'default': .1,
                'nonnegative': True,
            }),
            ('latest window', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ("excitation_time", {
                'type': float,
                'default': 150e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("readout_time", {
                'type': float,
                'default': 150e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("delay_time", {
                'type': float,
                'default': 2e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("reset_time", {
                'type': float,
                'default': 5e-6,
                'suffix': ' s',
                'units': "s"
            }),
        ]
        w = ParamWidget(params)
        return w

    @Element(name='Latest counts versus frequency')
    def latest(self):
        p = LinePlotWidget()
        p.plot('Channel 1')
        p.plot('Channel 2')
        return p

    @latest.on(sweep.acquired)
    def latest_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        w.set('Channel 1', xs=latest_data.v, ys=latest_data.x)
        w.set('Channel 2', xs=latest_data.v, ys=latest_data.y*(1 + self.measure_num))
        return

    @Element(name='Differential counts (latest)')
    def diff_latest(self):
        p = LinePlotWidget()
        p.plot('Channel 1-2')
        return p

    @diff_latest.on(sweep.acquired)
    def diff_latest_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        diff = latest_data.x - latest_data.y*(1 + self.measure_num)
        w.set('Channel 1-2', xs=latest_data.v, ys=diff)
        return

    @Element(name='Averaged counts versus frequency')
    def averaged(self):
        p = LinePlotWidget()
        p.plot('Channel 1')
        p.plot('Channel 2')
        return p

    @averaged.on(sweep.acquired)
    def averaged_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        xs = grouped.x
        ys = grouped.y
        xs_averaged = xs.mean()
        ys_averaged = ys.mean()
        w.set('Channel 1', xs=xs_averaged.index, ys=xs_averaged, yerrs=xs.std())
        w.set('Channel 2', xs=ys_averaged.index, ys=ys_averaged*(1 + self.measure_num), yerrs=ys.std())
        return

    @Element(name='Differential counts (averaged)')
    def diff_average(self):
        p = LinePlotWidget()
        p.plot('Channel 1-2')
        return p

    @diff_average.on(sweep.acquired)
    def diff_average_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        xs = grouped.x
        ys = grouped.y
        xs_averaged = xs.mean()
        ys_averaged = ys.mean()
        diff_averaged = xs_averaged - ys_averaged*(1 + self.measure_num)
        w.set('Channel 1-2', xs=xs_averaged.index, ys=diff_averaged, yerrs=np.sqrt(xs.var() + ys.var()))
        return

    @Element()
    def save(self):
        w = RepositoryWidget(self)
        return w

    # @save.on(sweep.running)
    def save_requested(self, ev):
        w = ev.widget
        state, = ev.event_args
        if state:
            return
        print(self.dataset.data)
        w.save_dataset(self.dataset)
        return

class PODMRSpyrelet(Spyrelet):

    requires = {
        'daq': Device,
        "awg": AG33522A,
        "pulses": Pulses,
        "siggen": SG396
    }

    @Task()
    def sweep(self, **kwargs):
        self.dataset.clear()
        params = self.run_parameters.widget.get()
        for sweep in range(params['sweeps']):
            for f in params['Frequency'].array:
                self.siggen.frequency = f
                t0 = time.time()
                prev = np.mean(self.task.read(samples_per_channel=samples))
                prev_t = t0
                time.sleep(delay)
                now = time.time()
                current = np.mean(self.task.read(samples_per_channel=samples))
                count_rate = (current - prev) / (now - prev_t)
                values = {
                    'sweep_idx': sweep,
                    'v': f,
                    'y': count_rate*self.ratio,
                }
                self.sweep.acquire(values)
        return


    @sweep.initializer
    def initialize(self):
        params = self.run_parameters.widget.get()
        inp_ch = params['counter 1']
        if inp_ch in self.daq.counter_input_channels:
            self.task = CounterInputTask('TaskVsTime_CI')
            channel = CountEdgesChannel(inp_ch)
            self.task.add_channel(channel)
        else:
            # should never get here
            pass
        self.setup_pulses()
        volts = Q_(1, "V")
        self.awg.lower_limit[1] = -3. * volts
        self.awg.high_limit[1] = 3. * volts
        self.awg.limit[1] = "ON"
        self.awg.function[1] = "dc"
        self.awg.output[1] = "ON"
        self.awg.offset[1] = params["voltage"]
        self.siggen.rf_amplitude = params["power"]
        self.siggen.rf_toggle = True
        self.task.start()
        return


    @sweep.finalizer
    def finalize(self):
        self.task.clear()
        self.awg.output[1] = "OFF"
        self.siggen.rf_toggle = False
        return

    def setup_pulses(self):
        params = self.run_parameters.widget.get()
        self.pulses.laser_time = int(params["excitation_time"].to("ns").magnitude)
        self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
        self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
        self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
        seq = self.pulses.Pulsed_ODMR(int(params["Pi"].to("ns").magnitude))
        self.ratio = self.pulses.total_time/self.pulses.readout_time
        self.pulses.stream(seq)

    @Element(name='Run parameters')
    def run_parameters(self):
        params = [
            ('counter 1', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr0',
            }),
            ('Frequency', {
                'type': range,
                'units': "Hz",
                'default': {'func': 'linspace',
                            'start': 5e9,
                            'stop': 8e9,
                            'num': 301},
            }),
            ('Pi', {
                'type': float,
                'default': 50e-9,
                'suffix': ' ns',
                'units': "ns"
            }),
            ('voltage', {
                'type': float,
                'default': 0,
                'suffix': ' V',
                'units': "V"
            }),
            ('power', {
                'type': float,
                'default': -10,
            }),
            ('sweeps', {
                'type': int,
                'default': 10,
                'positive': True,
            }),
            ('samples', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ('interpoint delay', {
                'type': float,
                'default': 1,
                'nonnegative': True,
            }),
            ('latest window', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ("excitation_time", {
                'type': float,
                'default': 600e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("readout_time", {
                'type': float,
                'default': 150e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("delay_time", {
                'type': float,
                'default': 2e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("reset_time", {
                'type': float,
                'default': 5e-6,
                'suffix': ' s',
                'units': "s"
            }),
        ]
        w = ParamWidget(params)
        return w

    @Element()
    def latest_history(self):
        p = LinePlotWidget()
        p.plot('channel 1', symbol=None)
        return p

    @latest_history.on(sweep.acquired)
    def latest_history_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        w.set('channel 1', xs=latest_data.v, ys=latest_data.y)
        return

    @Element()
    def averaged(self):
        p = LinePlotWidget()
        p.plot('channel 1', symbol=None)
        return p

    @averaged.on(sweep.acquired)
    def averaged_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        ys = grouped.y
        ys_averaged = ys.mean()
        w.set('channel 1', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys.std())
        return

    @Element()
    def save(self):
        w = RepositoryWidget(self)
        return w

    # @save.on(sweep.running)
    def save_requested(self, ev):
        w = ev.widget
        state, = ev.event_args
        if state:
            return
        print(self.dataset.data)
        w.save_dataset(self.dataset)
        return

class PODMRPLESpyrelet(Spyrelet):

    requires = {
        'daq': Device,
        "awg": AG33522A,
        "pulses": Pulses,
        "siggen": SG396
    }

    @Task()
    def sweep(self, **kwargs):
        self.dataset.clear()
        params = self.run_parameters.widget.get()
        for sweep in range(params['sweeps']):
            for v in params['voltage'].array:
                self.awg.offset[1] = v
                t0 = time.time()
                prev = np.mean(self.task.read(samples_per_channel=samples))
                prev_t = t0
                time.sleep(delay)
                now = time.time()
                current = np.mean(self.task.read(samples_per_channel=samples))
                count_rate = (current - prev) / (now - prev_t)
                values = {
                    'sweep_idx': sweep,
                    'v': f,
                    'y': count_rate*self.ratio,
                }
                self.sweep.acquire(values)
        return


    @sweep.initializer
    def initialize(self):
        params = self.run_parameters.widget.get()
        inp_ch = params['channel 1']
        if inp_ch in self.daq.counter_input_channels:
            self.task = CounterInputTask('TaskVsTime_CI')
            channel = CountEdgesChannel(inp_ch)
            self.task.add_channel(channel)
        else:
            # should never get here
            pass
        self.setup_pulses()
        volts = Q_(1, "V")
        self.awg.lower_limit[1] = -3. * volts
        self.awg.high_limit[1] = 3. * volts
        self.awg.limit[1] = "ON"
        self.awg.function[1] = "dc"
        self.awg.output[1] = "ON"
        self.awg.frequency = params["Frequency"]
        self.siggen.rf_amplitude = params["power"]
        self.siggen.rf_toggle = True
        self.task.start()
        return


    @sweep.finalizer
    def finalize(self):
        self.task.clear()
        self.awg.output[1] = "OFF"
        self.siggen.rf_toggle = False
        return

    def setup_pulses(self):
        params = self.run_parameters.widget.get()
        self.pulses.laser_time = int(params["excitation_time"].to("ns").magnitude)
        self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
        self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
        self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
        seq = self.pulses.Pulsed_ODMR(int(params["Pi"].to("ns").magnitude))
        self.ratio = self.pulses.total_time/self.pulses.readout_time
        self.pulses.stream(seq)

    @Element(name='Run parameters')
    def run_parameters(self):
        params = [
            ('counter 1', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr0',
            }),
            ('voltage', {
                'type': range,
                'units': "V",
                'default': {'func': 'linspace',
                            'start': -2.94,
                            'stop': 2.94,
                            'num': 197},
            }),
            ('Pi', {
                'type': float,
                'default': 50e-9,
                'suffix': ' ns',
                'units': "ns"
            }),
            ('Frequency', {
                'type': float,
                'default': 1.e9,
                'suffix': ' GHz',
                'units': "GHz"
            }),
            ('power', {
                'type': float,
                'default': -10,
            }),
            ('sweeps', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ('samples', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ('interpoint delay', {
                'type': float,
                'default': .1,
                'nonnegative': True,
            }),
            ('latest window', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ("excitation_time", {
                'type': float,
                'default': 600e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("readout_time", {
                'type': float,
                'default': 150e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("delay_time", {
                'type': float,
                'default': 2e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("reset_time", {
                'type': float,
                'default': 5e-6,
                'suffix': ' s',
                'units': "s"
            }),
        ]
        w = ParamWidget(params)
        return w

    @Element()
    def latest_history(self):
        p = LinePlotWidget()
        p.plot('channel 1', symbol=None)
        return p

    @latest_history.on(sweep.acquired)
    def latest_history_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        w.set('channel 1', xs=latest_data.v, ys=latest_data.y)
        return

    @Element()
    def averaged(self):
        p = LinePlotWidget()
        p.plot('channel 1', symbol=None)
        return p

    @averaged.on(sweep.acquired)
    def averaged_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        ys = grouped.y
        ys_averaged = ys.mean()
        w.set('channel 1', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys.std())
        return

    @Element()
    def save(self):
        w = RepositoryWidget(self)
        return w

    # @save.on(sweep.running)
    def save_requested(self, ev):
        w = ev.widget
        state, = ev.event_args
        if state:
            return
        print(self.dataset.data)
        w.save_dataset(self.dataset)
        return

# class RabiSpyrelet(Spyrelet):

#     requires = {
#         'daq': Device,
#         "awg": AG33522A,
#         "pulses": Pulses,
#         "siggen": SG396
#     }

#     @Task()
#     def sweep(self, **kwargs):
#         self.dataset.clear()
#         params = self.run_parameters.widget.get()
#         samples = params['samples']
#         delay = params['interpoint delay']
#         for sweep in range(params['sweeps']):
#             for i, f in enumerate(params['Delay'].array):
#                 self.pulses.stream(self.seqs[i])
#                 t0 = time.time()
#                 prev = np.mean(self.task.read(samples_per_channel=samples))
#                 prev_t = t0
#                 time.sleep(delay)
#                 now = time.time()
#                 current = np.mean(self.task.read(samples_per_channel=samples))
#                 count_rate = (current - prev) / (now - prev_t)
#                 values = {
#                     'sweep_idx': sweep,
#                     'v': f,
#                     'y': count_rate,
#                 }
#                 self.sweep.acquire(values)
#         return


#     @sweep.initializer
#     def initialize(self):
#         params = self.run_parameters.widget.get()
#         inp_ch = params['counter 1']
#         if inp_ch in self.daq.counter_input_channels:
#             self.task = CounterInputTask('TaskVsTime_CI')
#             channel = CountEdgesChannel(inp_ch)
#             self.task.add_channel(channel)
#         else:
#             # should never get here
#             pass
#         self.setup_pulses()
#         volts = Q_(1, "V")
#         self.awg.lower_limit[1] = -3. * volts
#         self.awg.high_limit[1] = 3. * volts
#         self.awg.limit[1] = "ON"
#         self.awg.function[1] = "dc"
#         self.awg.output[1] = "ON"
#         self.awg.offset[1] = params["voltage"]
#         self.siggen.rf_amplitude = params["power"]
#         self.siggen.frequency = params["Frequency"]
#         self.siggen.rf_toggle = True
#         self.task.start()
#         return


#     @sweep.finalizer
#     def finalize(self):
#         self.task.clear()
#         self.awg.output[1] = "OFF"
#         self.siggen.rf_toggle = False
#         return

#     def setup_pulses(self):
#         params = self.run_parameters.widget.get()
#         self.pulses.laser_time = int(params["excitation_time"].to("ns").magnitude)
#         self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
#         self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
#         self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
#         self.seqs = self.pulses.Rabi(params["Delay"])


#     @Element(name='Run parameters')
#     def run_parameters(self):
#         params = [
#             ('counter 1', {
#                 'type': list,
#                 'items': list(self.daq.counter_input_channels),
#                 'default': 'Dev1/ctr0',
#             }),
#             ('Delay', {
#                 'type': range,
#                 'units': "ns",
#                 'default': {'func': 'linspace',
#                             'start': 5e-9,
#                             'stop': 1005e-9,
#                             'num': 101},
#             }),
#             ('Frequency', {
#                 'type': float,
#                 'default': 6.707e9,
#                 'suffix': ' GHz',
#                 'units': "GHz"
#             }),
#             ('voltage', {
#                 'type': float,
#                 'default': 0,
#                 'suffix': ' V',
#                 'units': "V"
#             }),
#             ('power', {
#                 'type': float,
#                 'default': -10,
#             }),
#             ('sweeps', {
#                 'type': int,
#                 'default': 10,
#                 'positive': True,
#             }),
#             ('samples', {
#                 'type': int,
#                 'default': 100,
#                 'positive': True,
#             }),
#             ('interpoint delay', {
#                 'type': float,
#                 'default': 1,
#                 'nonnegative': True,
#             }),
#             ('latest window', {
#                 'type': int,
#                 'default': 100,
#                 'positive': True,
#             }),
#             ("excitation_time", {
#                 'type': float,
#                 'default': 600e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("readout_time", {
#                 'type': float,
#                 'default': 150e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("delay_time", {
#                 'type': float,
#                 'default': 2e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("reset_time", {
#                 'type': float,
#                 'default': 5e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#         ]
#         w = ParamWidget(params)
#         return w

#     @Element()
#     def latest_history(self):
#         p = LinePlotWidget()
#         p.plot('channel 1', symbol=None)
#         return p

#     @latest_history.on(sweep.acquired)
#     def latest_history_update(self, ev):
#         w = ev.widget
#         latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
#         w.set('channel 1', xs=latest_data.v, ys=latest_data.y)
#         return

#     @Element()
#     def averaged(self):
#         p = LinePlotWidget()
#         p.plot('channel 1', symbol=None)
#         return p

#     @averaged.on(sweep.acquired)
#     def averaged_update(self, ev):
#         w = ev.widget
#         grouped = self.data.groupby('v')
#         ys = grouped.y
#         ys_averaged = ys.mean()
#         w.set('channel 1', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys.std())
#         return

#     @Element()
#     def save(self):
#         w = RepositoryWidget(self)
#         return w

#     # @save.on(sweep.running)
#     def save_requested(self, ev):
#         w = ev.widget
#         state, = ev.event_args
#         if state:
#             return
#         print(self.dataset.data)
#         w.save_dataset(self.dataset)
#         return

# class L_ERabiSpyrelet(Spyrelet):

#     requires = {
#         'daq': Device,
#         "awg": AG33522A,
#         "pulses": Pulses,
#         "siggen": SG396
#     }

#     @Task()
#     def sweep(self, **kwargs):
#         self.dataset.clear()
#         params = self.run_parameters.widget.get()
#         for sweep in range(params['sweeps']):
#             for i, f in enumerate(params['Delay'].array):
#                 self.pulses.stream(self.seqs[i])
#                 ctrs_start = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
#                 time.sleep(params['interpoint delay'])
#                 ctrs_end = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
#                 dctrs = ctrs_end - ctrs_start
#                 ctrs_rates = dctrs[:,0] / dctrs[:,1]
#                 values = {
#                     'sweep_idx': sweep,
#                     'v': f,
#                     'x': ctrs_rates[0],
#                     'y': ctrs_rates[1],
#                 }
#                 self.sweep.acquire(values)
#         return


#     @sweep.initializer
#     def initialize(self):
#         params = self.run_parameters.widget.get()
#         ctrs = [params['counter 1'], params['counter 2']]
#         if len(set(ctrs)) != len(ctrs):
#             raise RuntimeError('counter channels 1 and 2 must be different')
#         self.ctr_tasks = list()
#         for idx, ctr in enumerate(ctrs):
#             task = CounterInputTask('counter ch {}'.format(idx))
#             ch = CountEdgesChannel(ctr)
#             task.add_channel(ch)
#             task.start()
#             self.ctr_tasks.append(task)
#         self.setup_pulses()
#         volts = Q_(1, "V")
#         self.awg.lower_limit[1] = -3. * volts
#         self.awg.high_limit[1] = 3. * volts
#         self.awg.limit[1] = "ON"
#         self.awg.function[1] = "dc"
#         self.awg.output[1] = "ON"
#         self.awg.offset[1] = params["voltage"]
#         self.siggen.rf_amplitude = params["power"]
#         self.siggen.frequency = params["Frequency"]
#         self.siggen.rf_toggle = True
#         return


#     @sweep.finalizer
#     def finalize(self):
#         for ctr_task in self.ctr_tasks:
#             ctr_task.clear()
#         self.awg.output[1] = "OFF"
#         self.siggen.rf_toggle = False
#         return

#     def setup_pulses(self):
#         params = self.run_parameters.widget.get()
#         self.pulses.laser_time = int(params["excitation_time"].to("ns").magnitude)
#         self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
#         self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
#         self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
#         self.pulses.polarize_time = int(params["Burn"].to("ns").magnitude)
#         self.pulses.settle = int(params["Settle"].to("ns").magnitude)
#         self.seqs = self.pulses.L_Experimental_Rabi(params["Delay"], cycle = params["Duty Cycle"])

#     @Element(name='Run parameters')
#     def run_parameters(self):
#         params = [
#             ('counter 1', {
#                 'type': list,
#                 'items': list(self.daq.counter_input_channels),
#                 'default': 'Dev1/ctr0',
#             }),
#             ('counter 2', {
#                 'type': list,
#                 'items': list(self.daq.counter_input_channels),
#                 'default': 'Dev1/ctr1',
#             }),
#             ('Delay', {
#                 'type': range,
#                 'units': "ns",
#                 'default': {'func': 'linspace',
#                             'start': 5e-9,
#                             'stop': 1005e-9,
#                             'num': 101},
#             }),
#             ('Duty Cycle', {
#                 'type': int,
#                 'default': 1,
#             }),
#             ('Frequency', {
#                 'type': float,
#                 'default': 6.707e9,
#                 'suffix': ' GHz',
#                 'units': "GHz"
#             }),
#             ('Burn', {
#                 'type': float,
#                 'default': 900e-6,
#                 'suffix': ' us',
#                 'units': "us"
#             }),
#             ('Settle', {
#                 'type': float,
#                 'default': 150e-6,
#                 'suffix': ' us',
#                 'units': "us"
#             }),
#             ('voltage', {
#                 'type': float,
#                 'default': 0,
#                 'suffix': ' V',
#                 'units': "V"
#             }),
#             ('power', {
#                 'type': float,
#                 'default': -10,
#             }),
#             ('sweeps', {
#                 'type': int,
#                 'default': 10,
#                 'positive': True,
#             }),
#             ('samples', {
#                 'type': int,
#                 'default': 100,
#                 'positive': True,
#             }),
#             ('interpoint delay', {
#                 'type': float,
#                 'default': 1,
#                 'nonnegative': True,
#             }),
#             ('latest window', {
#                 'type': int,
#                 'default': 100,
#                 'positive': True,
#             }),
#             ("excitation_time", {
#                 'type': float,
#                 'default': 600e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("readout_time", {
#                 'type': float,
#                 'default': 150e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("delay_time", {
#                 'type': float,
#                 'default': 2e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("reset_time", {
#                 'type': float,
#                 'default': 5e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#         ]
#         w = ParamWidget(params)
#         return w


#     @Element(name='Latest counts versus frequency')
#     def latest(self):
#         p = LinePlotWidget()
#         p.plot('Channel 1')
#         p.plot('Channel 2')
#         return p

#     @latest.on(sweep.acquired)
#     def latest_update(self, ev):
#         w = ev.widget
#         latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
#         w.set('Channel 1', xs=latest_data.v, ys=latest_data.x)
#         w.set('Channel 2', xs=latest_data.v, ys=latest_data.y)
#         return

#     @Element(name='Differential counts (latest)')
#     def diff_latest(self):
#         p = LinePlotWidget()
#         p.plot('Channel 1-2')
#         return p

#     @diff_latest.on(sweep.acquired)
#     def diff_latest_update(self, ev):
#         w = ev.widget
#         latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
#         diff = latest_data.x - latest_data.y
#         w.set('Channel 1-2', xs=latest_data.v, ys=diff)
#         return

#     @Element(name='Averaged counts versus frequency')
#     def averaged(self):
#         p = LinePlotWidget()
#         p.plot('Channel 1')
#         p.plot('Channel 2')
#         return p

#     @averaged.on(sweep.acquired)
#     def averaged_update(self, ev):
#         w = ev.widget
#         grouped = self.data.groupby('v')
#         xs = grouped.x
#         ys = grouped.y
#         xs_averaged = xs.mean()
#         ys_averaged = ys.mean()
#         w.set('Channel 1', xs=xs_averaged.index, ys=xs_averaged, yerrs=xs.std())
#         w.set('Channel 2', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys.std())
#         return

#     @Element(name='Differential counts (averaged)')
#     def diff_average(self):
#         p = LinePlotWidget()
#         p.plot('Channel 1-2')
#         return p

#     @diff_average.on(sweep.acquired)
#     def diff_average_update(self, ev):
#         w = ev.widget
#         grouped = self.data.groupby('v')
#         xs = grouped.x
#         ys = grouped.y
#         xs_averaged = xs.mean()
#         ys_averaged = ys.mean()
#         diff_averaged = xs_averaged - ys_averaged
#         w.set('Channel 1-2', xs=xs_averaged.index, ys=diff_averaged, yerrs=np.sqrt(xs.var() + ys.var()))
#         return

#     @Element()
#     def save(self):
#         w = RepositoryWidget(self)
#         return w

#     # @save.on(sweep.running)
#     def save_requested(self, ev):
#         w = ev.widget
#         state, = ev.event_args
#         if state:
#             return
#         print(self.dataset.data)
#         w.save_dataset(self.dataset)
#         return

class ResettingRabiSpyrelet(Spyrelet):

    requires = {
        'daq': Device,
        "awg": AG33522A,
        "pulses": Pulses,
        "siggen": SG396,
        "agil": E8257C
    }

    @Task()
    def sweep(self, **kwargs):
        self.dataset.clear()
        params = self.run_parameters.widget.get()
        for sweep in range(params['sweeps']):
            for i, f in enumerate(params['Delay'].array):
                self.pulses.stream(self.seqs[i])
                ctrs_start = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
                time.sleep(params['interpoint delay'])
                ctrs_end = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
                dctrs = ctrs_end - ctrs_start
                ctrs_rates = dctrs[:,0] / dctrs[:,1]
                values = {
                    'sweep_idx': sweep,
                    'v': f,
                    'x': ctrs_rates[0]*self.ratio,
                    'y': ctrs_rates[1]*self.ratio,
                    "diff": (ctrs_rates[0] - ctrs_rates[1])*self.ratio
                }
                self.sweep.acquire(values)
        return


    @sweep.initializer
    def initialize(self):
        params = self.run_parameters.widget.get()
        ctrs = [params['counter 1'], params['counter 2']]
        if len(set(ctrs)) != len(ctrs):
            raise RuntimeError('counter channels 1 and 2 must be different')
        self.ctr_tasks = list()
        for idx, ctr in enumerate(ctrs):
            task = CounterInputTask('counter ch {}'.format(idx))
            ch = CountEdgesChannel(ctr)
            task.add_channel(ch)
            task.start()
            self.ctr_tasks.append(task)
        self.setup_pulses()
        volts = Q_(1, "V")
        self.awg.lower_limit[1] = -3. * volts
        self.awg.high_limit[1] = 3. * volts
        self.awg.limit[1] = "ON"
        self.awg.function[1] = "dc"
        self.awg.output[1] = "ON"
        self.awg.offset[1] = params["voltage"]

        self.siggen.rf_amplitude = params["power"]
        self.siggen.frequency = params["Frequency"]
        self.siggen.rf_toggle = True
        
        if params["EOM"]:
            self.agil.rf_amplitude = params["EOM power"]
            self.agil.rf_frequency = params["EOM Frequency"]
            self.agil.rf_toggle = True
        else:
            self.agil.rf_toggle = False
        return


    @sweep.finalizer
    def finalize(self):
        for ctr_task in self.ctr_tasks:
            ctr_task.clear()
        self.awg.output[1] = "OFF"
        self.siggen.rf_toggle = False
        return

    def setup_pulses(self):
        params = self.run_parameters.widget.get()
        self.pulses.laser_time = int(params["probe_time"].to("ns").magnitude)
        self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
        self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
        self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
        self.pulses.polarize_time = int(params["Burn"].to("ns").magnitude)
        self.pulses.settle = int(params["Settle"].to("ns").magnitude)
        self.pulses.reset = int(params["Reset"].to("ns").magnitude)
        self.seqs = self.pulses.Resetting_L_Rabi(params["Delay"])
        self.ratio = 2 * self.pulses.total_time / self.pulses.readout_time

    @Element(name='Run parameters')
    def run_parameters(self):
        params = [
            ('counter 1', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr0',
            }),
            ('counter 2', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr1',
            }),
            ('EOM', {
                'type': bool,
                'default': True,
            }),
            ('EOM Frequency', {
                'type': float,
                'default': 1.064e9,
                'suffix': ' GHz',
                'units': "GHz"
            }),
            ('EOM power', {
                'type': float,
                'default': 16,
            }),
            ('Delay', {
                'type': range,
                'units': "ns",
                'default': {'func': 'linspace',
                            'start': 5e-9,
                            'stop': 1005e-9,
                            'num': 101},
            }),
            ('Frequency', {
                'type': float,
                'default': 6.707e9,
                'suffix': ' GHz',
                'units': "GHz"
            }),
            ('Reset', {
                'type': float,
                'default': 1e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('Burn', {
                'type': float,
                'default': 900e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('Settle', {
                'type': float,
                'default': 150e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('voltage', {
                'type': float,
                'default': 0,
                'suffix': ' V',
                'units': "V"
            }),
            ('power', {
                'type': float,
                'default': -10,
            }),
            ('sweeps', {
                'type': int,
                'default': 10,
                'positive': True,
            }),
            ('samples', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ('interpoint delay', {
                'type': float,
                'default': 1,
                'nonnegative': True,
            }),
            ('latest window', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ("probe_time", {
                'type': float,
                'default': 50e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("readout_time", {
                'type': float,
                'default': 150e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("delay_time", {
                'type': float,
                'default': 2e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("reset_time", {
                'type': float,
                'default': 5e-6,
                'suffix': ' s',
                'units': "s"
            }),
        ]
        w = ParamWidget(params)
        return w


    @Element(name='Latest counts versus frequency')
    def latest(self):
        p = LinePlotWidget()
        p.plot('Channel 1')
        p.plot('Channel 2')
        return p

    @latest.on(sweep.acquired)
    def latest_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        w.set('Channel 1', xs=latest_data.v, ys=latest_data.x)
        w.set('Channel 2', xs=latest_data.v, ys=latest_data.y)
        return

    @Element(name='Differential counts (latest)')
    def diff_latest(self):
        p = LinePlotWidget()
        p.plot('Channel 1-2')
        return p

    @diff_latest.on(sweep.acquired)
    def diff_latest_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        diff = latest_data.x - latest_data.y
        w.set('Channel 1-2', xs=latest_data.v, ys=diff)
        return

    @Element(name='Averaged counts versus frequency')
    def averaged(self):
        p = LinePlotWidget()
        p.plot('Channel 1')
        p.plot('Channel 2')
        return p

    @averaged.on(sweep.acquired)
    def averaged_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        xs = grouped.x
        ys = grouped.y
        xs_averaged = xs.mean()
        ys_averaged = ys.mean()
        w.set('Channel 1', xs=xs_averaged.index, ys=xs_averaged, yerrs=xs.std())
        w.set('Channel 2', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys.std())
        return

    @Element(name='Differential counts (averaged)')
    def diff_average(self):
        p = LinePlotWidget()
        p.plot('Channel 1-2')
        return p

    @diff_average.on(sweep.acquired)
    def diff_average_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        xs = grouped.diff
        xs_averaged = xs.mean()
        w.set('Channel 1-2', xs=xs_averaged.index, ys=xs_averaged, yerrs=xs_averaged)
        return

    @Element()
    def save(self):
        w = RepositoryWidget(self)
        return w

    # @save.on(sweep.running)
    def save_requested(self, ev):
        w = ev.widget
        state, = ev.event_args
        if state:
            return
        print(self.dataset.data)
        w.save_dataset(self.dataset)
        return

# class ERabiSpyrelet(Spyrelet):

#     requires = {
#         'daq': Device,
#         "awg": AG33522A,
#         "pulses": Pulses,
#         "siggen": SG396
#     }

#     def count(self):
#         run_params = self.run_parameters.widget.get()
#         samples = run_params['samples']
#         delay = run_params['interpoint delay']
#         t0 = time.time()
#         prev = np.mean(self.task.read(samples_per_channel=samples))
#         prev_t = t0
#         time.sleep(delay)
#         now = time.time()
#         current = np.mean(self.task.read(samples_per_channel=samples))
#         count_rate = (current - prev) / (now - prev_t)
#         t = now - t0
#         return t, count_rate

#     @Task()
#     def sweep(self, **kwargs):
#         self.dataset.clear()
#         params = self.run_parameters.widget.get()
#         for sweep in range(params['sweeps']):
#             for i, f in enumerate(params['Delay'].array):
#                 self.pulses.stream(self.seqs[i])
#                 t, y = self.count()
#                 values = {
#                     'sweep_idx': sweep,
#                     'v': f,
#                     'y': y,
#                 }
#                 self.sweep.acquire(values)
#         return


#     @sweep.initializer
#     def initialize(self):
#         params = self.input_parameters.widget.get()
#         params2 = self.run_parameters.widget.get()
#         inp_ch = params['channel 1']
#         if inp_ch in self.daq.counter_input_channels:
#             self.task = CounterInputTask('TaskVsTime_CI')
#             channel = CountEdgesChannel(inp_ch)
#             self.task.add_channel(channel)
#         else:
#             # should never get here
#             pass
#         self.setup_pulses()
#         volts = Q_(1, "V")
#         self.awg.lower_limit[1] = -3. * volts
#         self.awg.high_limit[1] = 3. * volts
#         self.awg.limit[1] = "ON"
#         self.awg.function[1] = "dc"
#         self.awg.output[1] = "ON"
#         self.awg.offset[1] = params2["voltage"]
#         self.siggen.rf_amplitude = params2["power"]
#         self.siggen.frequency = params2["Frequency"]
#         self.siggen.rf_toggle = True
#         self.task.start()
#         return


#     @sweep.finalizer
#     def finalize(self):
#         self.task.clear()
#         self.awg.output[1] = "OFF"
#         self.siggen.rf_toggle = False
#         return

#     def setup_pulses(self):
#         params = self.run_parameters.widget.get()
#         self.pulses.laser_time = int(params["excitation_time"].to("ns").magnitude)
#         self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
#         self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
#         self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
#         self.pulses.polarize_time = int(params["Burn"].to("ns").magnitude)
#         self.pulses.settle = int(params["Settle"].to("ns").magnitude)
#         self.seqs = self.pulses.Experimental_Rabi(params["Delay"])

#     @Element(name='Input parameters')
#     def input_parameters(self):
#         channels = 4
#         comboboxes = [pg.ComboBox() for _ in range(channels)]
#         ci_channels = list(self.daq.counter_input_channels)
#         none_channel = ['None']
#         all_channels = none_channel + ci_channels
#         params = list()
#         for idx, combobox in enumerate(comboboxes):
#             channel_param = ('channel {}'.format(idx + 1), {
#                 'type': list,
#                 'items': all_channels,
#             })
#             params.append(channel_param)
#         w = ParamWidget(params)
#         return w

#     @Element(name='Run parameters')
#     def run_parameters(self):
#         params = [
#             ('Delay', {
#                 'type': range,
#                 'units': "ns",
#                 'default': {'func': 'linspace',
#                             'start': 5e-9,
#                             'stop': 1005e-9,
#                             'num': 101},
#             }),
#             ('Frequency', {
#                 'type': float,
#                 'default': 6.707e9,
#                 'suffix': ' GHz',
#                 'units': "GHz"
#             }),
#             ('Burn', {
#                 'type': float,
#                 'default': 900e-6,
#                 'suffix': ' us',
#                 'units': "us"
#             }),
#             ('Settle', {
#                 'type': float,
#                 'default': 150e-6,
#                 'suffix': ' us',
#                 'units': "us"
#             }),
#             ('voltage', {
#                 'type': float,
#                 'default': 0,
#                 'suffix': ' V',
#                 'units': "V"
#             }),
#             ('power', {
#                 'type': float,
#                 'default': -10,
#             }),
#             ('sweeps', {
#                 'type': int,
#                 'default': 10,
#                 'positive': True,
#             }),
#             ('samples', {
#                 'type': int,
#                 'default': 100,
#                 'positive': True,
#             }),
#             ('interpoint delay', {
#                 'type': float,
#                 'default': 1,
#                 'nonnegative': True,
#             }),
#             ('latest window', {
#                 'type': int,
#                 'default': 100,
#                 'positive': True,
#             }),
#             ("excitation_time", {
#                 'type': float,
#                 'default': 600e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("readout_time", {
#                 'type': float,
#                 'default': 150e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("delay_time", {
#                 'type': float,
#                 'default': 2e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("reset_time", {
#                 'type': float,
#                 'default': 5e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#         ]
#         w = ParamWidget(params)
#         return w

#     @Element()
#     def latest_history(self):
#         p = LinePlotWidget()
#         p.plot('channel 1', symbol=None)
#         return p

#     @latest_history.on(sweep.acquired)
#     def latest_history_update(self, ev):
#         w = ev.widget
#         latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
#         w.set('channel 1', xs=latest_data.v, ys=latest_data.y)
#         return

#     @Element()
#     def averaged(self):
#         p = LinePlotWidget()
#         p.plot('channel 1', symbol=None)
#         return p

#     @averaged.on(sweep.acquired)
#     def averaged_update(self, ev):
#         w = ev.widget
#         grouped = self.data.groupby('v')
#         ys = grouped.y
#         ys_averaged = ys.mean()
#         w.set('channel 1', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys.std())
#         return

#     @Element()
#     def save(self):
#         w = RepositoryWidget(self)
#         return w

#     # @save.on(sweep.running)
#     def save_requested(self, ev):
#         w = ev.widget
#         state, = ev.event_args
#         if state:
#             return
#         print(self.dataset.data)
#         w.save_dataset(self.dataset)
#         return

# class RamseySpyrelet(Spyrelet):

#     requires = {
#         'daq': Device,
#         "awg": AG33522A,
#         "pulses": Pulses,
#         "siggen": SG396
#     }

#     def count(self):
#         run_params = self.run_parameters.widget.get()
#         samples = run_params['samples']
#         delay = run_params['interpoint delay']
#         t0 = time.time()
#         prev = np.mean(self.task.read(samples_per_channel=samples))
#         prev_t = t0
#         time.sleep(delay)
#         now = time.time()
#         current = np.mean(self.task.read(samples_per_channel=samples))
#         count_rate = (current - prev) / (now - prev_t)
#         t = now - t0
#         return t, count_rate

#     @Task()
#     def sweep(self, **kwargs):
#         self.dataset.clear()
#         params = self.run_parameters.widget.get()
#         for sweep in range(params['sweeps']):
#             for i, f in enumerate(params['Delay'].array):
#                 self.pulses.stream(self.seqs[i])
#                 t, y = self.count()
#                 values = {
#                     'sweep_idx': sweep,
#                     'v': f,
#                     'y': y,
#                 }
#                 self.sweep.acquire(values)
#         return


#     @sweep.initializer
#     def initialize(self):
#         params = self.input_parameters.widget.get()
#         params2 = self.run_parameters.widget.get()
#         inp_ch = params['channel 1']
#         if inp_ch in self.daq.counter_input_channels:
#             self.task = CounterInputTask('TaskVsTime_CI')
#             channel = CountEdgesChannel(inp_ch)
#             self.task.add_channel(channel)
#         else:
#             # should never get here
#             pass
#         self.setup_pulses()
#         volts = Q_(1, "V")
#         self.awg.lower_limit[1] = -3. * volts
#         self.awg.high_limit[1] = 3. * volts
#         self.awg.limit[1] = "ON"
#         self.awg.function[1] = "dc"
#         self.awg.output[1] = "ON"
#         self.awg.offset[1] = params2["voltage"]
#         self.siggen.rf_amplitude = params2["power"]
#         self.siggen.frequency = params2["Frequency"]
#         self.siggen.rf_toggle = True
#         self.task.start()
#         return


#     @sweep.finalizer
#     def finalize(self):
#         self.task.clear()
#         self.awg.output[1] = "OFF"
#         self.siggen.rf_toggle = False
#         return

#     def setup_pulses(self):
#         params = self.run_parameters.widget.get()
#         self.pulses.laser_time = int(params["excitation_time"].to("ns").magnitude)
#         self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
#         self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
#         self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
#         self.seqs = self.pulses.Ramsey(params["Delay"], params["Pi"])

#     @Element(name='Input parameters')
#     def input_parameters(self):
#         channels = 4
#         comboboxes = [pg.ComboBox() for _ in range(channels)]
#         ci_channels = list(self.daq.counter_input_channels)
#         none_channel = ['None']
#         all_channels = none_channel + ci_channels
#         params = list()
#         for idx, combobox in enumerate(comboboxes):
#             channel_param = ('channel {}'.format(idx + 1), {
#                 'type': list,
#                 'items': all_channels,
#             })
#             params.append(channel_param)
#         w = ParamWidget(params)
#         return w

#     @Element(name='Run parameters')
#     def run_parameters(self):
#         params = [
#             ('Delay', {
#                 'type': range,
#                 'units': "ns",
#                 'default': {'func': 'linspace',
#                             'start': 5e-9,
#                             'stop': 1005e-9,
#                             'num': 101},
#             }),
#             ('Pi', {
#                 'type': float,
#                 'default': 50e-9,
#                 'suffix': ' ns',
#                 'units': "ns"
#             }),
#             ('Frequency', {
#                 'type': float,
#                 'default': 6.707e9,
#                 'suffix': ' GHz',
#                 'units': "GHz"
#             }),
#             ('voltage', {
#                 'type': float,
#                 'default': 0,
#                 'suffix': ' V',
#                 'units': "V"
#             }),
#             ('power', {
#                 'type': float,
#                 'default': -10,
#             }),
#             ('sweeps', {
#                 'type': int,
#                 'default': 10,
#                 'positive': True,
#             }),
#             ('samples', {
#                 'type': int,
#                 'default': 100,
#                 'positive': True,
#             }),
#             ('interpoint delay', {
#                 'type': float,
#                 'default': 1,
#                 'nonnegative': True,
#             }),
#             ('latest window', {
#                 'type': int,
#                 'default': 100,
#                 'positive': True,
#             }),
#             ("excitation_time", {
#                 'type': float,
#                 'default': 600e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("readout_time", {
#                 'type': float,
#                 'default': 150e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("delay_time", {
#                 'type': float,
#                 'default': 2e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("reset_time", {
#                 'type': float,
#                 'default': 5e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#         ]
#         w = ParamWidget(params)
#         return w

#     @Element()
#     def latest_history(self):
#         p = LinePlotWidget()
#         p.plot('channel 1', symbol=None)
#         return p

#     @latest_history.on(sweep.acquired)
#     def latest_history_update(self, ev):
#         w = ev.widget
#         latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
#         w.set('channel 1', xs=latest_data.v, ys=latest_data.y)
#         return

#     @Element()
#     def averaged(self):
#         p = LinePlotWidget()
#         p.plot('channel 1', symbol=None)
#         return p

#     @averaged.on(sweep.acquired)
#     def averaged_update(self, ev):
#         w = ev.widget
#         grouped = self.data.groupby('v')
#         ys = grouped.y
#         ys_averaged = ys.mean()
#         w.set('channel 1', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys.std())
#         return

#     @Element()
#     def save(self):
#         w = RepositoryWidget(self)
#         return w

#     # @save.on(sweep.running)
#     def save_requested(self, ev):
#         w = ev.widget
#         state, = ev.event_args
#         if state:
#             return
#         print(self.dataset.data)
#         w.save_dataset(self.dataset)
#         return

class ResettingRamseySpyrelet(Spyrelet):

    requires = {
        'daq': Device,
        "awg": AG33522A,
        "pulses": Pulses,
        "siggen": SG396,
        "agil": E8257C
    }

    @Task()
    def sweep(self, **kwargs):
        self.dataset.clear()
        params = self.run_parameters.widget.get()
        for sweep in range(params['sweeps']):
            for i, f in enumerate(params['Delay'].array):
                self.pulses.stream(self.seqs[i])
                ctrs_start = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
                time.sleep(params['interpoint delay'])
                ctrs_end = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
                dctrs = ctrs_end - ctrs_start
                ctrs_rates = dctrs[:,0] / dctrs[:,1]
                values = {
                    'sweep_idx': sweep,
                    'v': f,
                    'v': f,
                    'x': ctrs_rates[0]*self.ratio,
                    'y': ctrs_rates[1]*self.ratio,
                    "diff": (ctrs_rates[0] - ctrs_rates[1])*self.ratio
                }
                self.sweep.acquire(values)
        return


    @sweep.initializer
    def initialize(self):
        params = self.run_parameters.widget.get()
        ctrs = [params['counter 1'], params['counter 2']]
        if len(set(ctrs)) != len(ctrs):
            raise RuntimeError('counter channels 1 and 2 must be different')
        self.ctr_tasks = list()
        for idx, ctr in enumerate(ctrs):
            task = CounterInputTask('counter ch {}'.format(idx))
            ch = CountEdgesChannel(ctr)
            task.add_channel(ch)
            task.start()
            self.ctr_tasks.append(task)
        self.setup_pulses()
        volts = Q_(1, "V")
        self.awg.lower_limit[1] = -3. * volts
        self.awg.high_limit[1] = 3. * volts
        self.awg.limit[1] = "ON"
        self.awg.function[1] = "dc"
        self.awg.output[1] = "ON"
        self.awg.offset[1] = params["voltage"]
        self.siggen.rf_amplitude = params["power"]
        self.siggen.frequency = params["Frequency"]
        self.siggen.rf_toggle = True

        if params["EOM"]:
            self.agil.rf_amplitude = params["EOM power"]
            self.agil.rf_frequency = params["EOM Frequency"]
            self.agil.rf_toggle = True
        else:
            self.agil.rf_toggle = False

        return


    @sweep.finalizer
    def finalize(self):
        for ctr_task in self.ctr_tasks:
            ctr_task.clear()
        self.awg.output[1] = "OFF"
        self.siggen.rf_toggle = False
        return

    def setup_pulses(self):
        params = self.run_parameters.widget.get()
        self.pulses.laser_time = int(params["probe_time"].to("ns").magnitude)
        self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
        self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
        self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
        self.pulses.polarize_time = int(params["Burn"].to("ns").magnitude)
        self.pulses.settle = int(params["Settle"].to("ns").magnitude)
        self.pulses.reset = int(params["Reset"].to("ns").magnitude)
        self.seqs = self.pulses.Resetting_L_Ramsey(params["Delay"], pi = params["Pi"])
        self.ratio = 2 * self.pulses.total_time / self.pulses.readout_time

    @Element(name='Run parameters')
    def run_parameters(self):
        params = [
            ('counter 1', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr0',
            }),
            ('counter 2', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr1',
            }),
            ('EOM', {
                'type': bool,
                'default': True,
            }),
            ('EOM Frequency', {
                'type': float,
                'default': 1.064e9,
                'suffix': ' GHz',
                'units': "GHz"
            }),
            ('EOM power', {
                'type': float,
                'default': 16,
            }),
            ('Delay', {
                'type': range,
                'units': "ns",
                'default': {'func': 'linspace',
                            'start': 5e-9,
                            'stop': 1005e-9,
                            'num': 101},
            }),
            ('Pi', {
                'type': float,
                'default': 140e-9,
                'suffix': ' ns',
                'units': "ns"
            }),
            ('Frequency', {
                'type': float,
                'default': 6.707e9,
                'suffix': ' GHz',
                'units': "GHz"
            }),
            ('Reset', {
                'type': float,
                'default': 1e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('Burn', {
                'type': float,
                'default': 900e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('Settle', {
                'type': float,
                'default': 150e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('voltage', {
                'type': float,
                'default': 0,
                'suffix': ' V',
                'units': "V"
            }),
            ('power', {
                'type': float,
                'default': -10,
            }),
            ('sweeps', {
                'type': int,
                'default': 10,
                'positive': True,
            }),
            ('samples', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ('interpoint delay', {
                'type': float,
                'default': 1,
                'nonnegative': True,
            }),
            ('latest window', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ("probe_time", {
                'type': float,
                'default': 50e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("readout_time", {
                'type': float,
                'default': 150e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("delay_time", {
                'type': float,
                'default': 2e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("reset_time", {
                'type': float,
                'default': 5e-6,
                'suffix': ' s',
                'units': "s"
            }),
        ]
        w = ParamWidget(params)
        return w


    @Element(name='Latest counts versus frequency')
    def latest(self):
        p = LinePlotWidget()
        p.plot('Channel 1')
        p.plot('Channel 2')
        return p

    @latest.on(sweep.acquired)
    def latest_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        w.set('Channel 1', xs=latest_data.v, ys=latest_data.x)
        w.set('Channel 2', xs=latest_data.v, ys=latest_data.y)
        return

    @Element(name='Differential counts (latest)')
    def diff_latest(self):
        p = LinePlotWidget()
        p.plot('Channel 1-2')
        return p

    @diff_latest.on(sweep.acquired)
    def diff_latest_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        diff = latest_data.x - latest_data.y
        w.set('Channel 1-2', xs=latest_data.v, ys=diff)
        return

    @Element(name='Averaged counts versus frequency')
    def averaged(self):
        p = LinePlotWidget()
        p.plot('Channel 1')
        p.plot('Channel 2')
        return p

    @averaged.on(sweep.acquired)
    def averaged_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        xs = grouped.x
        ys = grouped.y
        xs_averaged = xs.mean()
        ys_averaged = ys.mean()
        w.set('Channel 1', xs=xs_averaged.index, ys=xs_averaged, yerrs=xs.std())
        w.set('Channel 2', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys.std())
        return

    @Element(name='Differential counts (averaged)')
    def diff_average(self):
        p = LinePlotWidget()
        p.plot('Channel 1-2')
        return p

    @diff_average.on(sweep.acquired)
    def diff_average_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        xs = grouped.diff
        xs_averaged = xs.mean()
        w.set('Channel 1-2', xs=xs_averaged.index, ys=xs_averaged, yerrs=xs_averaged)
        return

    @Element()
    def save(self):
        w = RepositoryWidget(self)
        return w

    # @save.on(sweep.running)
    def save_requested(self, ev):
        w = ev.widget
        state, = ev.event_args
        if state:
            return
        print(self.dataset.data)
        w.save_dataset(self.dataset)
        return

# class T2Spyrelet(Spyrelet):

#     requires = {
#         'daq': Device,
#         "awg": AG33522A,
#         "pulses": Pulses,
#         "siggen": SG396
#     }

#     def count(self):
#         run_params = self.run_parameters.widget.get()
#         samples = run_params['samples']
#         delay = run_params['interpoint delay']
#         t0 = time.time()
#         prev = np.mean(self.task.read(samples_per_channel=samples))
#         prev_t = t0
#         time.sleep(delay)
#         now = time.time()
#         current = np.mean(self.task.read(samples_per_channel=samples))
#         count_rate = (current - prev) / (now - prev_t)
#         t = now - t0
#         return t, count_rate

#     @Task()
#     def sweep(self, **kwargs):
#         self.dataset.clear()
#         params = self.run_parameters.widget.get()
#         for sweep in range(params['sweeps']):
#             for i, f in enumerate(params['Delay'].array):
#                 self.pulses.stream(self.seqs[i])
#                 t, y = self.count()
#                 values = {
#                     'sweep_idx': sweep,
#                     'v': f,
#                     'y': y,
#                 }
#                 self.sweep.acquire(values)
#         return


#     @sweep.initializer
#     def initialize(self):
#         params = self.input_parameters.widget.get()
#         params2 = self.run_parameters.widget.get()
#         inp_ch = params['channel 1']
#         if inp_ch in self.daq.counter_input_channels:
#             self.task = CounterInputTask('TaskVsTime_CI')
#             channel = CountEdgesChannel(inp_ch)
#             self.task.add_channel(channel)
#         else:
#             # should never get here
#             pass
#         self.setup_pulses()
#         volts = Q_(1, "V")
#         self.awg.lower_limit[1] = -3. * volts
#         self.awg.high_limit[1] = 3. * volts
#         self.awg.limit[1] = "ON"
#         self.awg.function[1] = "dc"
#         self.awg.output[1] = "ON"
#         self.awg.offset[1] = params2["voltage"]
#         self.siggen.rf_amplitude = params2["power"]
#         self.siggen.frequency = params2["Frequency"]
#         self.siggen.rf_toggle = True
#         self.task.start()
#         return


#     @sweep.finalizer
#     def finalize(self):
#         self.task.clear()
#         self.awg.output[1] = "OFF"
#         self.siggen.rf_toggle = False
#         return

#     def setup_pulses(self):
#         params = self.run_parameters.widget.get()
#         self.pulses.laser_time = int(params["excitation_time"].to("ns").magnitude)
#         self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
#         self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
#         self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
#         self.seqs = self.pulses.T2(params["Delay"], params["Pi"])

#     @Element(name='Input parameters')
#     def input_parameters(self):
#         channels = 4
#         comboboxes = [pg.ComboBox() for _ in range(channels)]
#         ci_channels = list(self.daq.counter_input_channels)
#         none_channel = ['None']
#         all_channels = none_channel + ci_channels
#         params = list()
#         for idx, combobox in enumerate(comboboxes):
#             channel_param = ('channel {}'.format(idx + 1), {
#                 'type': list,
#                 'items': all_channels,
#             })
#             params.append(channel_param)
#         w = ParamWidget(params)
#         return w

#     @Element(name='Run parameters')
#     def run_parameters(self):
#         params = [
#             ('Delay', {
#                 'type': range,
#                 'units': "ns",
#                 'default': {'func': 'linspace',
#                             'start': 5e-9,
#                             'stop': 1005e-9,
#                             'num': 101},
#             }),
#             ('Pi', {
#                 'type': float,
#                 'default': 50e-9,
#                 'suffix': ' ns',
#                 'units': "ns"
#             }),
#             ('Frequency', {
#                 'type': float,
#                 'default': 6.707e9,
#                 'suffix': ' GHz',
#                 'units': "GHz"
#             }),
#             ('voltage', {
#                 'type': float,
#                 'default': 0,
#                 'suffix': ' V',
#                 'units': "V"
#             }),
#             ('power', {
#                 'type': float,
#                 'default': -10,
#             }),
#             ('sweeps', {
#                 'type': int,
#                 'default': 10,
#                 'positive': True,
#             }),
#             ('samples', {
#                 'type': int,
#                 'default': 100,
#                 'positive': True,
#             }),
#             ('interpoint delay', {
#                 'type': float,
#                 'default': 1,
#                 'nonnegative': True,
#             }),
#             ('latest window', {
#                 'type': int,
#                 'default': 100,
#                 'positive': True,
#             }),
#             ("excitation_time", {
#                 'type': float,
#                 'default': 600e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("readout_time", {
#                 'type': float,
#                 'default': 150e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("delay_time", {
#                 'type': float,
#                 'default': 2e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("reset_time", {
#                 'type': float,
#                 'default': 5e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#         ]
#         w = ParamWidget(params)
#         return w

#     @Element()
#     def latest_history(self):
#         p = LinePlotWidget()
#         p.plot('channel 1', symbol=None)
#         return p

#     @latest_history.on(sweep.acquired)
#     def latest_history_update(self, ev):
#         w = ev.widget
#         latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
#         w.set('channel 1', xs=latest_data.v, ys=latest_data.y)
#         return

#     @Element()
#     def averaged(self):
#         p = LinePlotWidget()
#         p.plot('channel 1', symbol=None)
#         return p

#     @averaged.on(sweep.acquired)
#     def averaged_update(self, ev):
#         w = ev.widget
#         grouped = self.data.groupby('v')
#         ys = grouped.y
#         ys_averaged = ys.mean()
#         w.set('channel 1', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys.std())
#         return

#     @Element()
#     def save(self):
#         w = RepositoryWidget(self)
#         return w

#     # @save.on(sweep.running)
#     def save_requested(self, ev):
#         w = ev.widget
#         state, = ev.event_args
#         if state:
#             return
#         print(self.dataset.data)
#         w.save_dataset(self.dataset)
#         return

class ResettingT2Spyrelet(Spyrelet):

    requires = {
        'daq': Device,
        "awg": AG33522A,
        "pulses": Pulses,
        "siggen": SG396,
        "agil": E8257C
    }

    @Task()
    def sweep(self, **kwargs):
        self.dataset.clear()
        params = self.run_parameters.widget.get()
        for sweep in range(params['sweeps']):
            for i, f in enumerate(params['Delay'].array):
                self.pulses.stream(self.seqs[i])
                ctrs_start = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
                time.sleep(params['interpoint delay'])
                ctrs_end = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
                dctrs = ctrs_end - ctrs_start
                ctrs_rates = dctrs[:,0] / dctrs[:,1]
                values = {
                    'sweep_idx': sweep,
                    'v': f,
                    'x': ctrs_rates[0]*self.ratio,
                    'y': ctrs_rates[1]*self.ratio,
                    "diff": (ctrs_rates[0] - ctrs_rates[1])*self.ratio
                }
                self.sweep.acquire(values)
        return


    @sweep.initializer
    def initialize(self):
        params = self.run_parameters.widget.get()
        ctrs = [params['counter 1'], params['counter 2']]
        if len(set(ctrs)) != len(ctrs):
            raise RuntimeError('counter channels 1 and 2 must be different')
        self.ctr_tasks = list()
        for idx, ctr in enumerate(ctrs):
            task = CounterInputTask('counter ch {}'.format(idx))
            ch = CountEdgesChannel(ctr)
            task.add_channel(ch)
            task.start()
            self.ctr_tasks.append(task)
        self.setup_pulses()
        volts = Q_(1, "V")
        self.awg.lower_limit[1] = -3. * volts
        self.awg.high_limit[1] = 3. * volts
        self.awg.limit[1] = "ON"
        self.awg.function[1] = "dc"
        self.awg.output[1] = "ON"
        self.awg.offset[1] = params["voltage"]
        self.siggen.rf_amplitude = params["power"]
        self.siggen.frequency = params["Frequency"]
        self.siggen.rf_toggle = True

        if params["EOM"]:
            self.agil.rf_amplitude = params["EOM power"]
            self.agil.rf_frequency = params["EOM Frequency"]
            self.agil.rf_toggle = True
        else:
            self.agil.rf_toggle = False
        return


    @sweep.finalizer
    def finalize(self):
        for ctr_task in self.ctr_tasks:
            ctr_task.clear()
        self.awg.output[1] = "OFF"
        self.siggen.rf_toggle = False
        return

    def setup_pulses(self):
        params = self.run_parameters.widget.get()
        self.pulses.laser_time = int(params["probe_time"].to("ns").magnitude)
        self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
        self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
        self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
        self.pulses.polarize_time = int(params["Burn"].to("ns").magnitude)
        self.pulses.settle = int(params["Settle"].to("ns").magnitude)
        self.pulses.reset = int(params["Reset"].to("ns").magnitude)
        if params["Style"] == "Absolute":
            self.seqs = self.pulses.Resetting_T2(params["Delay"], pi = params["Pi"])
            self.ratio = self.pulses.total_time / self.pulses.readout_time
        elif params["Style"] == "Lockin":
            self.seqs = self.pulses.Resetting_L_T2(params["Delay"], pi = params["Pi"], style=[[0, 0], [0, 0], [0, 0]])
            self.ratio = 2 * self.pulses.total_time / self.pulses.readout_time
        elif params["Style"] == "GWS":
            self.seqs = self.pulses.Resetting_L_T2(params["Delay"], pi = params["Pi"], style=[[-1, 0], [1, 0], [1, 0]])
            self.ratio = 2 * self.pulses.total_time / self.pulses.readout_time
        elif params["Style"] == "Y":
            self.seqs = self.pulses.Resetting_L_Y_T2(params["Delay"], pi = params["Pi"])
            self.ratio = 2 * self.pulses.total_time / self.pulses.readout_time

    @Element(name='Run parameters')
    def run_parameters(self):
        params = [
            ('counter 1', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr0',
            }),
            ('counter 2', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr1',
            }),
            ('Style', {
                'type': list,
                'items': ["Absolute", "Lockin", "GWS", "Y"],
                'default': 'Absolute',
            }),
            ('EOM', {
                'type': bool,
                'default': True,
            }),
            ('EOM Frequency', {
                'type': float,
                'default': 1.064e9,
                'suffix': ' GHz',
                'units': "GHz"
            }),
            ('EOM power', {
                'type': float,
                'default': 16,
            }),
            ('Delay', {
                'type': range,
                'units': "us",
                'default': {'func': 'linspace',
                            'start': 0e-9,
                            'stop': 5e-6,
                            'num': 1001},
            }),
            ('Pi', {
                'type': float,
                'default': 140e-9,
                'suffix': ' ns',
                'units': "ns"
            }),
            ('Frequency', {
                'type': float,
                'default': 6.707e9,
                'suffix': ' GHz',
                'units': "GHz"
            }),
            ('Reset', {
                'type': float,
                'default': 150e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('Burn', {
                'type': float,
                'default': 900e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('Settle', {
                'type': float,
                'default': 150e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('voltage', {
                'type': float,
                'default': 0,
                'suffix': ' V',
                'units': "V"
            }),
            ('power', {
                'type': float,
                'default': -10,
            }),
            ('sweeps', {
                'type': int,
                'default': 10,
                'positive': True,
            }),
            ('samples', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ('interpoint delay', {
                'type': float,
                'default': 1,
                'nonnegative': True,
            }),
            ('latest window', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ("probe_time", {
                'type': float,
                'default': 50e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("readout_time", {
                'type': float,
                'default': 150e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("delay_time", {
                'type': float,
                'default': 2e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("reset_time", {
                'type': float,
                'default': 5e-6,
                'suffix': ' s',
                'units': "s"
            }),
        ]
        w = ParamWidget(params)
        return w


    @Element(name='Latest counts versus frequency')
    def latest(self):
        p = LinePlotWidget()
        p.plot('Channel 1')
        p.plot('Channel 2')
        return p

    @latest.on(sweep.acquired)
    def latest_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        w.set('Channel 1', xs=latest_data.v, ys=latest_data.x)
        w.set('Channel 2', xs=latest_data.v, ys=latest_data.y)
        return

    @Element(name='Differential counts (latest)')
    def diff_latest(self):
        p = LinePlotWidget()
        p.plot('Channel 1-2')
        return p

    @diff_latest.on(sweep.acquired)
    def diff_latest_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        w.set('Channel 1-2', xs=latest_data.v, ys=latest_data["diff"])
        return

    @Element(name='Averaged counts versus frequency')
    def averaged(self):
        p = LinePlotWidget()
        p.plot('Channel 1')
        p.plot('Channel 2')
        return p

    @averaged.on(sweep.acquired)
    def averaged_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        xs = grouped.x
        ys = grouped.y
        xs_averaged = xs.mean()
        ys_averaged = ys.mean()
        xs_err = xs.std()/np.sqrt(self.data.sweep_idx.max())
        ys_err = ys.std()/np.sqrt(self.data.sweep_idx.max())
        w.set('Channel 1', xs=xs_averaged.index, ys=xs_averaged, yerrs=xs_err)
        w.set('Channel 2', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys_err)
        return

    @Element(name='Differential counts (averaged)')
    def diff_average(self):
        p = LinePlotWidget()
        p.plot('Channel 1-2')
        return p

    @diff_average.on(sweep.acquired)
    def diff_average_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        diff = grouped.diff
        diff_averaged = diff.mean()
        #diff_err = diff.std()/np.sqrt(self.data.sweep_idx.max())
        xs_err = grouped.x.std()
        ys_err = grouped.y.std()
        err = np.sqrt(xs_err**2 + ys_err**2) / np.sqrt(self.data.sweep_idx.max())
        w.set('Channel 1-2', xs=diff_averaged.index, ys=diff_averaged, yerrs=err)
        return

    @Element()
    def save(self):
        w = RepositoryWidget(self)
        return w

    # @save.on(sweep.running)
    def save_requested(self, ev):
        w = ev.widget
        state, = ev.event_args
        if state:
            return
        print(self.dataset.data)
        w.save_dataset(self.dataset)
        return

class ResettingT1Spyrelet(Spyrelet):

    requires = {
        'daq': Device,
        "awg": AG33522A,
        "pulses": Pulses,
        "siggen": SG396,
        "agil": E8257C
    }

    @Task()
    def sweep(self, **kwargs):
        self.dataset.clear()
        params = self.run_parameters.widget.get()
        for sweep in range(params['sweeps']):
            for i, f in enumerate(params['Delay'].array):
                self.pulses.stream(self.seqs[i])
                ctrs_start = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
                time.sleep(params['interpoint delay'])
                ctrs_end = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
                dctrs = ctrs_end - ctrs_start
                ctrs_rates = dctrs[:,0] / dctrs[:,1]
                multiplier = self.generate_weight_for_seq(f)
                values = {
                    'sweep_idx': sweep,
                    'v': f,
                    'x_raw': ctrs_rates[0],
                    'y_raw': ctrs_rates[1],
                    "diff_raw": ctrs_rates[0] - ctrs_rates[1],
                    'x': ctrs_rates[0]*multiplier,
                    'y': ctrs_rates[1]*multiplier,
                    "diff": (ctrs_rates[0] - ctrs_rates[1])*multiplier
                }
                self.sweep.acquire(values)
        return


    @sweep.initializer
    def initialize(self):
        params = self.run_parameters.widget.get()
        ctrs = [params['counter 1'], params['counter 2']]
        if len(set(ctrs)) != len(ctrs):
            raise RuntimeError('counter channels 1 and 2 must be different')
        self.ctr_tasks = list()
        for idx, ctr in enumerate(ctrs):
            task = CounterInputTask('counter ch {}'.format(idx))
            ch = CountEdgesChannel(ctr)
            task.add_channel(ch)
            task.start()
            self.ctr_tasks.append(task)
        self.setup_pulses()
        volts = Q_(1, "V")
        self.awg.lower_limit[1] = -3. * volts
        self.awg.high_limit[1] = 3. * volts
        self.awg.limit[1] = "ON"
        self.awg.function[1] = "dc"
        self.awg.output[1] = "ON"
        self.awg.offset[1] = params["voltage"]
        self.siggen.rf_amplitude = params["power"]
        self.siggen.frequency = params["Frequency"]
        self.siggen.rf_toggle = True

        if params["EOM"]:
            self.agil.rf_amplitude = params["EOM power"]
            self.agil.rf_frequency = params["EOM Frequency"]
            self.agil.rf_toggle = True
        else:
            self.agil.rf_toggle = False
        return


    @sweep.finalizer
    def finalize(self):
        for ctr_task in self.ctr_tasks:
            ctr_task.clear()
        self.awg.output[1] = "OFF"
        self.siggen.rf_toggle = False
        return

    def setup_pulses(self):
        params = self.run_parameters.widget.get()
        self.pulses.laser_time = int(params["probe_time"].to("ns").magnitude)
        self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
        self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
        self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
        self.pulses.polarize_time = int(params["Burn"].to("ns").magnitude)
        self.pulses.settle = int(params["Settle"].to("ns").magnitude)
        self.pulses.reset = int(params["Reset"].to("ns").magnitude)
        if params["Mode"] == 'Fixed Duty Cycle':
            self.seqs = self.pulses.Resetting_L_T1(params["Delay"], pi = params["Pi"])
            self.ratio = self.pulses.total_time / self.pulses.readout_time
        elif params["Mode"] == 'Adaptive Duty Cycle':
            self.seqs = self.pulses.Resetting_L_T1_Adaptive(params["Delay"], pi = params["Pi"])
            self.ratio = 2*self.pulses.total_time / self.pulses.readout_time

    def generate_weight_for_seq(self, tau):
        params = self.run_parameters.widget.get()
        longest_time = int(round(params["Delay"]["stop"].to("ns").magnitude))
        pi_ns = int(round(params["Pi"].to("ns").magnitude))
        longest_length = self.pulses.reset + self.pulses.polarize_time + self.pulses.settle + longest_time + pi_ns +self.pulses.laser_time + self.pulses.buf_after_init + self.pulses.readout_time
        current_length = self.pulses.reset + self.pulses.polarize_time + self.pulses.settle + int(tau.to("ns").magnitude) + pi_ns + self.pulses.laser_time + self.pulses.buf_after_init + self.pulses.readout_time
        if params["Mode"] == 'Fixed Duty Cycle':
            multiplier = 1*self.ratio
        elif params["Mode"] == 'Adaptive Duty Cycle':
            multiplier = self.ratio*current_length/longest_length
        return multiplier


    @Element(name='Run parameters')
    def run_parameters(self):
        params = [
            ('counter 1', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr0',
            }),
            ('counter 2', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr1',
            }),
            ('Mode', {
                'type': list,
                'items': ["Fixed Duty Cycle", "Adaptive Duty Cycle"],
                'default': 'Fixed Duty Cycle',
            }),
            ('EOM', {
                'type': bool,
                'default': True,
            }),
            ('EOM Frequency', {
                'type': float,
                'default': 1.064e9,
                'suffix': ' GHz',
                'units': "GHz"
            }),
            ('EOM power', {
                'type': float,
                'default': 16,
            }),
            ('Delay', {
                'type': range,
                'units': "us",
                'default': {'func': 'linspace',
                            'start': 0e-9,
                            'stop': 5e-6,
                            'num': 1001},
            }),
            ('Pi', {
                'type': float,
                'default': 140e-9,
                'suffix': ' ns',
                'units': "ns"
            }),
            ('Frequency', {
                'type': float,
                'default': 6.707e9,
                'suffix': ' GHz',
                'units': "GHz"
            }),
            ('Reset', {
                'type': float,
                'default': 150e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('Burn', {
                'type': float,
                'default': 900e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('Settle', {
                'type': float,
                'default': 150e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('voltage', {
                'type': float,
                'default': 0,
                'suffix': ' V',
                'units': "V"
            }),
            ('power', {
                'type': float,
                'default': -10,
            }),
            ('sweeps', {
                'type': int,
                'default': 10,
                'positive': True,
            }),
            ('samples', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ('interpoint delay', {
                'type': float,
                'default': 1,
                'nonnegative': True,
            }),
            ('latest window', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ("probe_time", {
                'type': float,
                'default': 50e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("readout_time", {
                'type': float,
                'default': 150e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("delay_time", {
                'type': float,
                'default': 2e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("reset_time", {
                'type': float,
                'default': 5e-6,
                'suffix': ' s',
                'units': "s"
            }),
        ]
        w = ParamWidget(params)
        return w


    @Element(name='Latest counts versus frequency')
    def latest(self):
        p = LinePlotWidget()
        p.plot('Channel 1')
        p.plot('Channel 2')
        return p

    @latest.on(sweep.acquired)
    def latest_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        w.set('Channel 1', xs=latest_data.v, ys=latest_data.x)
        w.set('Channel 2', xs=latest_data.v, ys=latest_data.y)
        return

    @Element(name='Differential counts (latest)')
    def diff_latest(self):
        p = LinePlotWidget()
        p.plot('Channel 1-2')
        return p

    @diff_latest.on(sweep.acquired)
    def diff_latest_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        diff = latest_data.diff
        w.set('Channel 1-2', xs=latest_data.v, ys=diff)
        return

    @Element(name='Averaged counts versus frequency')
    def averaged(self):
        p = LinePlotWidget()
        p.plot('Channel 1')
        p.plot('Channel 2')
        return p

    @averaged.on(sweep.acquired)
    def averaged_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        xs = grouped.x
        ys = grouped.y
        xs_averaged = xs.mean()
        ys_averaged = ys.mean()
        xs_err = xs.std()/np.sqrt(self.data.sweep_idx.max())
        ys_err = ys.std()/np.sqrt(self.data.sweep_idx.max())
        w.set('Channel 1', xs=xs_averaged.index, ys=xs_averaged, yerrs=xs_err)
        w.set('Channel 2', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys_err)
        return

    @Element(name='Differential counts (averaged)')
    def diff_average(self):
        p = LinePlotWidget()
        p.plot('Channel 1-2')
        return p

    @diff_average.on(sweep.acquired)
    def diff_average_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        diff = grouped.diff
        diff_averaged = diff.mean()
        diff_err = diff.std()/np.sqrt(self.data.sweep_idx.max())
        w.set('Channel 1-2', xs=diff_averaged.index, ys=diff_averaged, yerrs=diff_err)
        return

    @Element()
    def save(self):
        w = RepositoryWidget(self)
        return w

    # @save.on(sweep.running)
    def save_requested(self, ev):
        w = ev.widget
        state, = ev.event_args
        if state:
            return
        print(self.dataset.data)
        w.save_dataset(self.dataset)
        return

class ResettingCPMGSpyrelet(Spyrelet):

    requires = {
        'daq': Device,
        "awg": AG33522A,
        "pulses": Pulses,
        "siggen": SG396,
        "agil": E8257C
    }

    @Task()
    def sweep(self, **kwargs):
        self.dataset.clear()
        params = self.run_parameters.widget.get()
        for sweep in range(params['sweeps']):
            for i, f in enumerate(params['Delay'].array):
                self.pulses.stream(self.seqs[i])
                ctrs_start = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
                time.sleep(params['interpoint delay'])
                ctrs_end = np.array([(task.read(samples_per_channel=1)[-1], time.time()) for task in self.ctr_tasks])
                dctrs = ctrs_end - ctrs_start
                ctrs_rates = dctrs[:,0] / dctrs[:,1]
                values = {
                    'sweep_idx': sweep,
                    'v': f,
                    'x': ctrs_rates[0]*self.ratio,
                    'y': ctrs_rates[1]*self.ratio,
                    "diff": (ctrs_rates[0] - ctrs_rates[1])*self.ratio
                }
                self.sweep.acquire(values)
        return


    @sweep.initializer
    def initialize(self):
        params = self.run_parameters.widget.get()
        ctrs = [params['counter 1'], params['counter 2']]
        if len(set(ctrs)) != len(ctrs):
            raise RuntimeError('counter channels 1 and 2 must be different')
        self.ctr_tasks = list()
        for idx, ctr in enumerate(ctrs):
            task = CounterInputTask('counter ch {}'.format(idx))
            ch = CountEdgesChannel(ctr)
            task.add_channel(ch)
            task.start()
            self.ctr_tasks.append(task)
        self.setup_pulses()
        volts = Q_(1, "V")
        self.awg.lower_limit[1] = -3. * volts
        self.awg.high_limit[1] = 3. * volts
        self.awg.limit[1] = "ON"
        self.awg.function[1] = "dc"
        self.awg.output[1] = "ON"
        self.awg.offset[1] = params["voltage"]
        self.siggen.rf_amplitude = params["power"]
        self.siggen.frequency = params["Frequency"]
        self.siggen.rf_toggle = True
        if params["EOM"]:
            self.agil.rf_amplitude = params["EOM power"]
            self.agil.rf_frequency = params["EOM Frequency"]
            self.agil.rf_toggle = True
        else:
            self.agil.rf_toggle = False
        return


    @sweep.finalizer
    def finalize(self):
        for ctr_task in self.ctr_tasks:
            ctr_task.clear()
        self.awg.output[1] = "OFF"
        self.siggen.rf_toggle = False
        return

    def setup_pulses(self):
        params = self.run_parameters.widget.get()
        self.pulses.laser_time = int(params["probe_time"].to("ns").magnitude)
        self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
        self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
        self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
        self.pulses.polarize_time = int(params["Burn"].to("ns").magnitude)
        self.pulses.settle = int(params["Settle"].to("ns").magnitude)
        self.pulses.reset = int(params["Reset"].to("ns").magnitude)
        self.seqs = self.pulses.Resetting_L_CPMG(params["Delay"], pi = params["Pi"], N=params["N"])
        self.ratio = self.pulses.total_time / self.pulses.readout_time


    @Element(name='Run parameters')
    def run_parameters(self):
        params = [
            ('counter 1', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr0',
            }),
            ('counter 2', {
                'type': list,
                'items': list(self.daq.counter_input_channels),
                'default': 'Dev1/ctr1',
            }),
            ('EOM', {
                'type': bool,
                'default': True,
            }),
            ('EOM Frequency', {
                'type': float,
                'default': 1.064e9,
                'suffix': ' GHz',
                'units': "GHz"
            }),
            ('EOM power', {
                'type': float,
                'default': 16,
            }),
            ('Delay', {
                'type': range,
                'units': "us",
                'default': {'func': 'linspace',
                            'start': 0e-9,
                            'stop': 5e-6,
                            'num': 1001},
            }),
            ('Pi', {
                'type': float,
                'default': 140e-9,
                'suffix': ' ns',
                'units': "ns"
            }),
            ('N', {
                'type': int,
                'default': 2,
            }),
            ('Frequency', {
                'type': float,
                'default': 1.068e9,
                'suffix': ' GHz',
                'units': "GHz"
            }),
            ('Reset', {
                'type': float,
                'default': 150e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('Burn', {
                'type': float,
                'default': 900e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('Settle', {
                'type': float,
                'default': 150e-6,
                'suffix': ' us',
                'units': "us"
            }),
            ('voltage', {
                'type': float,
                'default': 0,
                'suffix': ' V',
                'units': "V"
            }),
            ('power', {
                'type': float,
                'default': -10,
            }),
            ('sweeps', {
                'type': int,
                'default': 10000,
                'positive': True,
            }),
            ('samples', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ('interpoint delay', {
                'type': float,
                'default': 1,
                'nonnegative': True,
            }),
            ('latest window', {
                'type': int,
                'default': 100,
                'positive': True,
            }),
            ("probe_time", {
                'type': float,
                'default': 50e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("readout_time", {
                'type': float,
                'default': 150e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("delay_time", {
                'type': float,
                'default': 2e-6,
                'suffix': ' s',
                'units': "s"
            }),
            ("reset_time", {
                'type': float,
                'default': 5e-6,
                'suffix': ' s',
                'units': "s"
            }),
        ]
        w = ParamWidget(params)
        return w


    @Element(name='Latest counts versus frequency')
    def latest(self):
        p = LinePlotWidget()
        p.plot('Channel 1')
        p.plot('Channel 2')
        return p

    @latest.on(sweep.acquired)
    def latest_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        w.set('Channel 1', xs=latest_data.v, ys=latest_data.x)
        w.set('Channel 2', xs=latest_data.v, ys=latest_data.y)
        return

    @Element(name='Differential counts (latest)')
    def diff_latest(self):
        p = LinePlotWidget()
        p.plot('Channel 1-2')
        return p

    @diff_latest.on(sweep.acquired)
    def diff_latest_update(self, ev):
        w = ev.widget
        latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
        w.set('Channel 1-2', xs=latest_data.v, ys=latest_data["diff"])
        return

    @Element(name='Averaged counts versus frequency')
    def averaged(self):
        p = LinePlotWidget()
        p.plot('Channel 1')
        p.plot('Channel 2')
        return p

    @averaged.on(sweep.acquired)
    def averaged_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        xs = grouped.x
        ys = grouped.y
        xs_averaged = xs.mean()
        ys_averaged = ys.mean()
        xs_err = xs.std()/np.sqrt(self.data.sweep_idx.max())
        ys_err = ys.std()/np.sqrt(self.data.sweep_idx.max())
        w.set('Channel 1', xs=xs_averaged.index, ys=xs_averaged, yerrs=xs_err)
        w.set('Channel 2', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys_err)
        return

    @Element(name='Differential counts (averaged)')
    def diff_average(self):
        p = LinePlotWidget()
        p.plot('Channel 1-2')
        return p

    @diff_average.on(sweep.acquired)
    def diff_average_update(self, ev):
        w = ev.widget
        grouped = self.data.groupby('v')
        diff = grouped.diff
        diff_averaged = diff.mean()
        #diff_err = diff.std()/np.sqrt(self.data.sweep_idx.max())
        xs_err = grouped.x.std()
        ys_err = grouped.y.std()
        err = np.sqrt(xs_err**2 + ys_err**2) / np.sqrt(self.data.sweep_idx.max())
        w.set('Channel 1-2', xs=diff_averaged.index, ys=diff_averaged, yerrs=err)
        return

    @Element()
    def save(self):
        w = RepositoryWidget(self)
        return w

    # @save.on(sweep.running)
    def save_requested(self, ev):
        w = ev.widget
        state, = ev.event_args
        if state:
            return
        print(self.dataset.data)
        w.save_dataset(self.dataset)
        return

# class T1Spyrelet(Spyrelet):

#     requires = {
#         'daq': Device,
#         "awg": AG33522A,
#         "pulses": Pulses,
#         "siggen": SG396
#     }

#     def count(self):
#         run_params = self.run_parameters.widget.get()
#         samples = run_params['samples']
#         delay = run_params['interpoint delay']
#         t0 = time.time()
#         prev = np.mean(self.task.read(samples_per_channel=samples))
#         prev_t = t0
#         time.sleep(delay)
#         now = time.time()
#         current = np.mean(self.task.read(samples_per_channel=samples))
#         count_rate = (current - prev) / (now - prev_t)
#         t = now - t0
#         return t, count_rate

#     @Task()
#     def sweep(self, **kwargs):
#         self.dataset.clear()
#         params = self.run_parameters.widget.get()
#         for sweep in range(params['sweeps']):
#             for i, f in enumerate(params['Delay'].array):
#                 self.pulses.stream(self.seqs[i])
#                 t, y = self.count()
#                 values = {
#                     'sweep_idx': sweep,
#                     'v': f,
#                     'y': y,
#                 }
#                 self.sweep.acquire(values)
#         return


#     @sweep.initializer
#     def initialize(self):
#         params = self.input_parameters.widget.get()
#         params2 = self.run_parameters.widget.get()
#         inp_ch = params['channel 1']
#         if inp_ch in self.daq.counter_input_channels:
#             self.task = CounterInputTask('TaskVsTime_CI')
#             channel = CountEdgesChannel(inp_ch)
#             self.task.add_channel(channel)
#         else:
#             # should never get here
#             pass
#         self.setup_pulses()
#         volts = Q_(1, "V")
#         self.awg.lower_limit[1] = -3. * volts
#         self.awg.high_limit[1] = 3. * volts
#         self.awg.limit[1] = "ON"
#         self.awg.function[1] = "dc"
#         self.awg.output[1] = "ON"
#         self.awg.offset[1] = params2["voltage"]
#         self.siggen.rf_amplitude = params2["power"]
#         self.siggen.frequency = params2["Frequency"]
#         self.siggen.rf_toggle = True
#         self.task.start()
#         return


#     @sweep.finalizer
#     def finalize(self):
#         self.task.clear()
#         self.awg.output[1] = "OFF"
#         self.siggen.rf_toggle = False
#         return

#     def setup_pulses(self):
#         params = self.run_parameters.widget.get()
#         self.pulses.laser_time = int(params["excitation_time"].to("ns").magnitude)
#         self.pulses.readout_time = int(params["readout_time"].to("ns").magnitude)
#         self.pulses.buf_after_init = int(params["delay_time"].to("ns").magnitude)
#         self.pulses.buf_after_readout = int(params["reset_time"].to("ns").magnitude)
#         self.seqs = self.pulses.T1(params["Delay"], params["Pi"])

#     @Element(name='Input parameters')
#     def input_parameters(self):
#         channels = 4
#         comboboxes = [pg.ComboBox() for _ in range(channels)]
#         ci_channels = list(self.daq.counter_input_channels)
#         none_channel = ['None']
#         all_channels = none_channel + ci_channels
#         params = list()
#         for idx, combobox in enumerate(comboboxes):
#             channel_param = ('channel {}'.format(idx + 1), {
#                 'type': list,
#                 'items': all_channels,
#             })
#             params.append(channel_param)
#         w = ParamWidget(params)
#         return w

#     @Element(name='Run parameters')
#     def run_parameters(self):
#         params = [
#             ('Delay', {
#                 'type': range,
#                 'units': "ns",
#                 'default': {'func': 'linspace',
#                             'start': 5e-9,
#                             'stop': 1005e-9,
#                             'num': 101},
#             }),
#             ('Pi', {
#                 'type': float,
#                 'default': 50e-9,
#                 'suffix': ' ns',
#                 'units': "ns"
#             }),
#             ('Frequency', {
#                 'type': float,
#                 'default': 6.707e9,
#                 'suffix': ' GHz',
#                 'units': "GHz"
#             }),
#             ('voltage', {
#                 'type': float,
#                 'default': 0,
#                 'suffix': ' V',
#                 'units': "V"
#             }),
#             ('power', {
#                 'type': float,
#                 'default': -10,
#             }),
#             ('sweeps', {
#                 'type': int,
#                 'default': 10,
#                 'positive': True,
#             }),
#             ('samples', {
#                 'type': int,
#                 'default': 100,
#                 'positive': True,
#             }),
#             ('interpoint delay', {
#                 'type': float,
#                 'default': 1,
#                 'nonnegative': True,
#             }),
#             ('latest window', {
#                 'type': int,
#                 'default': 100,
#                 'positive': True,
#             }),
#             ("excitation_time", {
#                 'type': float,
#                 'default': 600e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("readout_time", {
#                 'type': float,
#                 'default': 150e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("delay_time", {
#                 'type': float,
#                 'default': 2e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#             ("reset_time", {
#                 'type': float,
#                 'default': 5e-6,
#                 'suffix': ' s',
#                 'units': "s"
#             }),
#         ]
#         w = ParamWidget(params)
#         return w

#     @Element()
#     def latest_history(self):
#         p = LinePlotWidget()
#         p.plot('channel 1', symbol=None)
#         return p

#     @latest_history.on(sweep.acquired)
#     def latest_history_update(self, ev):
#         w = ev.widget
#         latest_data = self.data[self.data.sweep_idx == self.data.sweep_idx.max()]
#         w.set('channel 1', xs=latest_data.v, ys=latest_data.y)
#         return

#     @Element()
#     def averaged(self):
#         p = LinePlotWidget()
#         p.plot('channel 1', symbol=None)
#         return p

#     @averaged.on(sweep.acquired)
#     def averaged_update(self, ev):
#         w = ev.widget
#         grouped = self.data.groupby('v')
#         ys = grouped.y
#         ys_averaged = ys.mean()
#         w.set('channel 1', xs=ys_averaged.index, ys=ys_averaged, yerrs=ys.std())
#         return

#     @Element()
#     def save(self):
#         w = RepositoryWidget(self)
#         return w

#     # @save.on(sweep.running)
#     def save_requested(self, ev):
#         w = ev.widget
#         state, = ev.event_args
#         if state:
#             return
#         print(self.dataset.data)
#         w.save_dataset(self.dataset)
#         return
