print(f'the name:{__name__}')
from .agilis import Agilis

from lantzdrivers.ni.ni_motion_controller import NIDAQMotionController, NIDAQAxis
from lantz.core import Driver, Feat, Action
from lantz import Q_
#from nspyre.widgets import feat
#from nspyre.widgets.feat import get_feat_widget
#from nspyre.widgets import instrument_manager
#from nspyre.widgets.instrument_manager import Instrument_Manager_Widget
#from nspyre.widgets.instrument_manager import FeatTreeWidgetItem
#from PyQt5 import QtWidgets, QtCore

class FSM(Driver):
    def __init__(self, x_ch, y_ch, ctr_ch):#, z_ch, ctr_ch):
        x_axis = NIDAQAxis(x_ch, 'um', Q_(1/8.8, 'V/um'), limits=(Q_(-88,'um'), Q_(88,'um')))
        y_axis = NIDAQAxis(y_ch, 'um', Q_(1/7.2, 'V/um'), limits=(Q_(-72,'um'), Q_(72,'um')))
        # z_axis = NIDAQAxis(z_ch, 'um', Q_(1/25, 'V/um'), limits=(Q_(0.0,'um'), Q_(250,'um')))
        self.daq_controller = NIDAQMotionController(ctr_ch, Q_(20, 'kHz'), {'x': x_axis, 'y': y_axis}, ao_smooth_steps=Q_(5000, '1/V'))
        self.increment_X = "Q_(10,'nm')"
        self.multiplier_X = 0
        self.increment_Y = "Q_(10,'nm')"
        self.multiplier_Y = 0 
        self.increment_Z = "Q_(10,'nm')"
        self.multiplier_Z = 0         

    def initialize(self):
        return self.daq_controller.initialize()
        
    def finalize(self):
        return self.daq_controller.finalize()   

    @Feat(units='Hz')
    def acq_rate(self):
        return self.daq_controller.acq_rate

    @Action()
    def new_ctr_task(self, ctr_ch):
        return self.daq_controller.new_ctr_task(ctr_ch)

    @Action()
    def close_current_ctr_task(self):
        task = self.daq_controller.current_counter_task
        task.close()

    @Action()
    def move(self, point):
        return self.daq_controller.move(point)

    @Action()
    def line_scan(self, init_point, final_point, steps, pts_per_step):
        # print('line_scan')
        return self.daq_controller.line_scan(init_point, final_point, steps, pts_per_step)

    @acq_rate.setter
    def acq_rate(self, acq_rate):
        self.daq_controller.acq_rate = 1

    @Feat(units='um')
    def x(self):
        print(self.daq_controller.position['x'])
        return self.daq_controller.position['x']

    @x.setter
    def x(self, pos):
        print('x set {}'.format(pos))
        self.daq_controller.move({'x': pos, 'y': self.y})

    @Feat(units='um')
    def y(self):
        return self.daq_controller.position['y']

    @y.setter
    def y(self, pos):
        self.move({'x': self.x, 'y': pos})

    # @Feat(units='um')
    # def z(self):
    #     return self.daq_controller.position['z']

    # @z.setter
    # def z(self, pos):
    #     self.daq_controller.move({'x': self.x, 'y': self.y, 'z': pos})
