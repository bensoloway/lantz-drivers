import numpy as np
from enum import Enum
import six

class OutputState():
    """class that defines an output state of Pulse Streamer 8/2"""
    digital_channel=8
    analog_channel=2
    analog_range=1
    analog_conv=0x7fff
    conv_type=int

    def __init__(self, digi, A0=0.0, A1=0.0):
        #set analog values
        assert (abs(A0)<=OutputState.analog_range and abs(A1)<=OutputState.analog_range), "Pulse Streamer 8/2 supports "\
                "analog voltage range of +/-"+str(OutputState.analog_range) +"V" #check hardware
        self.__A0=OutputState.conv_type(round(OutputState.analog_conv*A0))
        self.__A1=OutputState.conv_type(round(OutputState.analog_conv*A1))

        ##set obligatory argument for digital channels
        if isinstance(digi, list):        
            self.__digi=self.chans_to_mask(digi)
        else:
            raise TypeError("digi has to be a list with elements of type int")        

    def getData(self):
        """return data in specific format"""
        #collect return values in list, start with obligatory digital value
        return(self.__digi, self.__A0, self.__A1)
        
    def chans_to_mask(self, chans):
        """convert channel list into bitmask"""
        mask = 0
        for chan in chans:
            assert chan in range(OutputState.digital_channel),"Pulse Streamer 8/2 supports "\
            "up to " +str(OutputState.digital_channel)+ " digital channels"
            mask |= 1<<chan
        return mask

    @staticmethod
    def ZERO():
        """Static method to create an OutputState object for the Pulse Streamer 8/2 which sets all outputs to 0"""
        return OutputState([],)


class Sequence():

    digital_channel=8
    analog_channel=2
    analog_range=1
    analog_conv=0x7fff
    conv_type=int
    #max_seq_step=2e6 #currently not used

    def __init__(self, pulser=None):
        self.__union_seq=[(0,0,0,0)]
        self.__sequence_up_to_date = True
        self.__duration =0
        self.__all_t_cumsums_concatinated=np.array([], dtype=np.int64)
        self.__pad_seq={}
        self.__channel_digital= dict.fromkeys(range(Sequence.digital_channel), ([(0, 0)], np.array([0], dtype=np.int64)))
        self.__channel_analog=dict.fromkeys(range(Sequence.analog_channel), ([(0, 0.0)], np.array([0], dtype=np.int64)))
    
    def __union(self):
        """Merges all channels to final list of sequence-steps"""
        # idea of the algorithm
        # 1) for each channel, get the absolute timestamps (not the relative) -> t_cumsums stored in self.__pad_seq[x][1]
        # 2) join all these absolute timestamps together to a sorted unique list. This results in the timestamps (unique_t_cumsum) of the final pulse list 
        # 3) expand every single channel to final pulse list (unique_t_cumsum)
        # 4) join the channels together
        # 5) Simplify Sequence by concatenating states with same digital and analog values

        #get channel numbers - currently just use the class atrributes
        count_digital_channels = Sequence.digital_channel
        count_analog_channels = Sequence.analog_channel
        count_all_channels = count_digital_channels + count_analog_channels
        
        # 1) Done in setDigital/setAnalog functions
        # 2) join all these absolute timestamps together to a sorted unique list which. This results in the timestamps (unique_t_cumsum) of the final pulse list
        self.__pad()
        unique_t_cumsum = np.unique(self.__all_t_cumsums_concatinated)

        # 3) expand every single channel to final pulse list (unique_t_cumsum)
        # create 2d array for every channel and timestamp
        data = np.zeros([count_all_channels, len(unique_t_cumsum)], dtype=np.float64)
        
        for i in range(count_all_channels):
            #get channel data and cumsum of current channel
            sequence_array = self.__pad_seq[i][0]
            cumsum = self.__pad_seq[i][1]
            np_array = np.array([k[1] for k in sequence_array])
            data[i] = np_array[np.searchsorted(cumsum, unique_t_cumsum, side='left')]

        # 4) join the channels together
        #digital channels
        if count_digital_channels != 0:
            digi = np.int_(data[0,:])
        else:
            digi = np.zeros(len(unique_t_cumsum))

        for i in range(1, count_digital_channels):
            digi = digi + np.int_((2**i)*data[i,:])
        
        ana=[]
        for i in range(count_analog_channels):
            ana.append(np.round((data[count_digital_channels + i]*Sequence.analog_conv)).astype(Sequence.conv_type))


        #5) Simplify Sequence by concatenating states with same digital and analog values
        if len(digi)>1:
            #get indexes of equal digital values in a row
            redundant = digi[:-1]==digi[1:]
            for a in ana:
                redundant &= a[:-1]==a[1:]
            index = np.nonzero(redundant)

            #delete the elements (absolute timestamps, digital and analog values)
            unique_t_cumsum = np.delete(unique_t_cumsum,index)
            digi = np.delete(digi,index)
            for i in range(len(ana)):
                ana[i]=np.delete(ana[i],index)

        # revert the cumsum to get the relative pulse durations
        ts = np.insert(np.diff(unique_t_cumsum), 0, unique_t_cumsum[0])
        
        #create the final pulse list
        result = list(zip(ts, digi, *ana))
        
        # there might be a pulse duration of 0 in the very beginning - remove it
        if len(result) > 1:
            if result[0][0] == 0:
                return result[1::]
        
        return result

    def __pad(self):
        """Pad all channels to the maximal channel duration"""
        #get the max duration of all channels
        duration = np.int64(self.getDuration())
        #initialize padded channel data
        self.__pad_seq = dict.fromkeys(range (Sequence.digital_channel), ([(0, 0)], np.array([0], dtype=np.int64)))
        pad_seq_ana = dict.fromkeys(range(Sequence.digital_channel, Sequence.digital_channel+Sequence.analog_channel),([(0, 0.0 )], np.array([0], dtype=np.int64)))
        self.__pad_seq.update(pad_seq_ana)
        for key, pattern_data in self.__channel_digital.items():
            self.__pad_seq[key] = pattern_data
        for key, pattern_data in self.__channel_analog.items():
            self.__pad_seq[key+Sequence.digital_channel] = pattern_data
        #pad each channel with last value
        for key, pattern_data in self.__pad_seq.items():
            pad_value=duration-pattern_data[1][-1]
            pad_level=pattern_data[0][-1][1] 
            if (pad_value!=0):
                if (duration==pad_value):
                    #channel has only been initialized
                    self.__pad_seq[key]=([(pad_value.astype(int),  pad_level)], np.array([duration], dtype=np.int64))
                else:
                    #pad with last value and update cumulated timestamps
                    new_seq = pattern_data[0] + [(pad_value.astype(int),  pad_level)]
                    self.__pad_seq[key]=(new_seq, np.append(pattern_data[1], duration))
    
        self.__all_t_cumsums_concatinated=np.array([], dtype=np.int64)
        for key in self.__pad_seq:
            self.__all_t_cumsums_concatinated = np.append(self.__all_t_cumsums_concatinated, self.__pad_seq[key][1])


    def get_pad(self):
        """return padded Sequence and the number of the last used/existing digital channel"""
        self.__pad()
        return self.__pad_seq

    def setDigital(self, channel, channel_sequence):
        """set one or list of digital channel"""
        if not isinstance(channel, list):
            channel = [channel]
        #channel number check
        for i in channel:
            assert i in range(Sequence.digital_channel), "Pulse Streamer 8/2 supports "\
            "up to " +str(Sequence.digital_channel)+ " digital "\
            "and " +str(Sequence.analog_channel)+ " analog channels"
        #set control flag that sequence has to be unified again before returned
        self.__sequence_up_to_date = False
        
        if len(channel_sequence) > 0:            
            # store last entry
            last_touple = channel_sequence[-1]
            #remove all entries which have dt = 0
            channel_sequence = list(filter(lambda x: x[0]!=0, channel_sequence))
            #if the last dt is 0, add it again to the list
            if (last_touple[0]) == 0:
                channel_sequence.append(last_touple)
            ### argument check value == 0/1
            timeline=[]
            for t,p in channel_sequence:
                assert p in [0,1], "Digital values must either be 0 or 1"
                assert t >=0
                #timeline=list(zip(*channel_sequence))[0]
                timeline.append(t) #(1<<channel)*
            # make a numeric array of the cumulated timestamps
            cumsum=np.cumsum(np.array(timeline, dtype=np.int64))
        else:
            cumsum = np.array([0], dtype=np.int64)

        #Store data and cumulated timestamps
        for i in channel:
            self.__channel_digital[i]= (channel_sequence, cumsum)   

    def getDigital(self):
        """return digital channel data"""
        return self.__channel_digital

    def invertDigital(self, channel):
        """set one or list of digital channel"""
        if not isinstance(channel, list):
            channel = [channel]
        #channel number check
        for i in channel:
            assert i in range(Sequence.digital_channel), "Pulse Streamer 8/2 supports "\
            "up to " +str(Sequence.digital_channel)+ " digital "\
            "and " +str(Sequence.analog_channel)+ " analog channels"
        #set control flag that sequence has to be unified again before returned
        self.__sequence_up_to_date = False
        
        #invert channels
        for i in channel:
            if (len(self.__channel_digital[i][0]) > 1):
                self.setDigital(i, list(map(lambda x: (x[0], x[1]^1), self.__channel_digital[i][0])))
            elif (self.__channel_digital[i][0][0][0] != 0):
                self.setDigital(i, [(self.__channel_digital[i][0][0][0], 1-self.__channel_digital[i][0][0][1])])

    def setAnalog(self, channel, channel_sequence):
        """set one or list of analog channel"""
        if not isinstance(channel, list):
            channel = [channel]
        for i in channel:
            assert i in range(Sequence.analog_channel), "Pulse Streamer 8/2 supports "\
            "up to " +str(Sequence.digital_channel)+ " digital "\
            "and " +str(Sequence.analog_channel)+ " analog channels"

        self.__sequence_up_to_date = False

        if len(channel_sequence) > 0:            
            # store last entry
            last_touple = channel_sequence[-1]
            #remove all entries which have dt = 0
            channel_sequence = list(filter(lambda x: x[0]!=0, channel_sequence))
            #if the last dt is 0, add it again to the list
            if (last_touple[0]) == 0:
                channel_sequence.append(last_touple)
            ### argument check
            timeline=[]
            for t,p in channel_sequence:
                assert abs(p) <= Sequence.analog_range, "Pulse Streamer 8/2 supports "\
                "analog voltage range of +/-"+str(Sequence.analog_range) +"V"
                assert t >=0
                #timeline=list(zip(*channel_sequence))[0]
                timeline.append(t) #(1<<channel)*
            # make a numeric array of the cumulated timestamps
            cumsum=np.cumsum(np.array(timeline, dtype=np.int64))
        else:
            cumsum = np.array([0], dtype=np.int64)

        for i in channel:
            self.__channel_analog[i] = (channel_sequence, cumsum)
	
    def getAnalog(self):
        """return analog channel data"""
        return self.__channel_analog

    def invertAnalog(self, channel):
        """set one or list of analog channel"""
        if not isinstance(channel, list):
            channel = [channel]
        for i in channel:
            assert i in range(Sequence.analog_channel), "Pulse Streamer 8/2 supports "\
            "up to " +str(Sequence.digital_channel)+ " digital "\
            "and " +str(Sequence.analog_channel)+ " analog channels"

        self.__sequence_up_to_date = False

        #invert channels
        for i in channel:
            if (len(self.__channel_analog[i][0]) > 1):
                self.setAnalog(i, list(map(lambda x: (x[0], x[1]*(-1)), self.__channel_analog[i][0])))
            elif (self.__channel_analog[i][0][0][0] != 0):
                self.setAnalog(i, [(self.__channel_analog[i][0][0][0], self.__channel_analog[i][0][0][1]*(-1))])

    def getData(self):
        """return list of sequence steps"""
        #check if sequence has to be rebuild
        if not self.__sequence_up_to_date:
            #merge all channels over final unified timestamps
            self.__union_seq = self.__union()
            #set flag that final data has been build
            self.__sequence_up_to_date = True
         
        return self.__union_seq

    def isEmpty(self):
        """returns True if no channel has been set"""
        for key, pattern_data in self.__channel_digital.items():
            if pattern_data[1][-1] != 0:
                return False
            else:
                pass

        for key, pattern_data in self.__channel_digital.items():
            if pattern_data[1][-1] != 0:
                return False
            else:
                pass

        return True

    def getDuration(self):
        """returns time in ns of the padded sequence"""
        duration =0
        for key, pattern_data in self.__channel_digital.items():
            channel_duration = pattern_data[1][-1]
            duration = max(duration, channel_duration)

        for key, pattern_data in self.__channel_analog.items():
            channel_duration = pattern_data[1][-1]
            duration = max(duration, channel_duration)
        
        return duration

    def getLastState(self):
        """returns the OutputState of the last sequence step"""
        ana={}
        digi=[]
        self.__pad()
        for key, pattern_data in self.__pad_seq.items():
            if key < Sequence.digital_channel:
                if (pattern_data[0][-1][1]):
                    #save digital values if set
                    digi.append(key)
            else:
                #save analog values
                ana.update({'A'+str(key-Sequence.digital_channel):pattern_data[0][-1][1]})
                
        #create and return OutputState object
        return OutputState(digi, ana['A0'], ana['A1'])

    @staticmethod
    def concatenate(seq1, seq2):
        """return sequence object as a Concatenation of two sequences """
        #get padded data of left side
        pad= seq1.get_pad()
        duration_1= seq1.getDuration()
        #get data of right side
        digital_2 = seq2.getDigital()
        analog_2 = seq2.getAnalog()
        #create new Sequence
        ret=seq1.__class__()
        #tool lists for internal use
        used_list_ana=[]
        used_list_digi=[]
        for key in pad:
            if key > (Sequence.digital_channel-1):
                #set analog channels of new Sequence
                if analog_2[key-Sequence.digital_channel][0] != [(0,0.0)]:
                    #channel was set in seq1 and seq2 => concatenate and remember (used_list)
                    ret.setAnalog(key-Sequence.digital_channel, pad[key][0]+analog_2[key-Sequence.digital_channel][0])
                    used_list_ana.append(key-Sequence.digital_channel)
                else:
                    #channel was only used in seq1
                    ret.setAnalog((key-Sequence.digital_channel), pad[key][0])
                    used_list_ana.append(key-Sequence.digital_channel)
            else:
                #set digital channels of new sequence
                if digital_2[key][0] != [(0,0)]:
                    #channel was set in seq1 and seq2 => concatenate and remember (used_list)
                    ret.setDigital(key, pad[key][0]+digital_2[key][0])
                    used_list_digi.append(key)
                else:
                    #channel was only used in seq1
                    ret.setDigital(key, pad[key][0])
                    used_list_digi.append(key)

        return ret

    def __add__(self, another):
        """Overriding __add__  to make concatenate() also working with seq3=seq1+seq2"""
        return Sequence.concatenate(self, another)


    @staticmethod
    def repeat(seq, n_times):
        """return sequence object as n_times repetition of seq"""
        #get padded sequence data
        pad= seq.get_pad()
        #create new sequence
        ret=seq.__class__()
        #set channels with repeated channel data
        for key in pad:
            if key > (Sequence.digital_channel-1):
                ret.setAnalog(key-Sequence.digital_channel, n_times*pad[key][0])
            else:             
                ret.setDigital(key,n_times*pad[key][0])
        
        return ret
    
    def __mul__(self, other):
        """Overrideing __mul__ to make sure repeat() also works with seq2=seq1*x"""
        return Sequence.repeat(self, other)

    def __rmul__(self,other):
        """Overrideing __rmul__ to make sure repeat() also works with seq2=x*seq1"""
        return Sequence.repeat(self, other)

    def plot(self):
        """plot a digital sequence using matplotlib"""
        import matplotlib.pyplot as plt
        self.__pad()

        fig = plt.figure()
        for key, pattern_data in self.__pad_seq.items():
            #create timeline to plot - add 0 to cumsum
            t=np.concatenate((np.array([0], dtype=np.int64), pattern_data[1]))
            #create channel data for plotting
            plot_ch_data=[]
            for k in pattern_data[0]:
                plot_ch_data.append(k[1])
            #append last element
            plot_ch_data= np.append(plot_ch_data, plot_ch_data[-1])

            ax=plt.subplot(10,1, (10-key))
            
            if key >0:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.xlabel("time/ns")
            if key>(Sequence.digital_channel-1):
                plt.ylabel("A"+str(key-Sequence.digital_channel), labelpad=20, rotation='horizontal')
                plt.setp(ax.get_yticklabels(), fontsize=6)
                plt.ylim(-1.5, 1.5)
                plt.box(on=None)
            else:
                plt.ylabel("D"+str(key), labelpad=20, rotation='horizontal')
                plt.setp(ax.get_yticklabels(), fontsize=6)
                plt.ylim(0,1.5)
                plt.box(on=None)
            plt.step(t, plot_ch_data, ".-",  where='post', color='k')

        plt.tight_layout(0.5)
        fig.canvas.set_window_title('Sequence')
        plt.show()