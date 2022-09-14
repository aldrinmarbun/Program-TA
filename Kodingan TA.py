from tkinter import *
from tkinter.ttk import Combobox
from sklearn.metrics import mean_squared_error
from math import log10, sqrt

 
window=Tk()
var = StringVar()
var.set("one")
#data=("one", "two", "three", "four")
#cb=Combobox(window, values=data)
#cb.place(x=60, y=150)

#lb=Listbox(window, height=5, selectmode='multiple')
#for num in data:
#    lb.insert(END,num)
#lb.place(x=250, y=150)


class Table:
      
    def __init__(self,root,rows):
        # code for creating table
        newWindow = Toplevel(window)
        newWindow.title("Data Test")
        # sets the geometry of toplevel
        newWindow.geometry("600x400")

        for i in range(len(rows)):
            for j in range(10):
                self.e=Entry(newWindow)
                  
                self.e.grid(row=i, column=j)
                self.e.insert(END, rows[i][j])

         



#handle tombol klik
def tombol_klik():
    tombol["state"] = DISABLED
    if (v0.get()<2):
        metoda=1
    else:
        if (v0.get()==2):
            metoda=2
        else:
            metoda=3
    print(metoda)
    V=[]
    if (v1.get()==1):
        if not V:
            V=[1]
        else:
            V.append(1)
    if (v2.get()==1):
        if not V:
            V=[2]
        else:
            V.append(2)
    if (v3.get()==1):
        if not V:
            V=[3]
        else:
            V.append(3)
    if (v4.get()==1):
        if not V:
            V=[4]
        else:
            V.append(4)
    if (v5.get()==1):
        if not V:
            V=[5]
        else:
            V.append(5)
    if (v6.get()==1):
        if not V:
            V=[6]
        else:
            V.append(6)
    if (v7.get()==1):
        if not V:
            V=[7]
        else:
            V.append(7)
    if (v8.get()==1):
        if not V:
            V=[8]
        else:
            V.append(8)
    if (v9.get()==1):
        if not V:
            V=[9]
        else:
            V.append(9)
    if (v10.get()==1):
        if not V:
            V=[10]
        else:
            V.append(10)
    if not V:
        tombol["state"] = NORMAL
        print('Fitur tidak boleh kosong, Selesai ...')
        return
    print(V)
    from scipy.signal import kaiserord, lfilter, firwin, freqz
    import scipy as sp
    import numpy as np
    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import signal
    from numpy import savetxt
    from scipy.stats import entropy
    from math import log, e
    import os, sys
    from scipy.signal import butter, iirnotch, lfilter
    from scipy.stats import norm, kurtosis
    from scipy.stats import skew

    ## A high pass filter allows frequencies higher than a cut-off value
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
        return b, a
    ## A low pass filter allows frequencies lower than a cut-off value
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
        return b, a
    def notch_filter(cutoff, q):
        nyq = 0.5*fs
        freq = cutoff/nyq
        b, a = iirnotch(freq, q)
        return b, a

    # def highpass(data, fs, order=5):
    #     b,a = butter_highpass(cutoff_high, fs, order=order)
    #     x = lfilter(b,a,data)
    #     return x

    # def lowpass(data, fs, order =5):
    #     b,a = butter_lowpass(cutoff_low, fs, order=order)
    #     y = lfilter(b,a,data)
    #     return y

    # def notch(data, powerline, q):
    #     b,a = notch_filter(powerline,q)
    #     z = lfilter(b,a,data)
    #     return z

    def final_filter(data, fs, order=5):
        b, a = butter_highpass(cutoff_high, fs, order=order)
        x = lfilter(b, a, data)
        d, c = butter_lowpass(cutoff_low, fs, order = order)
        y = lfilter(d, c, x)
        f, e = notch_filter(powerline, 30)
        z = lfilter(f, e, y)     
        return z
    # def BandpassFilter(signal2, fs1):
    #     '''Bandpass filter the signal between 40 and 240 BPM'''
    
    #     # Convert to Hz
    #     lo, hi = 40/60, 240/60
    
    #     b, a = sp.signal.butter(3, (lo, hi), btype='bandpass', fs=fs1)
    #     return sp.signal.filtfilt(b, a, signal2)
    # def Length(data):
    #     """Returns the number of samples in a time series"""
    #     return len(data)

    def Mean(data):
        """Returns the mean of a time series"""
        return data.mean()

    def Std(data):
        """Returns the standard deviation a time series"""
        return data.std()

    # def Min(data):
    #     """Returns the mean of a time series"""
    #     return data.min()

    def Max(data):
        """Returns the standard deviation a time series"""
        return data.max()

    def entropy1(labels, base=None):
       value,counts = np.unique(labels, return_counts=True)
       return entropy(counts, base=base)

    # def entropy2(labels, base=None):
    #     """ Computes entropy of label distribution. """

    #     n_labels = len(labels)

    #     if n_labels <= 1:
    #         return 0

    #     value,counts = np.unique(labels, return_counts=True)
    #     probs = counts / n_labels
    #     n_classes = np.count_nonzero(probs)

    #     if n_classes <= 1:
    #         return 0

    #     ent = 0.

    #     # Compute entropy
    #     base = e if base is None else base
    #     for i in probs:
    #         ent -= i * log(i, base)

    #     return ent

    # def entropy3(labels, base=None):
    #     vc = pd.Series(labels).value_counts(normalize=True, sort=False)
    #     base = e if base is None else base
    #     return -(vc * np.log(vc)/np.log(base)).sum()

    # def entropy4(labels, base=None):
    #     value,counts = np.unique(labels, return_counts=True)
    #     norm_counts = counts / counts.sum()
    #     base = e if base is None else base
    #     return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

    #Data Collection

    mse = []
    snr_data = []
    snr_ori = []
    psnr_data = []
    psnr = []

    J=0 # jumlah file
    directory_path = 'dataset/sehat'
    for iter in range(0,2):
        for x in os.listdir(directory_path):
            if not x.lower().endswith('.csv'):
                continue
            J=J+1
        directory_path = 'dataset/pasien'
    n = J #jumlah file
    m = 11 
    FEAT = [] #bakal jadi Feature.csv
    for i in range(n): 
        FEAT.append([0] * m) #mengisi dengan angka 0 semua
 
    directory_path = 'dataset/sehat'
    J=-1
    K=0
    for iter in range(0,2):
        for x in os.listdir(directory_path):
            if not x.lower().endswith('.csv'):
                continue
            full_file_path = directory_path  +   '/'   + x
            J=J+1
            print ('Using file', full_file_path)
            try:
                dataraw = pd.read_csv(full_file_path,index_col='Timestamp', parse_dates=['Timestamp'])
                dataset = pd.DataFrame(dataraw['Value']) #ambil kolom value dari setiap file
            except:
                dataraw = pd.read_csv(full_file_path,index_col='timestamp', parse_dates=['timestamp'])
                dataset = pd.DataFrame(dataraw['values'])
            x1=np.array(dataset)  #ubah jadi array, namanya x1
            Dat=[]
            Dat=[0 for i in range(x1.size)] #bikin array kosong isinya 0 semua sepajang array x1
            for i in range(0,x1.size-1):
                Dat[i]=max(x1[i]) #why pakai max?
                
            # Savitzky-Golay filter
            y1_filtered = savgol_filter(Dat,1001,3) #why 1001
            fs = 1000
            cutoff_high = 0.5
            cutoff_low = 2
            powerline = 60
            # order = 5
            # filter_signal = final_filter(Dat, fs, order) #why ada disini?
            # Butterworth filter
            fs = 1000
            cutoff_high = 0.5
            cutoff_low = 2
            powerline = 60
            order = 5
            filter_signal = final_filter(Dat, fs, order)
            y2_filtered=[]
            y2_filtered=[0 for i in range(filter_signal.size)]
            for i in range(0,filter_signal.size-1):
                y2_filtered[i]=filter_signal[i]+cutoff_high

            # FIR Filter
            # The Nyquist rate of the signal.
            nyq_rate = fs / 2.0

            # The desired width of the transition from pass to stop,
            # relative to the Nyquist rate.  We'll design the filter
            # with a 5 Hz transition width.
            width = 5.0/nyq_rate
    
            # The desired attenuation in the stop band, in dB.
            # Compute the order and Kaiser parameter for the FIR filter.
            N, beta = kaiserord(powerline, width)
            # The cutoff frequency of the filter.
            cutoff_hz = cutoff_low
            # Use firwin with a Kaiser window to create a lowpass FIR filter.
            taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
            # Use lfilter to filter x with the FIR filter.
            y3_filtered = lfilter(taps, 1.0, Dat)
            if (metoda==1):
                # Savitzky-Golay filter
                y_filtered=y1_filtered
            if (metoda==2):
                # Butterworth filter
                y_filtered=np.array(y2_filtered)
            if (metoda==3):
                # FIR Filter
                y_filtered=y3_filtered

            # Mse
            mse.append(mean_squared_error(Dat, y_filtered))
            newfile = np.savetxt("mse.csv", np.dstack((np.arange(1, len(mse) + 1), mse))[0], "%.20f,%.20f",
                                 header="Id,Values")

            # SNR
            snr_ori.append(np.mean(Dat) / np.std(Dat))
            snr_data.append(np.mean(y_filtered) / np.std(y_filtered))
            newfile = np.savetxt("snr.csv", np.dstack((np.arange(1, len(snr_data) + 1), snr_data))[0], "%.20f,%.20f",
                                 header="Id,Values")

            #PSNR
            # psnr_data.append(20*math.log10(max(Dat)/math.sqrt(mse)))
            psnr.append(20*log10(max(Dat)/sqrt(mean_squared_error(Dat,y_filtered))))
            newfile = np.savetxt("psnr.csv", np.dstack((np.arange(1, len(psnr) + 1), psnr))[0], "%.20f,%.20f",
                                 header="Id,Values")

            # FEATURE EXTRACTION
            try:
                # Arithmetic Maximum
                MAKS=Max(y_filtered)
                # Arithmetic Mean
                MEAN=Mean(y_filtered)
                # Arithmetic Standard Deviation
                STD=Std(y_filtered)
                # Kurtosis
                KURT = kurtosis(y_filtered)
                # Entropy
                ENT = entropy1(y_filtered)
                # Skewness
                SKEW = skew(y_filtered)
                # First Difference
                FD = np.diff(y_filtered)
                # Normalized First Difference
                NLD = FD/STD

                N = y_filtered.size
                Y = 0
                #Mean Curve Length
                for m in range(1,N-1):
                    Y = Y + abs(y_filtered[m] - y_filtered[m-1])

                MCL = (1 / N) * Y;
                # First derivative
                x0  = y_filtered;
                x1  = np.diff(y_filtered)
                # Standard deviation
                sd0 = Std(x0)
                sd1 = Std(x1)
                HM  = sd1 / sd0
                FEAT[J][0] = MAKS
                FEAT[J][1] = STD
                FEAT[J][2] = MEAN
                FEAT[J][3] = KURT
                FEAT[J][4] = ENT
                FEAT[J][5] = SKEW
                FEAT[J][6] = np.mean(FD)
                FEAT[J][7] = np.mean(NLD)
                FEAT[J][8] = MCL
                FEAT[J][9] = HM
                FEAT[J][10] = K
            except:
                J=J-1
        directory_path = 'dataset/pasien'
        K=1

    #sehat = 0, pasien = 1

    print("Jumlah Data", len(y_filtered), "\n")

    print("MSE Data Latih")
    print("List Data : ", mse)
    print("Nilai terbesar dan terkecil : ", max(mse), ' , ', min(mse))
    print("Rata rata mse: ", np.mean(mse))
    print("Median \t: ", np.median(mse))

    # print("SNR Data Awal")
    # print("Rata rata SNR : ", np.mean(snr_ori))

    print("SNR Denoise")
    print("List Data sebelum Denoising : ", snr_ori)
    print("List Data : ", snr_data)
    print("Nilai terbesar dan terkecil : ", max(snr_data), ' , ', min(snr_data))
    print('Rata rata SNR : ', np.mean(snr_data), "\n")

    print("PSNR Denoise")
    print("List Data sebelum Denoising : ", snr_ori)
    print("Nilai PSNR : ", psnr)
    print("Nilai terbesar dan terkecil : ", max(psnr), ' , ', min(psnr))
    print('Rata rata PSNR : ', np.mean(psnr), "\n")


    tombol["state"] = NORMAL
    print('Selesai ...')




var1 = StringVar()
label1 = Label(window, textvariable=var1, relief=RAISED ,width = 107,bg ='cyan'  )

var1.set("Metoda denoding : ")
label1.place(x=20, y=25)
v0=IntVar()
v0.set(1)
r1=Radiobutton(window, text="Savitzky-Golay", variable=v0,value=1)
r2=Radiobutton(window, text="Butterworth", variable=v0,value=2)
r3=Radiobutton(window, text="FIR", variable=v0,value=3)
r1.place(x=20,y=50)
r2.place(x=140, y=50)
r3.place(x=260, y=50)
var2 = StringVar()
label2 = Label(window, textvariable=var2, relief=RAISED ,width = 107,bg ='cyan'  )

var2.set("Fitur yang dipilih : ")
label2.place(x=20, y=100)
 
v1 = IntVar()
v2 = IntVar()
v3 = IntVar()
v4 = IntVar()
v5 = IntVar()
C1 = Checkbutton(window, text = "Arithmetic Maximum", variable = v1)
C2 = Checkbutton(window, text = "Arithmetic Standard Deviation", variable = v2)
C3 = Checkbutton(window, text = "Arithmetic Mean", variable = v3)
C4 = Checkbutton(window, text = "Kurtosis", variable = v4)
C5 = Checkbutton(window, text = "Entropy", variable = v5)
C1.place(x=20, y=125)
C2.place(x=180, y=125)
C3.place(x=340, y=125)
C4.place(x=500, y=125)
C5.place(x=660, y=125)
v6 = IntVar()
v7 = IntVar()
v8 = IntVar()
v9 = IntVar()
v10 = IntVar()

C6 = Checkbutton(window, text = "Skewness", variable = v6)
C7 = Checkbutton(window, text = "First Difference ", variable = v7)
C8 = Checkbutton(window, text = "Normalized First Difference", variable = v8)
C9 = Checkbutton(window, text = "Mean Curve Length", variable = v9)
C10 = Checkbutton(window, text = "Hjorth Mobility", variable = v0)
C6.place(x=20, y=150)
C7.place(x=180, y=150)
C8.place(x=340, y=150)
C9.place(x=500, y=150)
C10.place(x=660, y=150)

tombol = Button(window,
                   text="RUN Klasifikasi",
                   command=tombol_klik,bg='blue',fg='white')
tombol.place(x=350, y=200)

window.title('Klasifikasi PPG Machine Learning')
window.geometry("800x300+10+10")
window.mainloop()

