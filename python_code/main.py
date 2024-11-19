from scipy.io import loadmat
from scipy.signal import lfilter, firwin
import numpy as np
from functii1 import swPemAFCinit, swPemAFC_Modified_SC_Full
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    start = time.time()
    sel_alg = 2  # sel_alg = 0 distorsiuni mai lungi --> APA
    in_sig = 1  # 0) white noise, 1) speech weighted noise, 2) real speech, 3) music
    fs = 16000  # sampling frequency # trebuie verificata frecventa fisierelor Noizeu
    N = 80 * fs  # total number of samples  --> ar trebui modificat
    # APA parameter
    P = 6  # Projection order
    # nu se modifica
    Kdb = 30  # gain of forward path in dB
    K = 10 ** (Kdb / 20)
    d_k = 96  # delay of the forward path K(q) in samples
    d_fb = 1  # delay of the feedback cancellation path in samples
    Lg_hat = 64  # the full length of adaptive filter
    SNR = 20  # amplified signal to injected noise ratio
    delta = 1e-6
    UpdateFC = 1
    mu = 0.0008  # fixed step size
    mu1 = 0.0008
    mu2 = 0.8

    mFBPathIRs16kHz_FF = loadmat('./data_files/mFBPathIRs16kHz_FF.mat')['mFBPathIRs16kHz_FF']
    E = mFBPathIRs16kHz_FF[:, 2, 0, 0]
    g = E - np.mean(E)  # feedback path and remove mean value

    Lg = len(g)  # length of feedback path.
    Nfreq = 512
    G = np.fft.fft(g, Nfreq)

    mFBPathIRs16kHz_PhoneNear = loadmat('./data_files/mFBPathIRs16kHz_PhoneNear.mat')['mFBPathIRs16kHz_PhoneNear']
    Ec = mFBPathIRs16kHz_PhoneNear[:, 2, 0]
    gc = Ec - np.mean(Ec)  # feedback path and remove mean value
    Gc = np.fft.fft(gc, Nfreq)

    TDLy = np.zeros(Lg)  # time-delay vector true feedback path

    # Desired Signal (incoming signal)
    if in_sig == 0:
        # 0) incoming signal is a white noise
        Var_noise = 0.001
        np.random.seed(1)
        inputs = np.sqrt(Var_noise) * np.random.randn(N)
    elif in_sig == 1:
        # 1) incoming signal is a synthesized speech
        Var = 1
        h_den = np.array([1, -2 * 0.96 * np.cos(3000 * 2 * np.pi / 15750), 0.96 ** 2])
        # h_den = np.array([1,-0.8])     # a first-order system
        np.random.seed(1)
        v = np.sqrt(Var) * np.random.randn(N)  # v[k] is white noise with variance one
        inputs = lfilter([1], h_den, v, axis=0)  # speech weighted noise
    elif in_sig == 2:
        # 2) incoming signal is a real speech segment from NOIZEUS
        HeadMid2_Speech_Vol095_0dgs_m1 = loadmat('./data_files/HeadMid2_Speech_Vol095_0dgs_m1.mat')[
            'HeadMid2_Speech_Vol095_0dgs_m1']
        input1 = HeadMid2_Speech_Vol095_0dgs_m1
        inputs = input1[int(0.5 * fs) - 1:]

    else:
        # 3) incoming signal is a music
        HeadMid2_Music_Vol095_0dgs_m1 = loadmat('./data_files/HeadMid2_Music_Vol095_0dgs_m1.mat')[
            'HeadMid2_Music_Vol095_0dgs_m1']
        input1 = HeadMid2_Music_Vol095_0dgs_m1  # 80s
        inputs = input1[fs - 1:]

    # 2) incoming signal is a real speech segment from NOIZEUS
    # HeadMid2_Speech_Vol095_0dgs_m1 = loadmat('./data_files/HeadMid2_Speech_Vol095_0dgs_m1.mat')[
    #     'HeadMid2_Speech_Vol095_0dgs_m1']
    # input1 = HeadMid2_Speech_Vol095_0dgs_m1
    # inputs = input1[int(0.5 * fs) - 1:]
    inputs = inputs / np.max(np.abs(inputs))
    # print(f"L={len(inputs)}")
    ff = firwin(numtaps=65, cutoff=.025, pass_zero=False)
    # fix from https://stackoverflow.com/questions/16936558/matlab-filter-not-compatible-with-python-lfilter
    u_ = lfilter(ff, 1, inputs, axis=0)
    u = np.zeros(N)
    delta_eye = delta * np.eye(P)
    for k in range(N):
        # loop through input signal
        if k <= len(u_) - 1:
            u[k] = u_[k].item()
        else:
            u[k] = u_[np.remainder(k, len(u_)), 0]

    Var_P = 0.001
    w = np.zeros(N)  # Without probe signal
    err = 0
    La = 20
    framelength = 0.01 * fs
    AF, AR = swPemAFCinit(Lg_hat, mu, mu1, mu2, La, framelength, P, delta)

    # PEM algorithm
    #########################

    ##############################
    # initialisation data vectors
    ##############################

    y = np.zeros(N)  # loudspeaker signal
    e_delay = np.zeros(N + d_k)
    u_delay = np.zeros(N + d_k)
    y_delayfb = np.zeros(N + d_fb)
    m = np.zeros(N)  # received microphone signal

    step = int(np.floor(N / 100))
    for k in range(N):

        if k == N / 2 - 1:
            g = gc
            G = Gc
        if k % step == 0:
            print(f"{k / step + 1}%")
        y[k] = K * e_delay[k] + w[k]
        u_delay[k + d_k] = u[k]

        TDLy[1:] = TDLy[:-1]
        TDLy[0] = y[k]

        m[k] = u[k] + g.T @ TDLy  # received microphone signal
        y_delayfb[k + d_fb] = y[k]

        e_delay[k + d_k], AF, AR = swPemAFC_Modified_SC_Full(m[k], y_delayfb[k], AF, AR, UpdateFC, sel_alg, delta_eye)

        if k >= N - 6:
            print(f"k={k + 1} y={y[k]:.12f} u={u[k]:.12f} m={m[k]:.12f} e_delay={e_delay[k + d_k]:.12f} ")
    end = time.time()
    # print(f"d={end-start:.6f} seconds")
    t = np.linspace(0, (N - 1) / fs, N)
    plt.figure(1)
    plt.plot(t, y)
    plt.grid(True)
    plt.show()
