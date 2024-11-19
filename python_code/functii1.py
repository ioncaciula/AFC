import numpy as np
import numba as nb

#adapted from https://pyspectrum.readthedocs.io/en/latest/_modules/spectrum/levinson.html
@nb.jit(nopython=True)
def levinson(r, order=None, allow_singularity=False):
    T0 = r[0] * 1.0
    T = r[1:]
    M = order

    A = np.zeros(M + 1, dtype=float)
    ref = np.zeros(M, dtype=float)
    A[0] = 1
    P = T0

    for k in range(M):
        save = T[k]
        if k == 0:
            temp = -save / P
        else:
            # save += sum([A[j]*T[k-j-1] for j in range(0,k)])
            for j in range(k):
                save = save + A[j + 1] * T[k - j - 1]
            temp = -save / P

        P = P * (1. - temp ** 2.)

        if P <= 0 and not allow_singularity:
            raise ValueError("singular matrix")
        A[k + 1] = temp
        ref[k] = temp  # save reflection coeff at each step
        if k == 0:
            continue

        khalf = (k + 1) // 2

        for j in range(0, khalf):
            kj = k - j - 1
            save = A[j + 1]
            A[j + 1] = save + temp * A[kj + 1]
            if j != kj:
                A[kj + 1] += temp * save

    return A, P, ref


def swPemAFCinit(N, mu, mu1, mu2, N_ar, framelength_ar, P, delta):
    # Initialization of time-domain LMS-based implementation of PEM-based
    # adaptive feedback canceller Used in combination with swPemAFC_Modified_SC.m
    #
    # INPUTS:
    # * N    = filter length of the adaptive feedback canceller
    # * mu   = step-size of the NLMS-based adaptive feedback canceller
    # * N_ar = filter length of the AR model
    # * framelength_ar = framelength in number of samples on which the AR model is estimated
    #
    # OUTPUTS:
    # * AF        = time-domain feedback canceller and its properties
    #                -AF['wTD']: coefficients of the time-domain feedback canceller
    #                -AF['N']  : filter length
    #                -AF['mu'] : step size
    #                -AF['p']  : power in step-size normalization
    #                -AF['lambda']: weighing factor for power computation
    #                -AF['TDLLs']: time-delay line of loudspeaker samples
    #                           (dimensions: AF['N x 1)
    #                -AF['TDLLswh']: time-delay line of pre-whitened loudspeaker
    #                             samples (dimensions: AF['N x 1)
    # * AR        = auto-regressive model and its properties
    #                -AR['w']           : coefficients of previous AR-model
    #                -AR['N']           : filter length AR model (Note: N=Nh+1)
    #                -AR['framelength'] : framelength on which AR model is estimated
    #                -AR['TDLMicdelay'] : time-delay line of microphone samples
    #                                  (dimensions: AR['framelength+1 x 1)
    #                -AR['TDLLsdelay']  : time-delay line of loudspeaker samples
    #                                  (dimensions: AR['framelength+1 x 1)
    #                -AR['TDLMicwh']    : time-delay line of pre-whitened
    #                                  microphone signal
    #                                  (dimensions: AR['N x 1)
    #                -AR['TDLLswh']     : time-delay line of pre-whitened
    #                                  loudspeaker signal
    #                                  (dimensions: AR['N x 1)
    #                -AR['frame']       : frame of AR['framelength error signals
    #                                  on which AR model is computed
    #                -AR['frameindex']
    #
    #
    #
    # Date: December, 2007
    # Copyright: (c) 2007 by Ann Spriet
    # e-mail: ann.spriet@esat.kuleuven.be
    # modified by Linh Tran
    AR = dict()
    AF = dict()
    AR['N'] = int(N_ar)
    AR['w'] = np.zeros(AR['N'])
    AR['w'][0] = 1
    AR['framelength'] = int(framelength_ar)
    AR['frameindex'] = -1
    AR['TDLMicdelay'] = np.zeros(AR['framelength'] + 1)
    AR['TDLLsdelay'] = np.zeros(AR['framelength'] + 1)
    AR['TDLMicwh'] = np.zeros(AR['N'])
    AR['TDLLswh'] = np.zeros(AR['N'])
    AR['frame'] = np.zeros(AR['framelength'])

    AF['gTD'] = np.zeros(N)
    AF['N'] = N
    AF['P'] = P
    AF['mu'] = mu
    AF['mu1'] = mu1
    AF['mu2'] = mu2
    AF['delta'] = delta
    AF['TDLLs'] = np.zeros(N)
    AF['TDLLswh'] = np.zeros(N)
    AF['TDLLswh_d'] = np.zeros(N + P - 1)
    AF['Lswh_ap'] = np.zeros((N, P))
    AF['TDLMicwh'] = np.zeros(P)

    AF['TDLLs_d'] = np.zeros(N + P - 1)
    AF['Ls_ap'] = np.zeros((N, P))
    AF['TDLMic'] = np.zeros(P)

    AF['pow_Micwh'] = 0
    AF['pow_ep'] = 0
    AF['pow_Lswh'] = 0
    AF['pow_vp_hat'] = 0

    eps = 1e-5
    AF['R_mu'] = eps * np.eye(N)
    AF['pow_w'] = 1e-10
    AF['r_mu'] = eps
    return AF, AR


def FilterSample(x, w, delayline_in):
    #
    # Sample x is filtered with filter w. If x has multiple columns,
    # each column of x is filtered with w. If w has multiple colums,
    # the data x is filtered with both columns of w.
    #
    # Inputs:
    #   x             = input data (Dimensions:1xnr_channels)
    #   w             = time-domain filter coefficients (Dimensions:filterlength x nr_filters)
    #   delayline_in  = input delayline (Dimensions:filterlength x nr_channels)
    # Outputs:
    #   output        = output data (Dimensions: 1 x max(nr_channels,nr_filters)
    #   delayline_out = output delayline
    #
    # Date: December, 2007
    # Copyright: (c) 2007 by Ann Spriet
    # e-mail: ann.spriet@esat.kuleuven.be
    nr_channels = 1
    nr_filters = 1

    if len(x.shape) > 1:
        nr_channels = x.shape[1]
    if len(w.shape) > 1:
        nr_filters = w.shape[1]
    if nr_filters > 1 and nr_channels > 1:
        print('Nr_channels and nr_filters cannot be both larger than 1')
        exit(0)

    delayline_out = np.copy(delayline_in)
    if nr_channels > 1:  # x(1x2) w(1xK) delayline_in(Kx2)
        # delayline_out = [x;delayline_in(1:end-1,:)];
        delayline_out[1:, :] = delayline_out[:-1, :]
        delayline_out[0, :] = x
        output = np.matmul(w, delayline_out)
    else:  # x(1x1) w(KxnrF) delayline_in(K)
        delayline_out[1:] = delayline_out[:-1]
        delayline_out[0] = x
        output = np.matmul(delayline_out, w)
    return output, delayline_out


def DelaySample(x, delay, delayline_in):
    # Delays sample x with delay.
    #
    # INPUTS:
    #   x             = input data (Dimensions:1xnr_channels)
    #   delay         = discrete-time delay
    #   delayline_in  = input delayline (Dimensions:(delay+1)xnr_channels)
    # OUTPUTS:
    #   output        =  output data (Dimensions: 1xnr_channels)
    #   delayline_out = output delayline
    #
    # Date: December, 2007
    # Copyright: (c) 2007 by Ann Spriet
    # e-mail: ann.spriet@esat.kuleuven.be
    if len(delayline_in.shape) == 1:
        delayline_out = np.concatenate((np.array([x]), delayline_in[:-1]))
        output = delayline_out[delay]
    else:
        delayline_out = np.concatenate((np.array([x]), delayline_in[:-1, :]), axis=0)
        output = np.copy(delayline_out[delay, :])
    return output, delayline_out


@nb.jit(nopython=True)
def AP_alg(L, Nh, TDL_u_in):
    # Affine Projection Filter

    # u[n] - Input signal, dimension: Nx1
    # Uap - Affine input matrix, dimenson: NxL
    # L - projection order
    # Nh - filter length
    # N - total number of samples
    # d[n]-desired signal
    # w[n] - adaptive filter coefficients vector
    # e[n] - error at step n
    # y[n] - adaptive filter output

    # Set up parameters

    # N=40                               # total number of samples
    # L=4                              # projection order
    # Nh=8                               # the length of filter
    # mu_ap=0.01                         # step size of APA
    # offset=10**(-3)

    # d=np.ramdom.randn(Lpo)
    # u=np.random.randn(N)
    # w=np.zeros(Lg)
    # e=np.zeros(N)
    # y=np.zeros(N)

    # delay=0
    # u_delay=np.zeros(N+delay)
    u_ap = np.zeros((Nh, L))
    # TDL_u_in=np.zeros(N*L)
    # TDL_u_out=np.zeros(N*L)

    # for k in range(N):
    #     u[k+delay]=u[k]
    #     TDL_u_out=np.copy(uTDL_u_in[:-1])      # length N*Lpo

    for i in range(L):
        #     print(f"i={i} TDL_u_in[i:(Nh + i)= {TDL_u_in[i:(Nh + i)].shape}")
        u_ap[:, i] = TDL_u_in[i:(Nh + i)]
    return u_ap


def swPemAFC_Modified_SC_(Mic, Ls, AF, AR, UpdateFC, delta_eye):
    #
    # Update equation of time-domain LMS-based implementation of PEM-based
    # adaptive feedback canceller
    #
    # INPUTS:
    # * Mic = microphone sample
    # * Ls  = loudspeaker sample
    # * AF  = time-domain LMS-based feedback canceller and its properties
    #          -AF['wTD']: time-domain filter coefficients (dimensions: AF['N')
    #          -AF['N'] : time-domain filter length
    #          -AF['mu'] : stepsize
    #          -AF['p']  : power in step-size normalization
    #          -AF['lambda']: weighing factor for power computation
    #          -AF['TDLLs']: time-delay line of loudspeaker samples# (dimensions: AF['N'])
    #          -AF['TDLLswh']: time-delay line of pre-whitened loudspeaker
    #          samples (dimensions: AF['N'])
    #
    # * AR        = auto-regressive model and its properties
    #                -AR['w']           : coefficients of previous AR-model
    #                -AR['N']           : filter length AR model (Note: N=Nh+1)
    #                -AR['framelength'] : framelength on which AR model is estimated
    #                -AR['TDLMicdelay'] : time-delay line of microphone samples
    #                                  (dimensions: AR['framelength+1'])
    #                -AR['TDLLsdelay']  : time-delay line of loudspeaker samples
    #                                  (dimensions: AR['framelength+1'])
    #                -AR['TDLMicwh']    : time-delay line of pre-whitened
    #                                  microphone signal
    #                                  (dimensions: AR['N'])
    #                -AR['TDLLswh']     : time-delay line of pre-whitened
    #                                  loudspeaker signal
    #                                  (dimensions: AR['N'])
    #                -AR['frame']       : frame of AR['framelength'] error signals
    #                                  on which AR model is computed
    #                -AR['frameindex']
    # * UpdateFC = boolean that indicates whether or not the feedback canceller should be updated
    #                    (1 = update feedback canceller; 0 = do not update feedback canceller)
    # * RemoveDC = boolean that indicates whether or not the DC component of the estimated feedback
    #              path should be removed
    #                    (1 = remove DC of feedback canceller; 0 = do not remove DC of feedback canceller)
    # OUTPUTS:
    # * e          = feedback-compensated signal
    # * AR         = updated AR-model and its properties
    # * AF         = updated feedback canceller and its properties
    #
    #
    #
    #####################

    AF['TDLLs'][1:] = AF['TDLLs'][:-1]
    AF['TDLLs'][0] = Ls
    e = Mic - AF['gTD'] @ AF['TDLLs']
    hv = e
    # limit the error signal
    e = 2 * np.tanh(0.5 * e)
    aaa = (np.abs(hv - e) < .15)
    # print(f"Mic={Mic:.12f} e={e:.12f}")
    # print(f"{AF['gTD'][0]:.12f}")
    # print(f"{AF['gTD'][1]:.12f}")
    # print(f"{AF['gTD'][2]:.12f}")
    # Delay microphone and loudspeaker signal by framelength
    Micdelay, AR['TDLMicdelay'] = DelaySample(Mic, AR['framelength'], AR['TDLMicdelay'])
    Lsdelay, AR['TDLLsdelay'] = DelaySample(Ls, AR['framelength'], AR['TDLLsdelay'])
    #
    # Filter microphone and loudspeaker signal with AR-model
    Micwh, AR['TDLMicwh'] = FilterSample(Micdelay, AR['w'], AR['TDLMicwh'])
    Lswh, AR['TDLLswh'] = FilterSample(Lsdelay, AR['w'], AR['TDLLswh'])
    # print(f"{Micdelay} {np.sum(np.abs(AR['TDLMicdelay']))} {Lsdelay} {np.sum(np.abs(AR['TDLLsdelay']))}")

    # Update AR-model
    AR['frame'][1:] = AR['frame'][:-1]
    AR['frame'][0] = e
    # print(f"frameindex = {AR['frameindex']+1} framelength={AR['framelength']} N={AR['N']} e={e:.12f}")

    if AR['frameindex'] == AR['framelength'] - 2 and AR['N'] - 1 > 0:
        # print(AR['frame'][0:5])
        R = np.zeros(AR['N'])
        for j in range(AR['N']):
            R[j] = AR['frame'] @ np.concatenate((AR['frame'][j:len(AR['frame'])], np.zeros(j)))
            R[j] /= AR['framelength']
        a, Ep, _ = levinson(R, AR['N'] - 1)
        AR['w'] = a.T

    AR['frameindex'] = AR['frameindex'] + 1

    if AR['frameindex'] == AR['framelength'] - 1:
        AR['frameindex'] = -1

    AF['TDLLswh'][1:] = AF['TDLLswh'][:-1]
    AF['TDLLswh'][0] = Lswh
    ep = Micwh - AF['gTD'] @ AF['TDLLswh']

    if UpdateFC == 1:
        # APA alg. with pre-filters
        AF['TDLMicwh'][1:] = AF['TDLMicwh'][:-1]
        AF['TDLMicwh'][0] = Micwh

        AF['TDLLswh_d'][1:] = AF['TDLLswh_d'][:-1]
        AF['TDLLswh_d'][0] = Lswh
        # print(f"ep={ep:.12f} Micwh={Micwh:.12f} Lswh={Lswh:.12f}")
        AF['Lswh_ap'] = AP_alg(AF['P'], AF['N'], AF['TDLLswh_d'])  # size=(Lg_hat,P);
        # print(f"Lswh_ap.shape = {AF['Lswh_ap'].shape} gTD.shape={AF['gTD'].shape}")
        ewh_p = AF['TDLMicwh'] - AF['Lswh_ap'].T @ AF['gTD']  # size=(P,1);

        # APA alg. without pre-filters
        AF['TDLMic'][1:] = AF['TDLMic'][:-1]  # size=(P,1);
        AF['TDLMic'][0] = Mic

        AF['TDLLs_d'][1:] = AF['TDLLs_d'][:-1]  # size=(Lg_hat+P-1,1);
        AF['TDLLs_d'][0] = Ls

        AF['Ls_ap'] = AP_alg(AF['P'], AF['N'], AF['TDLLs_d'])  # size=(Lg_hat,P);)
        # e_ap = AF['TDLMic'] - AF['Ls_ap'].T @ AF['gTD']  # size=(P,1);

        # classical NLMS and APA (P=1 -> NLMS-l1; P=2 -> APA-l1)
        PEMSC_APA_term = AF['Lswh_ap'] @ np.linalg.inv(
            AF['Lswh_ap'].T @ AF['Lswh_ap'] + delta_eye) @ ewh_p
        # print(f"PEMSC_APA_term={np.sum(PEMSC_APA_term)}")
        AF['gTD'] = AF['gTD'] + AF['mu'] * PEMSC_APA_term

        # Remove DC
        AF['gTD'] = AF['gTD'] - np.mean(AF['gTD'])
    return e, AF, AR


def swPemAFC_Modified_SC_Full(Mic, Ls, AF, AR, UpdateFC, sel_alg, delta_eye):
    #
    # Update equation of time-domain LMS-based implementation of PEM-based
    # adaptive feedback canceller
    #
    # INPUTS:
    # * Mic = microphone sample
    # * Ls  = loudspeaker sample
    # * AF  = time-domain LMS-based feedback canceller and its properties
    #          -AF['wTD']: time-domain filter coefficients (dimensions: AF['N')
    #          -AF['N'] : time-domain filter length
    #          -AF['mu'] : stepsize
    #          -AF['p']  : power in step-size normalization
    #          -AF['lambda']: weighing factor for power computation
    #          -AF['TDLLs']: time-delay line of loudspeaker samples# (dimensions: AF['N'])
    #          -AF['TDLLswh']: time-delay line of pre-whitened loudspeaker
    #          samples (dimensions: AF['N'])
    #
    # * AR        = auto-regressive model and its properties
    #                -AR['w']           : coefficients of previous AR-model
    #                -AR['N']           : filter length AR model (Note: N=Nh+1)
    #                -AR['framelength'] : framelength on which AR model is estimated
    #                -AR['TDLMicdelay'] : time-delay line of microphone samples
    #                                  (dimensions: AR['framelength+1'])
    #                -AR['TDLLsdelay']  : time-delay line of loudspeaker samples
    #                                  (dimensions: AR['framelength+1'])
    #                -AR['TDLMicwh']    : time-delay line of pre-whitened
    #                                  microphone signal
    #                                  (dimensions: AR['N'])
    #                -AR['TDLLswh']     : time-delay line of pre-whitened
    #                                  loudspeaker signal
    #                                  (dimensions: AR['N'])
    #                -AR['frame']       : frame of AR['framelength'] error signals
    #                                  on which AR model is computed
    #                -AR['frameindex']
    # * UpdateFC = boolean that indicates whether or not the feedback canceller should be updated
    #                    (1 = update feedback canceller; 0 = do not update feedback canceller)
    # * RemoveDC = boolean that indicates whether or not the DC component of the estimated feedback
    #              path should be removed
    #                    (1 = remove DC of feedback canceller; 0 = do not remove DC of feedback canceller)
    # OUTPUTS:
    # * e          = feedback-compensated signal
    # * AR         = updated AR-model and its properties
    # * AF         = updated feedback canceller and its properties
    #
    #
    #
    #####################

    AF['TDLLs'][1:] = AF['TDLLs'][:-1]
    AF['TDLLs'][0] = Ls
    e = Mic - AF['gTD'] @ AF['TDLLs']
    hv = e
    e = 2 * np.tanh(0.5 * e) # limit the error signal
    aaa = (np.abs(hv - e) < .15)

    # Delay microphone and loudspeaker signal by framelength
    Micdelay, AR['TDLMicdelay'] = DelaySample(Mic, AR['framelength'], AR['TDLMicdelay'])
    Lsdelay, AR['TDLLsdelay'] = DelaySample(Ls, AR['framelength'], AR['TDLLsdelay'])
    #
    # Filter microphone and loudspeaker signal with AR-model
    Micwh, AR['TDLMicwh'] = FilterSample(Micdelay, AR['w'], AR['TDLMicwh'])
    Lswh, AR['TDLLswh'] = FilterSample(Lsdelay, AR['w'], AR['TDLLswh'])

    # Update AR-model
    AR['frame'][1:] = AR['frame'][:-1]
    AR['frame'][0] = e

    if AR['frameindex'] == AR['framelength'] - 2 and AR['N'] - 1 > 0:
        R = np.zeros(AR['N'])
        for j in range(AR['N']):
            R[j] = AR['frame'] @ np.concatenate((AR['frame'][j:len(AR['frame'])], np.zeros(j)))
            R[j] /= AR['framelength']
        a, Ep, _ = levinson(R, AR['N'] - 1)
        AR['w'] = a.T

    AR['frameindex'] = AR['frameindex'] + 1

    if AR['frameindex'] == AR['framelength'] - 1:
        AR['frameindex'] = -1

    AF['TDLLswh'][1:] = AF['TDLLswh'][:-1]
    AF['TDLLswh'][0] = Lswh
    ep = Micwh - AF['gTD'] @ AF['TDLLswh']

    if UpdateFC == 1:
        # APA alg. with pre-filters
        AF['TDLMicwh'][1:] = AF['TDLMicwh'][:-1]
        AF['TDLMicwh'][0] = Micwh

        AF['TDLLswh_d'][1:] = AF['TDLLswh_d'][:-1]
        AF['TDLLswh_d'][0] = Lswh

        AF['Lswh_ap'] = AP_alg(AF['P'], AF['N'], AF['TDLLswh_d'])  # size=(Lg_hat,P);
        ewh_p = AF['TDLMicwh'] - AF['Lswh_ap'].T @ AF['gTD']  # size=(P,1);

        # APA alg. without pre-filters
        AF['TDLMic'][1:] = AF['TDLMic'][:-1]  # size=(P,1);
        AF['TDLMic'][0] = Mic

        AF['TDLLs_d'][1:] = AF['TDLLs_d'][:-1]  # size=(Lg_hat+P-1,1);
        AF['TDLLs_d'][0] = Ls

        AF['Ls_ap'] = AP_alg(AF['P'], AF['N'], AF['TDLLs_d'])  # size=(Lg_hat,P);)
        # e_ap = AF['TDLMic'] - AF['Ls_ap'].T @ AF['gTD']  # size=(P,1);
        if sel_alg == 0:
            # classical NLMS and APA (P=1 -> NLMS-l1; P=2 -> APA-l1)
            PEMSC_APA_term = AF['Lswh_ap'] @ np.linalg.inv(
                AF['Lswh_ap'].T @ AF['Lswh_ap'] + delta_eye) @ ewh_p
            AF['gTD'] = AF['gTD'] + AF['mu'] * PEMSC_APA_term
        elif sel_alg == 1:
            # swPEMSC: switch between PEMSC-NLMS and PEMSC-APA
            PEMSC_NLMS_term = AF['TDLLswh'] * np.conj(ep) / (np.linalg.norm(AF['TDLLswh']) ** 2 + AF['delta'])
            PEMSC_APA_term = AF['Lswh_ap'] @ np.linalg.inv(AF['Lswh_ap'].T @ AF['Lswh_ap'] + delta_eye) @ ewh_p
            AF['gTD'] = AF['gTD'] + aaa * AF['mu1'] * PEMSC_NLMS_term + (1 - aaa) * AF['mu2'] * PEMSC_APA_term
        else:
            # H-NLMS: switch NLMS, large mu (when the system is unstable) and
            # PEMSC-NLMS, small mu (when the system is converged)
            PEMSC_NLMS_term = AF['TDLLswh'] * np.conj(ep) / (np.linalg.norm(AF['TDLLswh']) ** 2 + AF['delta'])
            NLMS_term = AF['TDLLs'] * np.conj(e) / (np.linalg.norm(AF['TDLLs']) ** 2 + AF['delta'])
            AF['gTD'] = AF['gTD'] + aaa * AF['mu1'] * PEMSC_NLMS_term + (1 - aaa) * AF['mu2'] * NLMS_term

        # Remove DC
        AF['gTD'] = AF['gTD'] - np.mean(AF['gTD'])
    return e, AF, AR
