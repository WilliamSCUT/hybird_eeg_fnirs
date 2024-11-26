import numpy as np
from scipy.signal.windows import tukey, gaussian


def gen_STFT(x,L_window,window_type,overlap,Fs, STFT_OR_SPEC=0):
    """
    %-------------------------------------------------------------------------------
    % gen_STFT: Short-time Fourier transform (or spectrogram)
    %
    % Syntax: [S_stft,Nfreq,f_scale,win_epoch]=gen_STFT(x,L_window,window_type,overlap,Fs)
    %
    % Inputs:
    %     x            - input signal
    %     L_window     - window length
    %     window_type  - window type
    %     overlap      - percentage overlap
    %     Fs           - sampling frequency (Hz)
    %     STFT_OR_SPEC - return short-time Fourier transform (STFT) or spectrogram
    %                    (0=spectrogram [default] and 1=STFT)
    %
    % Outputs:
    %     S_stft     - spectrogram
    %     Nfreq      - length of FFT
    %     f_scale    - frequency scaling factor
    %     win_epoch  - window
    %
    % Example:
    %     Fs=64;
    %     data_st=gen_test_EEGdata(32,Fs,1);
    %     x=data_st.eeg_data(1,:);
    %
    %     L_window=2;
    %     window_type='hamm';
    %     overlap=80;
    %
    %     S_stft=gen_STFT(x,L_window,window_type,overlap,Fs);
    %
    %     figure(1); clf; hold all;
    %     imagesc(S_stft); axis('tight');
    %
    %-------------------------------------------------------------------------------
    """
    L_hop,L_epoch,win_epoch = gen_epoch_window(overlap, L_window, window_type, Fs, 1)
    N = len(x)
    N_epochs = int(np.ceil((N-(L_epoch-L_hop))/L_hop))

    if N_epochs < 1:
        N_epochs = 1
    nw = list(range(0,L_epoch))
    Nfreq = L_epoch

    # ---------------------------------------------------------------------
    #  generate short-time FT on all data:
    # ---------------------------------------------------------------------
    K_stft = np.zeros([N_epochs,L_epoch])
    for k in range(N_epochs):
        nf = np.mod(nw+k*L_hop,N)
        nf = nf.astype(int)

        K_stft[k,:] = x[nf] * win_epoch

    f_scale = Nfreq/Fs

    if STFT_OR_SPEC:
        S_stft = np.fft.fft(K_stft, Nfreq, axis=1)
    else:
        S_stft = np.abs(np.fft.fft(K_stft, Nfreq, axis=1))**2  # K_stft(3,128)

    S_stft = S_stft[:, :Nfreq//2+1]

    return S_stft, Nfreq, f_scale, win_epoch

"""
    gen_epoch_window: calculate overlap size (in samples) and window length for
    overlap-and-add type analysis

    Syntax: [L_hop,L_epoch,win_epoch]=gen_epoch_window(L_overlap,L_epoch,win_type,Fs)

    Inputs:
        L_overlap - precentage overlap
        L_epoch   - epoch size (in seconds)
        win_type  - window type, e.g. 'hamm' for Hamming window
        Fs        - sampling frequency (in Hz)

    Outputs:
        L_hop     - hop size (in samples)
        L_epoch   - epoch size (in samples)
        win_epoch - window, of length L_epoch

    Example:
        overlap=50; win_length=2; Fs=64

        [L_hop,L_epoch,win_epoch]=gen_epoch_window(overlap,win_length,'hamm',Fs);

        fprintf('hop length=%d; epoch length=%d\n',L_hop,L_epoch);
        figure(1); clf;
        ttime=(0:(length(win_epoch)-1))./Fs;
        plot(ttime,win_epoch);


    John M. O' Toole, University College Cork
    Started: 28-05-2013

    last update: Time-stamp: <2017-03-14 17:31:08 (otoolej)>
"""


def gen_epoch_window(L_overlap, L_epoch, win_type, Fs, GEN_PSD=0):
    L_hop = (100 - L_overlap) / 100
    L_epoch = int(np.floor(L_epoch * Fs))

    # check for window type to force constant-overlap add constraint
    # i.e. \sum_m w(n-mR)=1 for all n, where R=overlap size
    win_type = win_type[0:4].lower()

    if GEN_PSD:
        # ---------------------------------------------------------------------
        # if PSD
        # ---------------------------------------------------------------------
        L_hop = np.ceil((L_epoch - 1) * L_hop)

        win_epoch = get_window(L_epoch, win_type)
        win_epoch = np.roll(win_epoch, int(np.floor(L_epoch / 2)))
    else:
        # ---------------------------------------------------------------------
        # otherwise, if using an overlap- and -add method, then for window w[n]
        # ∑ₘw[n - mR] = 1 over all n(R=L_hop)
        # Smith, J.O."Overlap-Add (OLA) STFT Processing", in
        # Spectral Audio Signal Processing,
        # http: // ccrma.stanford.edu / ~jos / sasp / Hamming_Window.html, online book,
        # 2011 edition, accessed Nov.2016.

        # there are some restrictions on this:
        # e.g. for Hamming window, L_hop = (L_epoch-1) / 2, (L_epoch-1) / 4, ...
        # ---------------------------------------------------------------------
        if win_type == 'hamm':
            L_hop = (L_epoch - 1) * L_hop
        elif win_type == 'hann':
            L_hop = (L_epoch + 1) * L_hop
        else:
            L_hop = L_epoch * L_hop

        L_hop = np.ceil(L_hop)
        win_epoch = get_window(L_epoch, win_type)
        win_epoch = np.roll(win_epoch, int(np.floor(L_epoch / 2)))

        if (win_type.lower() == 'hamm') and (L_epoch % 2 == 1):
            win_epoch[1] = win_epoch[1] / 2
            win_epoch[-1] = win_epoch[-1] / 2
        elif win_type.lower() == 'rect':
            win_epoch = np.transpose(win_epoch)

    return L_hop, L_epoch, win_epoch


"""
function [L_hop,L_epoch,win_epoch]=gen_epoch_window(L_overlap,L_epoch,win_type,Fs, ...
                                                    GEN_PSD)
if(nargin<5 || isempty(GEN_PSD)), GEN_PSD=0; end


L_hop=(100-L_overlap)/100;


L_epoch=floor( L_epoch*Fs );

% check for window type to force constant-overlap add constraint
% i.e. \sum_m w(n-mR)=1 for all n, where R=overlap size
win_type=lower(win_type(1:4));


if(GEN_PSD)
    %---------------------------------------------------------------------
    % if PSD 
    %---------------------------------------------------------------------
    L_hop=ceil( (L_epoch-1)*L_hop );

    win_epoch=get_window(L_epoch,win_type);
    win_epoch=circshift(win_epoch,floor(length(win_epoch)/2));

else
    %---------------------------------------------------------------------
    % otherwise, if using an overlap-and-add method, then for window w[n]
    % ∑ₘ w[n - mR] = 1 over all n (R = L_hop )
    %
    % Smith, J.O. "Overlap-Add (OLA) STFT Processing", in 
    % Spectral Audio Signal Processing,
    % http://ccrma.stanford.edu/~jos/sasp/Hamming_Window.html, online book, 
    % 2011 edition, accessed Nov. 2016.
    %
    %
    % there are some restrictions on this:
    % e.g. for Hamming window, L_hop = (L_epoch-1)/2, (L_epoch-1)/4, ... 
    %---------------------------------------------------------------------

    switch win_type
      case 'hamm'
        L_hop=(L_epoch-1)*L_hop;
      case 'hann'
        L_hop=(L_epoch+1)*L_hop;
      otherwise
        L_hop=L_epoch*L_hop;
    end

    L_hop=ceil(L_hop);

    win_epoch=get_window(L_epoch,win_type);
    win_epoch=circshift(win_epoch,floor(length(win_epoch)/2));

    if( strcmpi(win_type,'hamm')==1 && rem(L_epoch,2)==1 )
        win_epoch(1)=win_epoch(1)/2;
        win_epoch(end)=win_epoch(end)/2;
    end    
end
"""


def get_window(win_length, win_type, win_param=[], DFT_WINDOW=0, Npad=0):
    win = get_win(win_length, win_type, win_param, DFT_WINDOW)
    win = shift_win(win)

    if Npad > 0:
        win = pad_win(win, Npad)

    return win


def get_win(win_length, win_type, win_param, DFT_WINDOW):
    # --------------------------------------------------------------------------------
    # Get the window. Negative indices are first.
    # --------------------------------------------------------------------------------
    win = np.zeros([1, win_length])
    if (win_type == 'delt') or (win_type == 'delta'):
        wh = np.floor(win_length / 2)
        win[0, wh + 1] = 1
    elif (win_type == 'rect') or (win_type == 'rectangular'):
        win[0, :win_length] = 1
    elif (win_type == 'bart') or (win_type == 'bartlett'):
        win = np.bartlett(win_length)
    elif (win_type == 'hamm') or (win_type == 'hamming'):
        win = np.hamming(win_length)
    elif (win_type == 'hann') or (win_type == 'hanning'):
        win = np.hanning(win_length)
    elif (win_type == 'tuke') or (win_type == 'tukey'):
        if len(win_param) == 0:
            win = tukey(win_length)
        else:
            win = tukey(win_length, win_param)
    elif win_type == 'gauss':
        if len(win_param) == 0:
            win_param = 7
            win = gaussian(win_length, win_param)
        else:
            win = gaussian(win_length, win_param)
    elif win_type == 'cosh':
        win_hlf = int(np.fix(win_length / 2))  # 使浮点数向靠近0的方向取整
        if len(win_param) == 0:
            win_param = 0.01
        for m in range(-win_hlf, win_hlf):
            win[m % win_hlf] = np.power(np.cosh(m), -2 * win_param)
        win = np.fft.fftshift(win)  # 快速傅里叶变换的数值转换功能
    else:
        print('unknown window type')

    if DFT_WINDOW:
        win = np.roll(win, int(np.ceil(win_length / 2)))
        win = np.fft.fft(win)  # 快速傅里叶变换
        win = np.roll(win, int(np.floor(win_length / 2)))

    return win


def shift_win(w):
    """
    Shift the window so that positive indices are first
    使窗的中心位于第一个位置
    """
    N = max(w.shape)
    w = np.roll(w, int(np.ceil(N / 2)))
    return w


def pad_win(w, Npad):
    """
    Pad window to Npad.

    Presume that positive window indices are first.
    在win中填补0

    When N is even use method described in [1]

    References:
        [1] S. Lawrence Marple, Jr., Computing the discrete-time analytic
        signal via FFT, IEEE Transactions on Signal Processing, Vol. 47,
        No. 9, September 1999, pp.2600--2603.
    """
    w_pad = np.zeros([1, Npad])
    w_pad = w_pad[0, :]
    N = max(w.shape)
    Nh = int(np.floor(N / 2))  # 向下取整
    if Npad < N:
        print('Npad is less than N')
    elif N == Npad:
        w_pad = w
    elif N != Npad:
        if N % 2 == 1:
            for n in range(0, Nh):
                w_pad[n] = w[n]
            for n in range(1, Nh):
                w_pad[Npad - n - 1] = w[N - n - 1]
        elif N % 2 == 0:
            for n in range(0, Nh):
                w_pad[n] = w[n]
                w_pad[Nh] = w[Nh - 1] / 2

            for n in range(0, Nh):
                w_pad[Npad - n] = w[N - n - 1]
                w_pad[Npad - Nh] = w[Nh] / 2

    return w_pad