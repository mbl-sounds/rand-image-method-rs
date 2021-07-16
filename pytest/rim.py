
import numpy as np
import librimrs as rimrs


# def test(xr):
#     return rimrs.rim(xr)


def rim(xs, xr, Nt, L, beta, Fs, c=343, Rd=1e-2, N=(0, 0, 0), Tw=20, Fc=0.9, seed=0, MicDirs=None, MicTypes=None):
    """
    `h = rim(xs, xr, L, T60, Nt, Fs)`
    Randomized Image Source Method
    #### Arguments:
    * `xs`  : Source position in meters (must be array-like)
    * `xr`  : Microphone position in meters (must be array-like for mic array)
    * `Nt`  : Time samples
    * `L`   : 3 element array-like containing dimensions of the room
    * `beta`/`T60` : 6 element array-like containing reflection coefficients of walls/reverberation time
    * `Fs`  : sampling frequency
    #### Keyword Arguments:
    * `c = 343`    : speed of sound
    * `Rd = 1e-2`  : random displacement (in meters)
    * `N = (0,0,0)`: 3 element array-like representing order of reflection when `N == (0;0;0)` full order is computed.
    * `Tw = 20`    : taps of fractional delay filter
    * `Fc = 0.9`   : cut-off frequency of fractional delay filter
    #### Outputs:
    * `h`: vector or matrix where each column is an impulse response corresponding to the microphone positions `xr`
    * `seed`: randomization seed to preserve spatial properties when other RIR at different position are needed
    """
    xs = np.asarray(xs, dtype=np.float64)
    assert xs.shape == (3,)

    xr = np.atleast_2d(np.asarray(xr, dtype=np.float64)).T.copy()
    assert xr.shape[0] == 3

    M = xr.shape[1]  # nuber of microphones

    if MicDirs != None:
        mic_dirs = np.atleast_2d(np.asarray(
            MicDirs, dtype=np.float64)).T.copy()
        assert mic_dirs.shape[0] == 3
    else:
        mic_dirs = np.ones(shape=(3, M))

    if MicTypes != None:
        mic_types_ = np.asarray(MicTypes)
        mic_types = np.zeros(shape=mic_types_.shape)
        for i in range(mic_types.size):
            if mic_types_[i] == 'b' or mic_types[i] == 'bidirectional':
                mic_types[i] = 1
            if mic_types_[i] == 'h' or mic_types[i] == 'hypercardoid':
                mic_types[i] = 2
            if mic_types_[i] == 'c' or mic_types[i] == 'cardoid':
                mic_types[i] = 3
            if mic_types_[i] == 's' or mic_types[i] == 'subcardoid':
                mic_types[i] = 4

        assert mic_types.shape[0] == M
    else:
        mic_types = np.zeros(shape=(M,))

    L = np.asarray(L, dtype=np.float64)
    assert L.shape == (3,)

    N = np.asarray(N, dtype=np.float64)
    assert N.shape == (3,)

    beta = np.asarray(beta, dtype=np.float64)
    assert beta.shape == (6,)

    if (xr > L[:, None]).any() or (xr < 0).any():
        raise ValueError("xr is outside the room")

    if (xs > L).any() or (xs < 0).any():
        raise ValueError("xs is outside the room")

    if (N < 0).any():
        raise ValueError("N should be positive")

    Fsc = Fs/c
    L = L*(Fsc*2)  # convert dimensions to indices
    xr = xr*Fsc
    xs = xs*Fsc
    Rd = Rd*Fsc

    h = np.zeros(shape=(Nt, M))            # initialize output
    if (N == 0).all():
        N = np.floor(Nt/L)+1  # compute full order

    N = N.astype(np.int32)
    mic_types = mic_types.astype(np.uint8)

    if (seed == 0):
        seed = np.random.randint(0, 9223372036854775807, dtype=np.uint64)

    rimrs.rim(h, xs, xr, L, beta, N, Rd, Nt, Tw, Fc, seed, mic_dirs, mic_types)
    h *= Fsc

    return h, seed
