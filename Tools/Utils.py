"""
=========================================================
Class for spectrum-related calculation and evaluation
=========================================================

This class includes functions for sers spectra baseline subtraction, 
smoothing or noise reduction, normalization and quality estimation as
well as visualizations.
"""

from matplotlib import ticker, patches
import numpy as np


class Utils:

    def norm_signal(self, xx):
        """
        Normalizing a single spectrum
        Parameters
        ----------
        xx: a single spectrum represented by 1D numpy array (since the 1st dimension-
        Raman shifts of all spectra are the same therefore this dimension is omitted)

        Returns
        -------
        Normalized spectrum
        """
        return (xx - min(xx)) / (max(xx) - min(xx))

    def normalize(self, X, feature_range=(0, 1)):
        """
        Batch normalization of spectra
        Parameters
        ----------
        X: Matrix with each row as a spectrum represented by 1D numpy array
        feature_range: compression value range

        Returns
        -------
        Normalized spectra collection
        """
        if X.ndim != 2:
            raise ValueError('X must be 2D matrix with instances X features!')
        X_std = (X - np.min(X, axis=1, keepdims=True)) / \
                (np.max(X, axis=1, keepdims=True) - np.min(X, axis=1, keepdims=True))
        X_normed = X_std * (max(feature_range) - min(feature_range)) + min(feature_range)
        return X_normed


    def baseline_als(self, x, lam=1e4, p=0.005, niter=10):
        """
        This function computes the estimated baseline alignments of spectra.

        Parameters
        ----------
        x: single spectrum represented by 1D numpy array
        lam: aligning factor, must be 1e4 currently
        p: aligning factor, must be 0.005 currently
        niter: iteration rounds of fitting, using 10 currently

        Returns
        -------
        Estimated baseline of a single spectrum represented by 1D numpy array.
        """

        L = len(x)
        D = csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        baseline = 0
        for i in range(niter):
            W = spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            baseline = linalg.spsolve(Z, w * x)
            w = p * (x > baseline) + (1 - p) * (x < baseline)
        return baseline

    def baseline_sub(self, X, **kwargs):
        """
        Batch baseline subtraction of spectra
        Parameters
        ----------
        X: Matrix with each row as a spectrum represented by 1D numpy array

        Returns
        -------
        Baseline subtracted spectra collection
        """
        if X.ndim != 2:
            raise ValueError('X must be 2D matrix with instances X features!')
        X_baseline = np.zeros(shape=X.shape)
        init = 0
        for xx in X:
            xx_baseline = self.baseline_als(xx, **kwargs)
            X_baseline[init, :] = xx_baseline
            init += 1
        return X - X_baseline

    def smoothing(self, X, window_length=7, polyorder=1):
        """
        Batch smoothing of spectra collection or noise removal of spectra
        collection based on Savitzky-Golay filtering
        Parameters
        ----------
        X: Matrix with each row as a spectrum represented by 1D numpy array
        window_length: number of neighboring points used for local fitting
        polyorder: order of polynomial function for the fitting.

        Returns
        -------
        Smoothed spectra collection
        """
        if X.ndim != 2:
            raise ValueError('X must be 2D matrix with instances X features!')
        return savgol_filter(X, window_length, polyorder, axis=1)

    def check_single_spike(self, xx):
        """
        Find cosmic ray of a single spectrum (not stable and efficient
        currently, need further improvement)

        Parameters
        ----------
        xx: a single spectrum represented by 1D numpy array

        Returns
        -------
        Judgement of whether a single spectrum contains cosmic ray
        """
        xx_norm = self.norm_signal(xx)
        peaks, _ = find_peaks(xx_norm, height=0.2)
        if len(peaks) <= 1:
            '''single spike'''
            return False
        if max(peaks) - min(peaks) <= 3:
            '''2 closely-connected spikes'''
            return False
        if max(np.gradient(xx_norm)) >= 0.2:
            '''Spike hidden in signals'''
            return False
        return True

    def estimate_snr(self, xx):
        """
        Estimate the quality in terms of signal-to-noise ratio of a single 
        spectrum
        Parameters
        ----------
        xx: a single spectrum represented by 1D numpy array

        Returns
        -------
        A quantified value of the quality of a single spectrum
        """

        xx_base_sub = xx - self.baseline_als(xx, lam=1e4, p=0.005)
        xx_signal = savgol_filter(xx_base_sub, window_length=7, polyorder=1)
        
        # The maximal value of the spectrum as the 'S' value
        S = max(xx_signal)
        
        # The average of noise extracted by savgol filter as the 'N' value
        N = np.mean(abs(xx_base_sub - xx_signal))
        return S / N, xx_signal
    
    
class Visulization:

    def __init__(self):
        self.cmap = ['b', 'orange', 'g', 'r', 'purple', 'peru', 'gold', 'skyblue', 'brown', 'salmon', 'darkred',
                     'orangered', 'sienna', 'wheat']
        # self.cmap = cm.get_cmap('tab10')

    def __get_cov_ellipse(self, cov, centre, nstd, **kwargs):
        # Return a matplotlib Ellipse patch representing the covariance matrix
        # cov centred at centre and scaled by the factor nstd.

        # Find and sort eigenvalues and eigenvectors into descending order
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # The anti-clockwise angle to rotate our ellipse by
        vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
        theta = np.arctan2(vy, vx)

        # Width and height of ellipse to draw
        width, height = 2 * nstd * np.sqrt(eigvals)

        return patches.Ellipse(xy=centre, width=width, height=height, angle=np.degrees(theta), **kwargs)

    def __split_group(self, label):
        for num in list(set(label)):
            index = [i for i, x in enumerate(label) if x == num]
            yield [index, num]

    def scatter_plot(self, ax, x, y, label, marker='o', conf_eval=False):
        for index in self.__split_group(label):
            if index[1] != -1:
                xx, yy = x[index[0]], y[index[0]]
                ax.scatter(xx, yy, c=self.cmap[index[1] % len(self.cmap)], marker=marker, label=index[1], s=10, lw=0)
                if conf_eval:
                    conf_ellipse = self.__get_cov_ellipse(np.cov(xx, yy), (np.mean(xx), np.mean(yy)), nstd=2.0,
                                                        ec=self.cmap[index[1]], fill=False, alpha=0.6)
                    ax.add_artist(conf_ellipse)
            else:
                xx, yy = x[index[0]], y[index[0]]
                ax.scatter(xx, yy, c='gray', label=index[1], s=8, alpha=0.4, lw=0)
