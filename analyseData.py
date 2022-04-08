import numpy as np
import astropy.io.fits as astfit
import astropy.io.ascii as astasc
import astropy.table as asttab
import astropy.modeling as astmod
import astropy.units as astun
import astropy.convolution as astcov
import matplotlib.gridspec as gridspec
import sys, os
import useful_functions as uf
from scipy.special import gammainc, erfinv, gammaincc
from scipy.interpolate import splrep, sproot, splev
from scipy.stats import chisquare
import scipy.optimize as opt 
import timeit
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

msun = astun.Mpc*astun.Mpc*astun.Jy*astun.km/astun.s

"""
Returns the square identity matrix of given size.

    Parameters
    ----------
    n : int
        Size of the returned identity matrix.
    dtype : data-type, optional
        Data-type of the output. Defaults to ``float``.

    Returns
    -------
    out : matrix
        `n` x `n` matrix with its main diagonal set to one,
        and all other elements zero.

    See Also
    --------
    numpy.identity : Equivalent array function.
    matlib.eye : More general matrix identity function.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.identity(3, dtype=int)
    matrix([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])
"""


class anaData(object):
	"""

	"""

	def __init__(self):
		self.nobj = None 
		self.spectralaxis = None 
		self.spec = None
		self.refspec = None
		self.noise = None
		self.fitweights = None
		self.rbspec = None
		self.rbspectralaxis = None
		self.rbnoise = None
		self.rbrefspec = None
		self.noiselevel = None
		self.rbfitweights = None
		self.averms1 = 0.
		self.averms2 = 0.		
		self.stackrms = None # other.stackrms
		self.weightsum = None # other.weightsum
		self.outloc = ''
		self.datafilename = ''
		self.massfilename = ''
		self.fitparam = [None, None, None, None, None, None, None]#['','','','','','','']
		self.fitparamuncert = [None, None, None, None, None, None, None]
		self.paramtable = None
		self.intflux = np.zeros(8)
		self.tflux = None
		self.fitrms = np.zeros((8,2))
		self.averagenoise = 0.
		self.rebinstatus = False
		self.specuncert = 0.
		self.functions = ['Single Gaussian', 'Double Gaussian', 'Lorentzian Distribution Function', 'Voigt Profile', '3rd Order Gauss-Hermite Polynomial', 'Busy Function (with double horn)', 'Busy Function (without double horn)','Flux Density of Stacked Spectrum']
		self.fitcolors = {'Single Gaussian':'blue', 'Double Gaussian':'cornflowerblue', 'Lorentzian Distribution Function':'limegreen', 'Voigt Profile':'darkorange', '3rd Order Gauss-Hermite Polynomial':'saddlebrown', 'Busy Function (without double horn)':'darkviolet', 'Busy Function (with double horn)':'deeppink'}
		self.runno = None
		self.fitw50 = ['','','','','','','']
		


	def __fillInitial(self, other, cat, runno):
		if runno == 0:
			logger.info('Initialising the analysis.')
		else: pass
		self.spec = (other.spec.value/other.weightsum)
		self.refspec = other.refspec.value/other.weightsum


		self.nobj = int(other.nobj)
		self.spectralaxis = other.spectralaxis.value
		self.noise = other.stackrms[-1]
		self.averagenoise = self.__averagenoise(other.stackrms)
		self.stackrms = other.stackrms
		self.weightsum = other.weightsum
		self.runno = runno
		return

	def __averagenoise(self, noiselist):
		x = (range(1, len(noiselist)+1, 1))
		y = noiselist 
		try:
			m, c = opt.curve_fit(uf.strlinelog, x, y)[0]
		except:
			m = 0
			c = noiselist[0]
		return m




## methods associated with fitting or determining the statistics
	def fit_one_gauss(self, flux, axis, cat, weights, uncert=True):
		""" The method provides the single gaussian fit used to quantify the significance of the stack.
			Inputs:
				- flux: the flux array
				- axis: the frequency/velocity array
				- weights: if specified, these are the weights used for the fitting
			Returns:
				This method returns a flux array corresponding to the fitted Single Gaussian."""
		kwargs = {'max_nfev': int(1e6)}
		if cat.linetype == 'emission':
			high = 2*np.nanmax((flux))
			low = 0.
			amp = np.nanmax((flux))
		else:
			low = 2*np.nanmin(flux)
			amp = np.nanmin(flux)
			high = 0.
		p0 = [amp, cat.mean, cat.restdv.value]
		# print(axis, flux, p0, weights, uncert)
		popt, pcov = opt.curve_fit( uf.singleGaussian, axis, flux, p0=p0, sigma=weights, absolute_sigma=uncert, bounds=([low, cat.mean-cat.restdv.value, cat.restdv.value/2.], [high, cat.mean+cat.restdv.value, cat.maxgalw.value]), **kwargs )

		sgauss = uf.singleGaussian(axis, popt[0], popt[1], popt[2])
		cat.w50 = round(popt[2],0)
		# cat.w50 = self.calcW50( sgauss(axis), axis )
		if self.runno == 1:
			logger.info('Approximate FWHM of Gaussian fit to stack: %.1f km/s'%cat.w50)
		else: pass
		return sgauss, cat


	def fit_single_gaussian(self, cat, spec, axis, weights, uncert=False, param=None, process='fit'):
		""" This method fits a Single Gaussian to the spectrum. The fitted Single Gaussian is plotted on a set of axes, using the fitted function the integrated flux is calculated, a goodness of fit parameter (RMS) is also calculated. The fit parameters are saved to the fitparam array that is written to file later. 
			Inputs:
				- flux: the flux array to fitted 
				- axis: the frequency/velocity array
				- weights: the weights used for fitting and quantifying the goodness of fit
				- param: if using plot mode - function needs the parameters
				- process: 
					- fit: fits the function to the data
					- plot: fit must have been previously run, creates an array of the fitted function 
			Returns:
				This method does not actively return anything, it saves the following to various arrays that are then written to file later.
					- Goodness of fit parameter (fitrms)
					- Integrated flux (intflux)
					- Best fit parameters, these parameters are needed to replot the fitted function (fitparam)
		"""
		if process == 'fit':
			kwargs = {'max_nfev': int(1e6)}
			if cat.linetype == 'emission':
				high = 2*np.nanmax(spec)
				low = 0.
				amp = np.nanmax(spec)
			else:
				low = 2*np.nanmin(spec)
				amp = np.nanmin(spec)
				high = 0.
			p0 = [amp, cat.mean, cat.restdv.value]
			popt, pcov = opt.curve_fit( uf.singleGaussian, axis, spec, p0=p0, sigma=weights, absolute_sigma=uncert, bounds=([low, cat.mean-cat.restdv.value, cat.restdv.value/2.], [high, cat.mean+cat.restdv.value, cat.maxgalw.value]), check_finite=False, **kwargs )
			sgauss = uf.singleGaussian(axis, popt[0], popt[1], popt[2])
			fitparam = astfit.Column( name='Single Gaussian', format='D', array=np.array(popt) )
			fitparamuncert = astfit.Column( name='Single Gaussian Uncert', format='D', array=np.sqrt(np.diag(pcov)) )
			intflux = np.sum(sgauss(axis))
			fitrms = [self.calcFitRMS(spec, sgauss(axis), weights), self.calChiSquare(spec, sgauss(axis), weights)]
			w50 = self.calcW50( sgauss(axis), axis )
			return fitparam, fitparamuncert, abs(intflux), fitrms, w50
		elif process == 'plot':
			sgauss = uf.singleGaussian(axis, param[0], param[1], param[2])
			return sgauss


	def fit_double_gaussian(self, cat, spec, axis, weights, uncert=True, param=None, process='fit'):
		""" This method fits a Double Gaussian to the spectrum. The fitted Double Gaussian is plotted on a set of axes, using the fitted function the integrated flux is calculated, a goodness of fit parameter (RMS) is also calculated. The fit parameters are saved to the fitparam array that is written to file later. 
			Inputs:
				- flux: the flux array to fitted 
				- axis: the frequency/velocity array
				- weights: the weights used for fitting and quantifying the goodness of fit
			Returns:
				This method does not actively return anything, it saves the following to various arrays that are then written to file later.
					- Goodness of fit parameter (fitrms)
					- Integrated flux (intflux)
					- Best fit parameters, these parameters are needed to replot the fitted function (fitparam)
		"""
		if process == 'fit':
			kwargs = {'max_nfev': int(1e6)}
			if cat.linetype == 'emission':
				high = 2*np.nanmax(spec)
				low = 0.
				amp = np.nanmax(spec)
			else:
				low = 2*np.nanmin(spec)
				amp = np.nanmin(spec)
				high = 0.
			p0 = [0.7*amp, cat.mean+cat.restdv.value, cat.restdv.value, 0.7*amp, cat.mean-cat.restdv.value, cat.restdv.value]
			popt, pcov = opt.curve_fit( uf.doubleGaussian, axis, spec, p0=p0, sigma=weights, absolute_sigma=uncert, bounds=( [low, cat.mean-cat.maxgalw.value/10., cat.restdv.value/2., low, cat.mean-cat.maxgalw.value/5., cat.restdv.value/2. ], [high, cat.mean+cat.maxgalw.value/5., cat.maxgalw.value, high, cat.mean+cat.maxgalw.value/10., cat.maxgalw.value] ), **kwargs )
			dgauss = uf.doubleGaussian(popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])

			fitparam = astfit.Column( name='Double Gaussian', format='D', array=popt )
			fitparamuncert = astfit.Column( name='Double Gaussian Uncert', format='D', array=np.sqrt(np.diag(pcov)) )
			intflux = np.sum(dgauss(axis))
			fitrms = [self.calcFitRMS(spec, dgauss(axis), weights),self.calChiSquare(data=spec, model=dgauss(axis), sigma=weights, null=None)]
			w50 = self.calcW50( dgauss(axis), axis )
			return fitparam, fitparamuncert, abs(intflux), fitrms, w50
		elif process == 'plot':
			dgauss = uf.DoubleGaussian(param[0], param[1], param[2], param[3], param[4], param[5])
			return dgauss(axis)


	def fit_lorentz(self, cat, spec, axis, weights, uncert=True, param=None, process='fit'):
		""" This method fits a 1D Lorentzian model to the spectrum. The fitted 1D Lorentzian model is plotted on a set of axes, using the fitted function the integrated flux is calculated, a goodness of fit parameter (RMS) is also calculated. The fit parameters are saved to the fitparam array that is written to file later. 
			Inputs:
				- flux: the flux array to fitted 
				- axis: the frequency/velocity array
				- weights: the weights used for fitting and quantifying the goodness of fit
			Returns:
				This method does not actively return anything, it saves the following to various arrays that are then written to file later.
					- Goodness of fit parameter (fitrms)
					- Integrated flux (intflux)
					- Best fit parameters, these parameters are needed to replot the fitted function (fitparam)
		"""
		if process == 'fit':
			kwargs = {'max_nfev': int(1e6)}
			if cat.linetype == 'emission':
				high = np.inf
				low = 0.
				amp = np.nanmax(spec)
			else:
				low = -np.inf
				amp = np.nanmin(spec)
				high = 0.
			p0 = [amp, cat.mean, cat.restdv.value]
			popt, pcov = opt.curve_fit( uf.lorentzian, axis, spec, p0=p0, sigma=weights, absolute_sigma=uncert, bounds=( [low, cat.mean-cat.maxgalw.value/2., cat.mean+cat.restdv.value/2.], [high, cat.maxgalw.value/2., np.inf] ), **kwargs )
			lorentz = uf.lorentzian(axis, popt[0], popt[1], popt[2])

			fitparam = astfit.Column( name='Lorentzian Distribution Function', format='D', array=popt ) 
			fitparamuncert = astfit.Column( name='Lorentzian Distribution Function Uncert', format='D', array=np.sqrt(np.diag(pcov)) )
			intflux = np.sum(lorentz)
			fitrms = [self.calcFitRMS(spec, lorentz, weights),self.calChiSquare(data=spec,model=lorentz, sigma=weights, null=None)]
			w50 = self.calcW50( lorentz, axis )
			return fitparam, fitparamuncert, abs(intflux), fitrms, w50
		elif process == 'plot':
			lorentz = uf.lorentzian(axis, param[0], param[1], param[2])
			return lorentz


	def fit_voigt(self, cat, spec, axis, weights, uncert=True, param=None, process='fit'):
		""" This method fits a Voigt profile to the spectrum. The fitted Voigt profile is plotted on a set of axes, using the fitted function the integrated flux is calculated, a goodness of fit parameter (RMS) is also calculated. The fit parameters are saved to the fitparam array that is written to file later. 
			Inputs:
				- flux: the flux array to fitted 
				- axis: the frequency/velocity array
				- weights: the weights used for fitting and quantifying the goodness of fit
			Returns:
				This method does not actively return anything, it saves the following to various arrays that are then written to file later.
					- Goodness of fit parameter (fitrms)
					- Integrated flux (intflux)
					- Best fit parameters, these parameters are needed to replot the fitted function (fitparam)
		"""
		if process == 'fit':
			kwargs = {'max_nfev': int(1E6)}
			if cat.linetype == 'emission':
				high = np.inf
				low = 0.
				amp = np.nanmax(spec)
			else:
				low = -np.inf
				amp = np.nanmin(spec)
				high = 0.
			p0 = [amp, cat.mean, cat.restdv.value, cat.restdv.value]
			popt, pcov = opt.curve_fit( uf.VoigtModel, axis, spec, p0=p0, sigma=weights, absolute_sigma=uncert, bounds=( [low, cat.mean-cat.maxgalw.value/2., 0., 0.], [high, cat.mean+cat.maxgalw.value/2., np.inf, np.inf] ), **kwargs )

			voigt = uf.VoigtModel(axis, popt[0], popt[1], popt[2], popt[3])
			fitparam = astfit.Column( name='Voigt Profile', format='D', array=popt ) 
			fitparamuncert = astfit.Column( name='Voigt Profile Uncert', format='D', array=np.sqrt(np.diag(pcov)) )
			intflux = np.sum(voigt)
			fitrms = [self.calcFitRMS(spec, voigt, weights),self.calChiSquare(data=spec, model=voigt, sigma=weights, null=None)]
			w50 = self.calcW50( voigt, axis )
			return fitparam, fitparamuncert, abs(intflux), fitrms, w50
		elif process == 'plot':
			voigt = uf.VoigtModel(axis, param[0], param[1], param[2], param[3])
			return voigt

	
	def fit_gausshermite3(self, cat, spec, axis, weights, uncert=True, param=None, process='fit'):
		""" This method fits a 3rd order Gauss-Hermite polynomial to the spectrum. The fitted 3rd order Gauss-Hermite polynomial is plotted on a set of axes, using the fitted function the integrated flux is calculated, a goodness of fit parameter (RMS) is also calculated. The fit parameters are saved to the fitparam array that is written to file later. 
			Inputs:
				- flux: the flux array to fitted 
				- axis: the frequency/velocity array
				- weights: the weights used for fitting and quantifying the goodness of fit
			Returns:
				This method does not actively return anything, it saves the following to various arrays that are then written to file later.
					- Goodness of fit parameter (fitrms)
					- Integrated flux (intflux)
					- Best fit parameters, these parameters are needed to replot the fitted function (fitparam)
		"""
		if process == 'fit':
			kwargs = {'max_nfev': int(1E6)}
			if cat.linetype == 'emission':
				high = 2*np.nanmax(spec)
				low = 0.
				amp = np.nanmax(spec)
			else:
				low = 2*np.nanmin(spec)
				amp = np.nanmin(spec)
				high = 0.
			p0 = [amp, cat.mean, cat.restdv.value, 1e-10]
			popt, pcov = opt.curve_fit( uf.gausshermite3, axis, spec, p0=p0, sigma=weights, absolute_sigma=uncert, bounds=([low, cat.mean-cat.maxgalw.value/2., cat.restdv.value/2., -np.inf],[high, cat.mean+cat.maxgalw.value/2., cat.maxgalw.value, np.inf]), **kwargs )

			gauss3 = uf.gausshermite3(axis, popt[0], popt[1], popt[2], popt[3])
			fitparam = astfit.Column( name='3rd Order Gauss-Hermite Polynomial', format='D', array=popt )  
			fitparamuncert = astfit.Column( name='3rd Order Gauss-Hermite Polynomial Uncert', format='D', array=np.sqrt(np.diag(pcov)) )
			intflux = np.sum(gauss3)
			fitrms = [self.calcFitRMS(spec, gauss3, weights),self.calChiSquare(data=spec, model=gauss3, sigma=weights)]
			w50 = self.calcW50( gauss3, axis )
			return fitparam, fitparamuncert, abs(intflux), fitrms, w50
		elif process == 'plot':
			gauss3 = uf.gausshermite3(axis, param[0], param[1], param[2], param[3])
			return gauss3


	def fit_busyfunction_withhorn(self, cat, spec, axis, weights, uncert=True, param=None, process='fit'):
		""" This method fits a Busy function (Westermier et al, 2013) to the spectrum. The fitted Busy function is plotted on a set of axes, using the fitted function the integrated flux is calculated, a goodness of fit parameter (RMS) is also calculated. The fit parameters are saved to the fitparam array that is written to file later. 
			Inputs:
				- flux: the flux array to fitted 
				- axis: the frequency/velocity array
				- weights: the weights used for fitting and quantifying the goodness of fit
			Returns:
				This method does not actively return anything, it saves the following to various arrays that are then written to file later.
					- Goodness of fit parameter (fitrms)
					- Integrated flux (intflux)
					- Best fit parameters, these parameters are needed to replot the fitted function (fitparam)
		"""
		if process == 'fit':
			kwargs = {'max_nfev': int(1E6)}
			if cat.linetype == 'emission':
				high = 2*np.nanmax(spec)
				low = 0.
				amp = np.nanmax(spec)
			else:
				low = 2*np.nanmin(spec)
				amp = np.nanmin(spec)
				high = 0.
			p0 = [amp, 0.001, 0.001, cat.mean, cat.mean, 5e-7, cat.restdv.value]
			popt, pcov = opt.curve_fit( uf.busyfunction, axis, spec, p0=p0, sigma=weights, absolute_sigma=uncert, bounds=([low, 1e-9, 1e-9, cat.mean-cat.maxgalw.value/2., cat.mean-cat.maxgalw.value/2., 0., cat.restdv.value/2.],[high, 1., 1., cat.mean+cat.maxgalw.value/2., cat.mean+cat.maxgalw.value/2., 10., cat.maxgalw.value]), **kwargs )

			busyfunc = uf.busyfunction(axis, popt[0], popt[1], popt[2], popt[3], popt[4],popt[5], popt[6])
			fitparam = astfit.Column( name='Busy Function (with double horn)', format='D', array=popt ) 
			fitparamuncert = astfit.Column( name='Busy Function (with double horn) Uncert', format='D', array=np.sqrt(np.diag(pcov)) )
			intflux = np.sum(busyfunc)
			fitrms = [self.calcFitRMS(spec, busyfunc, weights),self.calChiSquare(data=spec, model=busyfunc, sigma=weights, null=None)]
			w50 = self.calcW50( busyfunc, axis )
			return fitparam, fitparamuncert, abs(intflux), fitrms, w50
		elif process == 'plot':
			busyfunc = uf.busyfunction(axis, param[0], param[1], param[2], param[3], param[4], param[5], param[6])
			return busyfunc


	def fit_busyfunction_withouthorn(self, cat, spec, axis, weights, uncert=True, param=None, process='fit'):
		""" This method fits a Busy function (Westermier et al, 2013) to the spectrum. The fitted Busy function is plotted on a set of axes, using the fitted function the integrated flux is calculated, a goodness of fit parameter (RMS) is also calculated. The fit parameters are saved to the fitparam array that is written to file later. 
			Inputs:
				- flux: the flux array to fitted 
				- axis: the frequency/velocity array
				- weights: the weights used for fitting and quantifying the goodness of fit
			Returns:
				This method does not actively return anything, it saves the following to various arrays that are then written to file later.
					- Goodness of fit parameter (fitrms)
					- Integrated flux (intflux)
					- Best fit parameters, these parameters are needed to replot the fitted function (fitparam)
		"""
		if process == 'fit':
			kwargs = {'max_nfev': int(1E6)}
			if cat.linetype == 'emission':
				high = 2*np.nanmax(spec)
				low = 0.
				amp = np.nanmax(spec)
			else:
				low = 2*np.nanmin(spec)
				amp = np.nanmin(spec)
				high = 0.
			p0 = [amp, 0.001, 0.001, cat.mean, cat.restdv.value]
			popt, pcov = opt.curve_fit( uf.busyfunctionmodel, axis, spec, p0=p0, sigma=weights, absolute_sigma=uncert, bounds=([low,1e-9, 1e-9, cat.mean-cat.maxgalw.value/2., cat.restdv.value/2.],[high, 1., 1., cat.mean+cat.maxgalw.value/2., cat.maxgalw.value]), **kwargs )

			busyfunc = uf.busyfunctionmodel(axis, popt[0], popt[1], popt[2], popt[3], popt[4])
			fitparam = astfit.Column( name='Busy Function (without double horn)', format='D', array=popt ) 
			fitparamuncert = astfit.Column( name='Busy Function (without double horn) Uncert', format='D', array=np.sqrt(np.diag(pcov)) )
			intflux = np.sum(busyfunc)
			fitrms = [self.calcFitRMS(spec, busyfunc, weights),self.calChiSquare(data=spec, model=busyfunc, sigma=weights, null=None)]
			w50 = self.calcW50( busyfunc, axis )
			return fitparam, fitparamuncert, abs(intflux), fitrms, w50
		elif process == 'plot':
			busyfunc = uf.busyfunctionmodel(axis, param[0], param[1], param[2], param[3], param[4])
			return busyfunc
		

	def calcFitRMS(self, data, model, weight):
		c = (data-model)/weight
		c2 = c**2
		mc2 = np.mean(c2)
		rms = np.sqrt(mc2)
		return rms


	def calChiSquare(self, data, model, sigma, null=None):
		chi2 = np.sum(((data - model)/sigma)**2)
		if null != None:
			chin2 = np.sum(((data - null)/sigma)**2)
			return chi2, chin2
		else:
			return chi2


	def calcPvalue(self, gauss, spec, sigma):
		chi_gauss, chi_str = self.calChiSquare(spec, gauss, sigma, null=0.)#self.calcChiSquare(data=spec, df=3, model=gauss, null=noise/10.)
		D = (chi_str - chi_gauss)
		pval = gammaincc(3./2.,D/2.)
		P = gammainc(3./2.,D/2.) #1-pval
		if pval < np.finfo(pval.dtype).eps:
			sig = 8.2
		else:
			sig = np.sqrt(2.) * erfinv(P)
		return pval, sig


	def calcW50(self, spec, wavel ):
		"""
		adapted from https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
		"""
		half_max = np.nanmax(spec)/2.0
		s = splrep(wavel, spec - half_max)
		roots = sproot(s)
		w50 = np.round(abs(roots[-1] - roots[0]), 0)
		return w50


	def callFunctions(self, num, cat, spec, axis, weights, uncert=False, param=None, process='fit'):
		if num == '1':
			output = self.fit_single_gaussian(cat, spec, axis, weights, uncert, param, process)
			if process == 'fit':
				self.fitparam[0], self.fitparamuncert[0], self.intflux[0], self.fitrms[0], self.fitw50[0] = output
			else:
				return output
		elif num == '2':
			output = self.fit_double_gaussian(cat, spec, axis, weights, uncert, param, process)
			if process == 'fit':
				self.fitparam[1], self.fitparamuncert[1], self.intflux[1], self.fitrms[1], self.fitw50[1] = output
			else:
				return output
		elif num == '3':
			output = self.fit_lorentz(cat, spec, axis, weights, uncert, param, process)
			if process == 'fit':
				self.fitparam[2], self.fitparamuncert[2], self.intflux[2], self.fitrms[2], self.fitw50[2] = output
			else:
				return output
		elif num == '4':
			output = self.fit_voigt(cat, spec, axis, weights, uncert, param, process)
			if process == 'fit':
				self.fitparam[3], self.fitparamuncert[3], self.intflux[3], self.fitrms[3], self.fitw50[3] = output
			else:
				return output
		elif num == '5':
			output = self.fit_gausshermite3(cat, spec, axis, weights, uncert, param, process)
			if process == 'fit':
				self.fitparam[4], self.fitparamuncert[4], self.intflux[4], self.fitrms[4], self.fitw50[4] = output
			else:
				return output
		elif num == '6':
			output = self.fit_busyfunction_withhorn(cat, spec, axis, weights, uncert, param, process)
			if process == 'fit':
				self.fitparam[5], self.fitparamuncert[5], self.intflux[5], self.fitrms[5], self.fitw50[5] = output
			else:
				return output
		elif num == '7':
			output = self.fit_busyfunction_withouthorn(cat, spec, axis, weights, uncert, param, process)
			if process == 'fit':
				self.fitparam[6], self.fitparamuncert[6], self.intflux[6], self.fitrms[6], self.fitw50[6] = output
			else:
				return output

	def __plotFunctions(self, spec, spectral, weights, cat, options):
		plt.close('all')
		fig = plt.figure(figsize=(12.5,9))
		gs = gridspec.GridSpec(6,1)
		ax1 = fig.add_subplot(gs[:4])
		ax2 = fig.add_subplot(gs[4:])

		if (astun.Jy == cat.stackunit) :
			if 1E-3 < max(spec) < 1E-1:
				conv = 1E3
				unitstr = r'Average Stacked Flux Density ($\mathrm{mJy}$)'
			elif max(spec) < 1E-3:
				conv = 1E6
				unitstr = r'Average Stacked Flux Density ($\mathrm{\mu Jy}$)'
			else:
				conv = 1.
				unitstr = 'Average Stacked Flux Density ($\mathrm{Jy}$)'
		elif (uf.msun == cat.stackunit):
			masstext = '%.3E'%np.max(spec)
			masi = masstext.split('E')[0]
			masexp = masstext.split('E')[1]
			conv = 1./eval('1E%s'%masexp)
			unitstr = r'Average Stacked Mass ($10^{%i} \, \mathrm{M_\odot/chan}$)'%float(masexp)
		elif uf.gasfrac == cat.stackunit:
			conv = 1.
			unitstr = r'Average Stacked Gas Fraction (M$_\mathrm{HI}$/M$_\ast$)'
		else:
			conv = 1.
			unitstr = 'Average Stacked Quantity'

		ax1.set_ylabel(unitstr)
		ax1.bar(spectral, spec*conv, width=cat.restdv.value, color='gray', edgecolor='gray', label='Average Stacked Spectrum')
		ax1.set_xlim(spectral[0], spectral[-1])
		ax1.set_xlabel(r'Relative Velocity (km/s)')

		x = np.linspace(spectral[0], spectral[-1], 1000)
		colnames = ['Fitted Function', 'Fit RMS', r'$\chi ^2$', r'$\chi _\mathrm{red}^2$', 'Int. Flux', r'FWHM']
		celldata = []
		for num in options:
			ind = int(eval(num)-1)
			func = self.callFunctions(num, cat, spec, spectral, weights, uncert=True, process='fit')
			fname = self.functions[ind]
			param = self.fitparam[ind].array
			w50s = self.fitw50[ind]
			intfx = self.intflux[ind]
			func = self.callFunctions(num, cat, None, x, None, uncert=False, param=param, process='plot')
			celldata.append([np.unicode(fname), self.fitrms[ind][0], self.fitrms[ind][1], (self.fitrms[ind][1])/(len(spec) - len(param) ), intfx,  self.fitw50[ind] ])
			ax1.plot(x, func*conv, color=self.fitcolors[fname], label=fname)

		tab = asttab.Table(names=colnames, dtype=('S100', 'f8', 'f8', 'f8', 'f8', 'f8'))
		tab['FWHM'].unit = astun.km/astun.s
		tab['Int. Flux'].unit = cat.stackunit
		for k in range(len(celldata)):
			tab.add_row(celldata[k])

		if cat.latex == 'latex':
			tabstr = r'''\begin{tabular}{|l|c|c|c|c|c|} \hline %s \end{tabular}'''%uf.latextablestr(tab, cat)
			ax2.annotate(tabstr, xy=(0.5,0.5), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', color='k', fontsize=12)
		else:
			newtab = uf.convertable(tab)
			thetable = ax2.table(cellText=newtab.as_array(), colLabels=newtab.colnames, loc='center', fontsize=14)
			thetable.auto_set_column_width((-1,0,1,2,3))
		
		ax1.legend(loc='upper left',fontsize=8, numpoints=1)
		ax2.axis('off')
		plt.tight_layout()
		logger.info('Created Diagnostic Plot 3 - checking the fitted functions.')
		if cat.suppress == 'hide':
			plt.savefig('%sDiagnosticPlot3_Functions_%s.pdf'%(cat.outloc, cat.runtime),bbox_inches='tight', pad_inches=0.2)
		else:
			plt.show(False)

		return


	def fit_functions(self, cat, spec, axis, weights):
		try:
			if (self.runno == 0) and (cat.config is False):
				print ('\nWhat functions would you like to fit to the data? (you can choose more than 1)\n\t1. Single Gaussian\n\t2. Double Gaussian\n\t3. Lorentzian Distribution Function\n\t4. Voigt Profile\n\t5. 3rd Order Gauss-Hermite Polynomial\n\t6. Busy Function (with double horn)\n\t7. Busy Function (without double horn)')#\n\t8. Cancel fitting
				options = input('Please enter the numbers of the functions you want fitted to the stacked spectrum. (The numbers must be separated by commas): ')
				options = tuple(options.split(','))
				self.__plotFunctions(spec, axis, weights, cat, options)
				happfunc = input('Are you happy with the chosen functions [y/n]? ')
				if happfunc == 'n':
					self.fit_functions(cat, spec, axis, weights)
				else:
					plt.close()
					happfunc = 'y'
					plt.close('all')
					cat.funcnum = options
					funcs = [''+'%s'%self.functions[int(eval(i)-1)]+', ' for i in options]
					func = ''
					for i in range(len(funcs)):
						func += funcs[i] 
					func = func[:-2]+'.'
					logger.info('Fitting %s'%func)
				plt.close()
			else: pass
		except KeyboardInterrupt:
			logger.critical('Keyboard Interrupt during function fitting.')
			uf.earlyexit(cat)
			raise sys.exit()
		except SystemExit:
			uf.earlyexit(cat)
			raise sys.exit()
		except:
			logger.error('Exception occurred trying to fit the functions, trying function fitting again.', exc_info=True)
			uf.earlyexit(cat)
			raise sys.exit()
		try:
			for n in cat.funcnum:
				self.callFunctions(n, cat, spec, axis, weights, uncert=False, param=None, process='fit')
			return cat

		except KeyboardInterrupt:
			uf.earlyexit(cat)
			raise sys.exit()
		except SystemExit:
			uf.earlyexit(cat)
			raise sys.exit()
		except:
			logger.error('Exception occurred trying to fit the functions, trying function fitting again.', exc_info=True)
			uf.earlyexit(cat)
			raise sys.exit()


	def areaabovenoise(self, ydata, noise, cat):
		if cat.linetype == 'emission':
		    y = ydata - noise
		else:
			y = (-1)*ydata - noise
		use = np.where(y < 0.)
		y[use] = 0.
		frac = np.sum(y)
		total = abs(np.sum(ydata))
		massfrac = frac/total * 100.
		return massfrac


	def __plot_basic(self, noise, sgauss, spec, refspec, spectrala, num=None, cat=None, rebin=0, smooth=None):
		plt.close('all')

		plotbasic = plt.figure(figsize=(8,6))
		basicax = plotbasic.add_subplot(111)
		basicax.set_xlim(spectrala[0],spectrala[-1])
		self.locs, locs = plt.xticks()
		self.label = ["%5.1f" % a for a in self.locs]
		basicax.axhline(0, ls='--', color='dimgray')
		basicax.axvspan(xmin=cat.mean-cat.maxgalw.value/2., xmax=cat.mean+cat.maxgalw.value/2., color='lightgreen', alpha=0.5, label='Integration Window')
		spectral = spectrala*cat.spectralunit
		plt.xticks(self.locs,self.label)
		basicax.set_xlabel('Relative Velocity (km/s)')
		if cat.clean == 'clean':
			refspec = 0.
			noise = 0.
		else:
			h = 0.
		if (cat.stackunit == astun.Jy):
			if 1E-3 < max(spec) < 1E-1:
				conv = 1E3
				ystring =  r'Average Stacked Flux Density ($\mathrm{mJy}$)'
			elif max(spec) < 1E-3:
				conv = 1E6
				ystring =  r'Average Stacked Flux Density ($\mathrm{\mu Jy}$)'
			else:
				conv = 1.
				ystring =  r'Average Stacked Flux Density ($\mathrm{Jy}$)'
		elif (cat.stackunit == uf.msun):
			masstext = '%.3E'%np.max(spec)
			masi = masstext.split('E')[0]
			masexp = masstext.split('E')[1]
			conv = 1./eval('1E%s'%masexp)
			ystring = r'Average Stacked Mass ($10^{%i} \, \mathrm{M_\odot/chan}$)'%float(masexp)
		elif cat.stackunit == uf.gasfrac:
			ystring = r'Average Stacked Gas Fraction (M$_\mathrm{HI}$/M$_\ast$)'
			conv = 1
		else:
			print ('There was a problem with basic plot')

		basicax.set_ylabel(ystring)
		if cat.clean == 'clean':
			basicax.plot(spectral.value, spec*conv, 'k-', label='Stacked Spectrum')
			basicax.plot(spectral.value, sgauss*conv, color='mediumvioletred', ls='-', label='Fitted Single Gaussian')
		else:
			basicax.axhspan(ymin=-noise*conv, ymax=noise*conv, alpha=0.5, facecolor='lightskyblue', ec='lightskyblue', label=r'1$\sigma$ noise')
			basicax.plot(spectral.value, spec*conv, 'k-', label='Stacked Spectrum')
			basicax.plot(spectral.value, refspec*conv, color='gray', ls='--', label='Reference Spectrum', lw=1)
			basicax.plot(spectral.value, sgauss*conv, color='mediumvioletred', ls='-', label='Fitted Single Gaussian')
		basicax.set_xlim(spectral.value[0], spectral.value[-1])
		basicax.legend(loc='lower center',fontsize=8, numpoints=1, ncol=3)

		if rebin > 0 and smooth is None:
			extra = 'rebin='+rebin+'_'
			num = 2
		elif rebin > 0 and smooth != None:
			extra = '%s_window=%i_'%(smooth, rebin)
			num = 2
		else:
			extra = ''
			num = 1

		if cat.suppress == 'hide':
			plt.savefig(cat.outloc+'DiagnosticPlot%i_SpectrumCheck_%s%s.pdf'%(num, extra, cat.runtime), bbox_inches='tight', pad_inches=0.2)
		else:
			plt.show(False)
		return


	def __smoothData(self, cat, spec, refspec, smoothtype, windowsize):
		windowsize += (windowsize+1)%2
		if cat.smoothalgorithms['2'] == smoothtype:
			kernel = astcov.Box1DKernel(windowsize)
		elif cat.smoothalgorithms['1'] == smoothtype:
			kernel = np.hanning(windowsize+2)
			kernel = kernel/np.sum(kernel)
		else:
			logger.warning('The chosen smoothing algorithm did not match an available algorithm')
			return spec, refspec
		sspec = astcov.convolve(spec, kernel)
		if cat.clean == 'clean':
			srefspec = None
		else:
			srefspec = astcov.convolve(refspec, kernel)
		return sspec, srefspec

	
	def __getSmoothSpectrum(self, spec, refspec, axis, cat, runno):
		try:
			if (runno == 0) and (cat.config == False):
				smoothtype = input('\nPlease choose your favourite smoothing algorithm:\n \t1. Hanning Window\n\t2. Boxcar\nDesired option: ')
				if uf.checkkeys(cat.smoothalgorithms, smoothtype.split()[0]):
					cat.smoothtype = cat.smoothalgorithms[smoothtype.split()[0]]
				else:
					print ('The smoothing algorithm you chosen is not implemented in this software.\nPlease try again.')
					self.__getSmoothSpectrum(spec, refspec, axis, cat, runno)
				cat.smoothwin = eval(input('Please enter the desired size of the smoothing window: '))
			else:
				pass
			spectrum, referencespectrum = self.__smoothData(cat, spec, refspec, cat.smoothtype, cat.smoothwin)
			noiselevel = np.std( np.array( list(spectrum[ :cat.maskstart])+list(spectrum[cat.maskend: ] ) ))
			fitweights = np.ones(len(spectrum))*noiselevel

			if (runno == 0):
				percentage, sgauss, cat = self.__callInitialStats(spectrum, referencespectrum, axis, noiselevel, cat, fitweights, rebin=cat.smoothwin, smooth=cat.smoothtype)
			else:
				sgauss, cat = self.fit_one_gauss(spectrum, axis, cat, fitweights)

		except (SystemExit, KeyboardInterrupt):
			uf.exit(cat)
		except:
			logger.error('Error trying to smooth the stacked spectrum:', exc_info=True)
			uf.earlyexit(cat)
		return spectrum, referencespectrum, axis, fitweights, noiselevel, sgauss, cat

	
	def __callSmoothing(self, cat, options=[], n='1', check=0):
		# while check == 0:
		spectrum, referencespectrum, axis, fitweights, noiselevel, sgauss, cat = self.__getSmoothSpectrum(self.spec, self.refspec, self.spectralaxis, cat, self.runno)
		if ((self.runno == 0) or (cat.rebinstatus == True) ) and (cat.config == False):
			checkYN = input('\nWould you like to keep this version [y] or rebin again [r] or abort the rebin [a]: ').lower()
			plt.close()
			if checkYN == 'y':
				check = 1
				cat.rebinstatus = True
				self.rebinstatus = True
				return spectrum, referencespectrum, axis, fitweights, noiselevel, sgauss, options, n, cat;

			elif checkYN == 'r':
				del checkYN
				cat.rebinstatus = False
				return self.__callSmoothing(cat, options)

			elif checkYN == 'a':
				check = 0
				if len(cat.optnum) == 1:
					cat.optnum = []
				else:
					cat.optnum = cat.optnum[1:]
				cat.rebinstatus = False
				return spectrum, referencespectrum, axis, fitweights, noiselevel, sgauss, options, n, cat;

			else:
				print ('Please choose "y" or "r" or "a". \n')
				return self.__callSmoothing(cat, options, n)
		elif cat.config == True:
			check = 1
			cat.rebinstatus = True
			self.rebinstatus = True
			return spectrum, referencespectrum, axis, fitweights, noiselevel, sgauss, options, n, cat;
		else:
			return spectrum, referencespectrum, axis, fitweights, noiselevel, sgauss, options, n, cat;
			
		# return spectrum, referencespectrum, axis, fitweights, noiselevel, sgauss, options, n, cat;


	def __determineRebin(self, cat, runno):
		if (runno > 0) or (len(cat.rbnum) > 0):
			rebin = cat.rbnum[0]
		else:
			rebin = eval(input('\nPlease enter the number of old bins to new bins: '))
			(cat.rbnum).append(rebin)
		try:
			rbspec, rbrefspec, rbspectral, rbweights, noise, sgauss, mask = self.__getRebinSpectrum(rebin, self.spec, self.refspec, self.spectralaxis, self.fitweights, cat, runno)
			return rbspec, rbrefspec, rbspectral, rbweights, noise, sgauss, mask
		except (KeyboardInterrupt, SystemExit):
			uf.exit(cat)
		except:
			logger.error('Something went wrong while trying to rebin the spectrum.', exc_info=True)
			uf.earlyexit(cat)


	def __getRebinSpectrum(self, rebin, spec, refspec, spectral, weights, cat, runno):
		ol = len(spec) 
		nll = int(np.floor(ol/int(rebin)))
		nl = nll * rebin
		mask = int(round(cat.mask/rebin,0))
		cs = int(ol/2)
		csn = int(nl/2)
		noiselen = cat.noiselen/int(rebin)
		censpec = spec[int(cs-csn):int(cs+csn+nl%2)]
		censpectral = spectral[int(cs-csn):int(cs+csn+nl%2)]
		cenweights = weights[int(cs-csn):int(cs+csn+nl%2)]
		rbspec = np.mean(np.reshape(censpec, (nll,rebin)), axis=1)
		rbspectral = np.mean(np.reshape(censpectral, (nll,rebin)), axis=1)
		if cat.clean == 'clean':
			rbrefspec = None
			noise = 0.
			rbweights = np.ones(len(rbspec))*rebin*np.sqrt(self.nobj)
		else:
			cenrefspec = refspec[int(cs-csn):int(cs+csn+nl%2)]
			rbrefspec = np.mean(np.reshape(cenrefspec, (nll,rebin)), axis=1)
			noise = np.std( np.array( list(rbspec[:int(noiselen/2)]) + list(rbspec[int(-noiselen/2):]) ) )
			rbweights = np.ones(len(rbspec))*noise
		try:
			if runno == 0:
				percentage, sgauss, cat = self.__callInitialStats(rbspec, rbrefspec, rbspectral, noise, cat, rbweights)
			else:
				try:
					sgauss, cat = self.fit_one_gauss(rbspec, rbspectral, cat, rbweights)
				except (KeyboardInterrupt, SystemExit):
					uf.exit(cat)
				except:
					logger.error('Something went wrong while trying to rebin the spectrum.', exc_info=True)
					uf.earlyexit(cat)
		except (KeyboardInterrupt, SystemExit):
			uf.exit(cat)
		except:
			logger.error('Something went wrong while trying to rebin the spectrum.', exc_info=True)
			uf.earlyexit(cat)

		return rbspec, rbrefspec, rbspectral, rbweights, noise, sgauss, mask


	def __callRebin(self, cat, options=[], n='3', check=0):
		while check == 0:
			rbspec, rbrefspec, rbspectral, rbweights, noise, sgauss, mask = self.__determineRebin(cat, self.runno)
			if self.runno == 0:
				checkYN = input('\nWould you like to keep this version [y] or rebin again [r] or abort the rebin [a]: ').lower()
				plt.close()
				if checkYN == 'y':
					cat.rebinstatus = True
					cat.smoothtype = 'rebin'
					check = 1
					cat.mask = mask
					return rbspec, rbrefspec, rbspectral, rbweights, noise, sgauss, options, n, cat;
				elif checkYN == 'r':
					del checkYN	
					cat.rebinstatus = False		
					return self.__callRebin(cat, options)
				elif checkYN == 'a':
					cat.rebinstatus = False
					check = 1
					return rbspec, rbrefspec, rbspectral, rbweights, noise, sgauss, options, n, cat;
				else:
					print ('Please choose "y" or "r" or "a". \n')
					return self.__callRebin(cat, options, n)
			else:
				return rbspec, rbrefspec, rbspectral, rbweights, noise, sgauss, options, n, cat


	def __next_move(self, options, sgauss, cat):
		plt.close()
		try:
			for k in range(len(options)):
				n = options[k]
				if n == '1':
					self.rbspec, self.rbrefspec, self.rbspectralaxis, self.rbfitweights, self.rbnoise, sgauss, options, n, cat = self.__callSmoothing(cat, options, n)
					if len(options) == 1:
						self.__getOptions(sgauss, cat)
				elif n == '2':
					if cat.rebinstatus == True or self.rebinstatus == True:
						cat = self.fit_functions(cat, self.rbspec, self.rbspectralaxis, self.rbfitweights)
					else:
						cat = self.fit_functions(cat, self.spec, self.spectralaxis, self.fitweights)	
				elif n == '3':
					self.rbspec, self.rbrefspec, self.rbspectralaxis, self.rbfitweights, self.rbnoise, sgauss, options, n, cat = self.__callRebin(cat, options, n)
					if len(options) == 1:
						self.__getOptions(sgauss, cat)
				elif n == '5':
					print ('\nExiting. Thank you for your time.\n')
					logger.info('Selected option 5: Exit with out saving any progress.')
					uf.exit(cat)
				else:
					h = 0
			return cat
		except (SystemExit, KeyboardInterrupt):
			uf.exit(cat)
		except:
			logger.error('Exception occurred:', exc_info=True)
			uf.earlyexit(cat)
		return cat

		
	def __getOptions(self, sgauss, cat):
		if ((self.runno == 0) and (cat.config == False)):
			print ("\nYou now have a couple of options:\n\t1. Smooth the spectrum.\n\t2. Fit a number of other functions to the spectrum.\n\t3. Rebin the spectrum.\n\t4. Don't fit anything to the spectrum, but continue with the analysis.\n\t5. Exit this programme without doing anything else. [Will not save anything]")
			options = input('Please enter the numbers of the option of your next step. (The numbers must be separated by commas): ')
			option = options.strip().split(',')
			option.sort()
			[cat.optnum.append(i) for i in option]
			try:
				if any(eval(i) > 5. for i in option):
					return self.__getOptions(sgauss, cat)
				elif option[0] == '':
					return self.__getOptions(sgauss, cat)
				else:
					if len(option) > 1 and '5' in option:
						option.remove('5')
						cat = self.__next_move(option, sgauss, cat)
					else:
						cat = self.__next_move(option, sgauss, cat)
					return cat
			except (SystemExit, KeyboardInterrupt):
				uf.exit(cat)
			except:
				logger.error('Exception occurred:', exc_info=True)
				uf.earlyexit(cat)
		else:
			cat = self.__next_move(cat.optnum, sgauss, cat)
			return cat
		


	def calcSignalNoise(self, flux, axis, sgauss, cat, noise):
		cen = int(len(flux)/2)
		si = int(cen-cat.mask/2)
		se = int(cen+1+cat.mask/2)
		peak = np.nanmax(flux[si:se])

		if cat.clean == 'clean':
			integrated_snr = 0
			alfalfa_snr = 0
			standard_snr = 0
		else:
			standard_snr = abs(peak/noise)
			integrated_snr = (np.sum(flux[si:se]*(abs(axis[0]-axis[1]))))/( noise * abs(axis[0]-axis[1]) * np.sqrt(cat.mask) )
			w50loc = [np.argmin(abs(flux[si:cen] - 0.5*peak)), np.argmin(abs(flux[cen:se] - 0.5*peak))]
			# w50 = abs(axis[int(w50loc[0]+si)]) + abs(axis[int(w50loc[1]+cen)])
			alfalfa_snr = abs(( (np.sum(flux[si:se]*(abs(axis[0]-axis[1])))/cat.w50) * (cat.w50/(2*cat.restdv.to('km/s').value))**0.5 )/noise)
			# cat.w50 = np.round(w50,0)
		return standard_snr, alfalfa_snr, integrated_snr, cat	


	def __callInitialStats(self, flux, refflux, axis, noise, cat, weights=None, smooth=None, rebin=0):
		try:
			sgauss, cat = self.fit_one_gauss(flux, axis, cat, weights)
		except (KeyboardInterrupt, SystemExit):
			uf.exit(cat)
		except: 
			logger.error("Couldn't fit the first Gaussian.", exc_info=True)
			uf.earlyexit(cat)
		try:
			percentage = self.areaabovenoise(sgauss, noise, cat)
		except (KeyboardInterrupt, SystemExit):
			uf.exit(cat)
		except: 
			logger.error("Failed at calculating percentage of spectrum above the noise", exc_info=True)
			uf.earlyexit(cat)
		try:
			p, sig = self.calcPvalue(sgauss, flux, weights)
			snr, sna, isn, cat = self.calcSignalNoise(flux, axis, sgauss, cat, noise)
		except (KeyboardInterrupt, SystemExit):
			uf.exit(cat)
		except: 
			logger.error("Failed determining the spectrum statistics.", exc_info=True)
			uf.earlyexit(cat)

		if sig == 8.2:
			ex = '>'
			pval = '%.2E'%p
		else:
			ex = ''
			pval = '%.3g'%p

		if smooth != None:
			begin = 'Statistics (Smoothing alogrithm: %s, Smoothing window: %i channels):\n'%(cat.smoothtype, cat.smoothwin)
		elif rebin > 0:
			begin = 'Statistics (rebin: %i channels):\n'%cat.rbnum[0]
		else:
			begin = 'Statistics of original stacked spectrum:\n'
		
		info = 'Stacked %i profiles.'%self.nobj
		stats1 = 'The peak signal-to-noise ratio is %.2g'%snr
		stats0 = 'The integrated signal-to-noise ratio is %.2f'%isn
		stats2 = 'The ALFALFA signal-to-noise ratio is %.2g'%sna
		stats3 = "The single Gaussian fitted to the stacked spectrum has %s%.2g sigma significance (p-value = %s)"%(ex,sig,pval)
		stats4 = "%.1f %% of the stacked spectrum is above 1 sigma noise."%percentage
		stats5 = "The approximate full width half maximum of spectrum is %.1f km/s"%cat.w50

		line1 = '  |--'+'-'*105+'|\n'
		line2 = '   |  '+' '*105+'|\n'
		line3 = '   |  '+info+' '*(105-len(info))+'|\n'
		line4 = '   |  '+stats5+' '*(105-len(stats5))+'|\n'
		line5 = '   |  '+stats1+' '*(105-len(stats1))+'|\n'
		line6 = '   |  '+stats0+' '*(105-len(stats0))+'|\n'
		line7 = '   |  '+stats2+' '*(105-len(stats2))+'|\n'
		line8 = '   |  '+stats3+' '*(105-len(stats3))+'|\n'
		line9 = '   |  '+stats4+' '*(105-len(stats4))+'|\n'
		line10 = '   |  '+' '*105+'|\n'
		line11 = '   |--'+'-'*105+'|\n'
		displaytext = line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10 + line11
		print ('\n', displaytext)
		logger.info(begin+displaytext)


		try:
			self.__plot_basic(noise, sgauss, flux, refflux, axis, cat=cat, rebin=rebin, smooth=smooth)
		except (KeyboardInterrupt, SystemExit):
			uf.exit(cat)
		except: 
			logger.error("Failed plotting the diagnostic plot.", exc_info=True)
			uf.earlyexit(cat)
		return percentage, sgauss, cat


	def __callAnalysis(self, cat):
		if cat.clean == 'clean':
			self.stackrms = np.zeros(self.nobj)
			self.fitweights = np.ones(len(self.spec))*np.sqrt(self.nobj)
			self.noiselevel = 0.
		else:
			try:
				sgauss, cat = self.fit_one_gauss(self.spec, self.spectralaxis, cat, self.fitweights)
			except:
				logger.critical('Failed at first gaussian fit', exc_info=True)
				uf.earlyexit(cat)
			self.noiselevel = (np.percentile( np.array( list(self.spec[:cat.maskstart])+list(self.spec[cat.maskend:] ) ), 75) - np.percentile( np.array( list(self.spec[:cat.maskstart])+list(self.spec[cat.maskend:] ) ), 25))/(2*0.67) ## use the IQR to determine noise -- the factor 1/0.67 converts this to the standard deviation, the factor 2 is because the IQR is essentially the measure from -1sig to 1sig
			self.fitweights = np.ones(len(self.spec))*self.noiselevel
			x = (np.arange(1,self.nobj+1,1))
			y = (np.array(self.stackrms))
			try:
				params, cov = opt.curve_fit(uf.strlinelog, x, y, p0=[y[0], 0.], absolute_sigma=False)
			except TypeError:
				logger.error('There is a problem with fitting the noise.')
				params = [0,0]
			self.averms1 = params[0]
			self.averms2 = params[1]
		if self.runno > 0 or cat.clean == 'clean':
			percentage = 21
			sgauss = None
		else:

			percentage, sgauss, cat = self.__callInitialStats(self.spec, self.refspec, self.spectralaxis, self.noiselevel, cat, self.fitweights)
			widhappy = 'n'
			if cat.config == False:
				changewid = input('\nWould you like to change the width of the Integration Window (y/n): ')
				if changewid.lower() == 'y':
					while widhappy != 'y':
						cat.maxgalw = eval(input('Please enter the new Integration Width in %s: '%(cat.restdv.unit.to_string()))) * cat.restdv.unit
						cat.mask = (cat.maxgalw/cat.restdv).value
						cat.maskstart = np.argmin(abs(self.spectralaxis - cat.mean + cat.maxgalw.value/2.))
						cat.maskend = np.argmin(abs(self.spectralaxis - cat.mean - cat.maxgalw.value/2.)) + 1
						percentage, sgauss, cat = self.__callInitialStats(self.spec, self.refspec, self.spectralaxis, self.noiselevel, cat, self.fitweights)
						widhappy = input('Are you happy with the new width [y/n]? ').lower()
					logger.info('Changed Integration Window to %i km/s'%cat.maxgalw.value)
				else:
					pass
			else:
				pass
		try:
			cat = self.__getOptions(sgauss, cat)
		except (SystemExit, KeyboardInterrupt):
			uf.exit(cat)
		except:
			print ('\nPlease select at least one of the options below.')
			cat = self.__getOptions(sgauss, cat)
			cat.rebinstatus = self.rebinstatus
		return cat
		
			
	def saveResults(self, cat, uncert):
		if '4' in cat.optnum or '4.' in cat.optnum:
			names = [self.functions[7]]
			if uncert == False:
				col2 = np.array([self.intflux[7]])
				col5 = np.array([cat.w50])
				if (cat.stackunit == astun.Jy):
					col2 = col2*cat.restdv*cat.stackunit*astun.km/astun.s
				# elif cat.cluster == True:
				# 	col2 = col2*uf.msun
				else:
					col2 = col2*cat.stackunit
				data = asttab.Table([names, col2, col5], names=['Fitted Function','Integrated Flux', 'FWHM'])
				data['Integrated Flux'].unit = col2.unit
				data['FWHM'].unit = astun.km/astun.s
			else:
				col1 = np.array([self.intflux[:,0][7]])
				col2 = np.array([self.intflux[:,1][7]])
				col5 = np.array([cat.w50])
				if (cat.stackunit == astun.Jy):
					col1 = col1*cat.restdv*cat.stackunit
					col2 = col2*cat.restdv*cat.stackunit
				# elif cat.cluster == True:
				# 	col1 = col1*uf.msun
				# 	col2 = col2*uf.msun
				else:
					col1 = col1*cat.stackunit
					col2 = col2*cat.stackunit
				data = asttab.Table([names, col1, col2, col5], names=['Fitted Function','Integrated Flux', 'Uncertainty Integrated Flux', 'FWHM'])
				data['Integrated Flux'].unit = col1.unit
				data['Uncertainty Integrated Flux'].unit = col2.unit
				data['FWHM'].unit = astun.km/astun.s


			astasc.write(data, cat.outloc+'IntegratedFlux_'+cat.runtime+'.csv', format='ecsv')	
			logger.info('Saved Integrated Flux to disk.')

		else:
			ind = np.array([int(eval(i)-1) for i in cat.funcnum], dtype=np.int_)
			ind = np.append(ind, 7)
			self.fitw50 = np.append(self.fitw50, cat.w50)
			if len(ind) == 0:
				h = 0.
			else:
				names = [self.functions[int(i)] for i in ind]
				col3 = self.fitrms[:,0][ind]
				col4 = self.fitrms[:,1][ind]
				col5 = [self.fitw50[int(i)] for i in ind]
				if uncert == False:
					col2 = self.intflux[ind]
					if (cat.stackunit == astun.Jy):
						col2 = col2*cat.restdv*cat.stackunit
					elif cat.cluster == True:
						col2 = col2*uf.msun
					else:
						col2 = col2*cat.stackunit
					data = asttab.Table([names, col2, col3, col4, col5], names=['Fitted Function','Integrated Flux','Fit RMS', 'Fit ChiSquare', 'FWHM'])	
					data['Integrated Flux'].unit = col2.unit
					data['FWHM'].unit = astun.km/astun.s
				else:
					col1 = self.intflux[:,0][ind]
					col2 = self.intflux[:,1][ind]
					if (cat.stackunit == astun.Jy):
						col1 = col1*cat.restdv*cat.stackunit
						col2 = col2*cat.restdv*cat.stackunit
					# elif cat.cluster == True:
					# 	col1 = col1*uf.msun
					# 	col2 = col2*uf.msun
					else:
						col1 = col1*cat.stackunit
						col2 = col2*cat.stackunit

					data = asttab.Table([names, col1, col2, col3, col4, col5], names=['Fitted Function','Integrated Flux', 'Uncertainty Integrated Flux','Fit RMS', 'Fit ChiSquare', 'FWHM'])
					data['Integrated Flux'].unit = col1.unit
					data['Uncertainty Integrated Flux'].unit = col2.unit
					data['FWHM'].unit = astun.km/astun.s

				astasc.write(data, cat.outloc+'IntegratedFlux_'+cat.runtime+'.csv', format='ecsv')	
				logger.info('Saved Integrated Flux to disk.')
				self.fitparam = [col for col in self.fitparam if col is not None]
				self.fitparamuncert = [col for col in self.fitparamuncert if col is not None]

		thead = astfit.Header()
		thead['N_obj'] = self.nobj
		thead['TFlux'] = (self.tflux, 'units: %s'%(cat.stackunit.to_string()))
		thead['AveRMS_1'] = (self.averms1, 'Jy')
		thead['AveRMS_2'] = (self.averms2, 'Jy')
		thead['AveSig'] = (self.averagenoise, 'Jy')
		thead['AveHImass'] = (cat.avemass, 'Msun')

		if cat.rebinstatus == True and cat.uncert == 'n':
			spec = self.rbspec
			axis = self.rbspectralaxis
			refspec = self.rbrefspec
			origspec = self.spec
			origaxis = self.spectralaxis
			origrefspec = self.refspec
			specuncert = np.zeros(len(axis))
			noiseuncert = np.zeros(10)
			if cat.clean == 'clean':
				noise = np.zeros(10)
			else:
				noise = self.stackrms
		elif cat.rebinstatus == False and cat.uncert == 'n':
			spec = self.spec
			axis = self.spectralaxis
			refspec = self.refspec
			origspec = np.zeros(len(axis))
			origaxis = np.zeros(len(axis))
			origrefspec = np.zeros(len(axis))
			specuncert = np.zeros(len(axis))
			noiseuncert = np.zeros(len(spec))
			if cat.clean == 'clean':
				noise = np.zeros(len(spec))
			else:
				noise = self.stackrms
		elif cat.rebinstatus == True and cat.uncert == 'y':
			spec = self.finalspec
			axis = self.spectralaxis
			refspec = self.finalrefspec
			origspec = self.originalspec
			origaxis = self.originalspectralaxis
			origrefspec = self.originalrefspec
			specuncert = self.specuncert
			noiseuncert = self.noiseuncert
			if cat.clean == 'clean':
				noise =  np.zeros(len(spec))
				noiseuncert =  np.zeros(len(spec))
			else:
				noise = self.stackrms
				noiseuncert = self.noiseuncert
		elif cat.rebinstatus == False and cat.uncert == 'y':
			spec = self.finalspec
			axis = self.spectralaxis
			refspec = self.finalrefspec
			origspec = np.zeros(len(axis))
			origaxis = np.zeros(len(axis))
			origrefspec = np.zeros(len(axis))
			specuncert = self.specuncert
			if cat.clean == 'clean':
				noise = np.zeros(len(spec))
				noiseuncert = np.zeros(len(spec))
			else:
				noise = self.stackrms
				noiseuncert = self.noiseuncert

		# if cat.cluster == True:
		# 	stackunit = uf.msun
		# else:
		stackunit = cat.stackunit
		# 	pass

		## Table 1: Spectrum Information
		d1 = astfit.Column(name='Stacked Spectrum', format='D', unit=stackunit.to_string(), array=spec)
		d2 = astfit.Column(name='Stacked Spectrum Uncertainty', format='D', unit=stackunit.to_string(), array=specuncert)
		d3 = astfit.Column(name='Reference Spectrum', format='D', unit=stackunit.to_string(), array=refspec)
		d4 = astfit.Column(name='Spectral Axis', format='D', unit=cat.spectralunit.to_string(), array=axis)
		d5 = astfit.Column(name='Original Stacked Spectrum', format='D', unit=stackunit.to_string(), array=origspec)
		d6 = astfit.Column(name='Original Reference Spectrum', format='D', unit=stackunit.to_string(), array=origrefspec)
		d7 = astfit.Column(name='Original Spectral Axis', format='D', unit=cat.spectralunit.to_string(), array=origaxis)
		spectable = astfit.BinTableHDU.from_columns([d1,d2,d3,d4])
		origtable = astfit.BinTableHDU.from_columns([d5,d6,d7])

		## Table 2: Stacked Noise
		col1 = astfit.Column(name='Stacked RMS Noise', format='D', unit=cat.stackunit.to_string(), array=noise)
		col2 = astfit.Column(name='Stacked RMS Noise Uncertainty', format='D', unit=cat.stackunit.to_string(), array=noiseuncert)
		noisetable = astfit.BinTableHDU.from_columns([col1,col2])
		if '4' in cat.optnum or '4.' in cat.optnum:
			## Constructing the full HDU
			prihdu = astfit.PrimaryHDU(header=thead)
			thdulist = astfit.HDUList([prihdu,spectable,noisetable,origtable])
		else:
			## Table 3: Fitted Parameters
			parameters = self.fitparam + self.fitparamuncert
			self.paramtable = astfit.BinTableHDU.from_columns( astfit.ColDefs(parameters) )

			## Constructing the full HDU
			prihdu = astfit.PrimaryHDU(header=thead)
			thdulist = astfit.HDUList([prihdu,spectable,noisetable,self.paramtable,origtable])

		if cat.uncert == 'n':
			thdulist.writeto(cat.outloc+'OutputData_'+cat.runtime+'.FITS',clobber=True)
			logger.info('Saved Output Data to disk.')
		elif cat.uncert == 'y' and self.runno == 0:
			thdulist.writeto(cat.outloc+'OutputData_runno=0_'+cat.runtime+'.FITS',clobber=True)
			logger.info('Saved Output Data to disk.')
		else:
			thdulist.writeto(cat.outloc+'OutputData_'+cat.runtime+'.FITS',clobber=True)
			logger.info('Saved Output Data to disk.')
		return


	def savePlots(self, cat, uncert, disp=False, allflux=None, spec=None, spectral=None, refspec=0., specuncert=0., extra=None):
		logger.info('Creating plots.')
		if cat.clean == 'clean':
			plt.figure(200, facecolor='white')
			cols=1
			logger.info('-----There is no noise analysis.-----')
		else:
			cols = 2
			if disp == True: 
				fig = plt.figure(1, figsize=(14,6), facecolor='white')
				ax = fig.add_subplot(122)
			else:
				fig = plt.figure(100)
				ax = fig.add_subplot(111)
			X = range(1,len(self.stackrms)+1,1)
			ax.set_xlabel('Number of stacked spectra')
			if uncert == True:
				ax.errorbar(X, np.log10(self.stackrms), yerr=0.434*(self.noiseuncert/self.stackrms),ls='', ecolor='k', mec='gray', mfc='gray', marker='o', ms=4)
			else:
				ax.plot(X, np.log10(np.array(self.stackrms)), 'ko')
			ax.plot(X, np.log10(self.averagenoise/np.sqrt(X)),'g-')
			ax.set_ylabel(r'Stacked Average Noise ($\mathrm{Jy}$)')
			ax.set_xscale('log')
			yyorig = ax.get_yticks()
			majorloc = np.arange(int(yyorig[0]), int(yyorig[-1])+1., 1.)
			ran1 = np.arange(1.,10.,1.)
			ran2 = np.arange(int(yyorig[0])-1, int(yyorig[-1])+1, 1)
			loc = np.array([ran1*10**float(i) for i in ran2])
			minorloc = np.log10(loc.reshape(1,loc.shape[0]*loc.shape[1])[0])
			ax.set_yticks(minorloc, minor=True)
			ax.set_yticks(majorloc, minor=False)
			ax.set_yticklabels( [r"10$^{%i}$" % a for a in majorloc] )
			ax.set_ylim(yyorig[0], yyorig[-1])
			ax.set_xlim(0.9, self.nobj+0.1*self.nobj)
			if disp == False:
				plt.savefig(cat.outloc+'NoiseAnalysis_'+cat.runtime+'.pdf', format='pdf', bbox_inches='tight', pad_inches=0.2)
				logger.info('Saved Noise Analysis to file.')
		if disp == False:
			plt.figure(101)
		else:
			plt.subplot(1,cols,1)

		## stacked spectrum
		x = np.linspace(spectral[0],spectral[-1],10000)
		if (astun.Jy == cat.stackunit):
			if 1E-3 < max(spec) < 1E-1:
				conv = 1E3
				unitstr = r'Average Stacked Flux Density ($\mathrm{mJy}$)'
			elif max(spec) < 1E-3:
				conv = 1E6
				unitstr = r'Average Stacked Flux Density ($\mathrm{\mu Jy}$)'
			else:
				conv = 1.
				unitstr = 'Average Stacked Flux Density ($\mathrm{Jy}$)'
		elif (uf.msun == cat.stackunit):
			masstext = '%.3E'%np.max(spec)
			masi = masstext.split('E')[0]
			masexp = masstext.split('E')[1]
			conv = 1./eval('1E%s'%masexp)
			unitstr = r'Average Stacked Mass ($10^{%i} \, \mathrm{M_\odot/chan}$)'%float(masexp)
		elif uf.gasfrac == cat.stackunit:
			conv = 1.
			unitstr = r'Average Stacked Gas Fraction (M$_\mathrm{HI}$/M$_\ast$)'
		else:
			conv = 1.
			unitstr = 'Average Stacked Quantity'

		plt.ylabel(unitstr)
		plt.xlim([spectral[0],spectral[-1]])
		xx, locs = plt.xticks()
		ll = ["%5.1f" % a for a in xx]
		plt.xticks(xx,ll)
		plt.xlim([spectral[0],spectral[-1]])
		plt.axhline(0, color='#2E2E2E', ls='--')
		plt.errorbar(spectral, spec*conv, yerr=specuncert*conv, ecolor='k', ls='-', color='k', marker='.', label='Stacked Spectrum')
		plt.plot(spectral, refspec*conv, color='grey',ls='-', marker='.', label='Reference Spectrum')
		plt.xlabel(r'Relative Velocity (km/s)')
		if '4' in cat.optnum or '4.' in cat.optnum:			
			h=0
		else:
			for num in cat.funcnum:
				fname = self.functions[int(eval(num)-1)]
				param = self.paramtable.data[fname]
				func = self.callFunctions(num, cat, None, x, None, uncert=False, param=param, process='plot')
				plt.plot(x, func*conv, color=self.fitcolors[fname], label=fname)
		plt.legend(loc='upper left',fontsize=8, numpoints=1)
		plt.tight_layout()
		if disp == False:
			plt.savefig(cat.outloc+'StackedSpectrum_'+cat.runtime+'.pdf', format='pdf', bbox_inches='tight', pad_inches=0.2)
		else:
			h = 0

		## Histogram of integrated flux
		if uncert == True:

			if (astun.Jy == cat.stackunit):
				if 1E-3 < max(spec) < 1E-1:
					conv = 1E3
					unitstr = r'Ave. Stacked Integrated'+'\n'+r'Flux Density ($\mathrm{mJy}$)'
				elif max(spec) < 1E-3:
					conv = 1E6
					unitstr = r'Ave. Stacked Integrated'+'\n'+r'Flux Density ($\mathrm{\mu Jy}$)'
				else:
					conv = 1.
					unitstr = r'Ave Stacked Integrated'+'\n'+r'Flux Density ($\mathrm{Jy}$)'
			elif (uf.msun == cat.stackunit):
				masstext = '%.3E'%np.max(spec)
				masi = masstext.split('E')[0]
				masexp = masstext.split('E')[1]
				conv = 1./eval('1E%s'%masexp)
				unitstr = r'Ave. Stacked Mass'+'\n'+r'($10^{%i} \, \mathrm{M_\odot}$)'%float(masexp)
			elif uf.gasfrac == cat.stackunit:
				conv = 1.
				unitstr = r'Ave. Stacked Gas Fraction'+'\n'+r'(M$_\mathrm{HI}$/M$_\ast$)'
			else:
				conv = 1.
				unitstr = 'Ave. Stacked Quantity'


			if '4' in cat.optnum or '4.' in cat.optnum:
				options = 7
			else:
				options = np.array([int(eval(i)-1) for i in cat.funcnum])
				options = np.append(7,options)
				l = len(options)
			if type(options) == int:
				plt.figure(3, figsize=( 8,6 ), facecolor='white')
				fname = self.functions[options]
				plt.title(fname)
				plt.xlabel(unitstr)
				plt.ylabel('Number of stacks')
				flux = allflux*conv
				plt.hist( flux[options], bins=20)
				xx, locs = plt.xticks()
				ll = ["%10.3g" % a for a in xx]
				plt.xticks(xx,ll)
				plt.xlim(flux[options].min(), flux[options].max())
				plt.tight_layout()
			else:
				rows = int(np.ceil(l/2.))
				fig = plt.figure(figsize=( 16,8.5 ))
				if extra != None:
					ex = 2.
				else:
					ex = 0.
				gs = gridspec.GridSpec(2,int(np.ceil((len(options)+ex)/2.)))
				grid = [[],[]]
				for n in range(0,l,2):
					## top row	
					grid[0].append(fig.add_subplot(gs[0,int(n/2.)]))				
					grid[0][int(n/2.)].set_title(self.functions[options[n]], fontsize=11)
					grid[0][int(n/2.)].set_xlabel(unitstr)
					grid[0][int(n/2.)].set_ylabel('Number of stacks')
					grid[0][int(n/2.)].hist((allflux[options[n]]*conv), bins=20)
					grid[0][int(n/2.)].set_xlim((allflux[options[n]]*conv).min(), (allflux[options[n]]*conv).max())
					xx = grid[0][int(n/2.)].get_xticks()
					ll = ["%10.3g" % a for a in xx]
					grid[0][int(n/2.)].set_xticks(xx)
					grid[0][int(n/2.)].set_xticklabels(ll)
					grid[0][int(n/2.)].set_xlim((allflux[options[n]]*conv).min(), (allflux[options[n]]*conv).max())

					## bottom row
					if (n+1) != len(options):
						grid[1].append(fig.add_subplot(gs[1,int(n/2.)]))
						grid[1][int(n/2.)].set_title(self.functions[options[n+1]], fontsize=11)
						grid[1][int(n/2.)].set_xlabel(unitstr)
						grid[1][int(n/2.)].set_ylabel('Number of stacks')
						grid[1][int(n/2.)].hist((allflux[options[n+1]]*conv), bins=20)
						grid[1][int(n/2.)].set_xlim((allflux[options[n+1]]*conv).min(), (allflux[options[n+1]]*conv).max())
						xx = grid[1][int(n/2.)].get_xticks()
						ll = ["%10.3g" % a for a in xx]
						grid[1][int(n/2.)].set_xticks(xx)
						grid[1][int(n/2.)].set_xticklabels(ll)
						grid[1][int(n/2.)].set_xlim((allflux[options[n+1]]*conv).min(), (allflux[options[n+1]]*conv).max())
					else:
						h = 0


				if extra != None:
					ax1 = fig.add_subplot(gs[0,-1])
					ax1.set_title('Mean of Single Gaussian Fit', fontsize=11)
					ax1.set_xlabel('km/s')
					ax1.set_ylabel('Number of stacks')
					minbin = np.floor(extra[0].min())
					maxbin = np.ceil(extra[0].max())
					binrange = np.arange(minbin, maxbin+0.1, 5.)
					ax1.hist(extra[0], bins=binrange, color='limegreen')
					ax1.set_xlim(extra[0].min(), extra[0].max())
					xx = ax1.get_xticks()
					ll = ["%.3g" % a for a in xx]
					ax1.set_xticks(xx)
					ax1.set_xticklabels(ll)
					ax1.set_xlim(extra[0].min(), extra[0].max())

					ax2 = fig.add_subplot(gs[1,-1])
					ax2.set_title('RMS of Stacked Spectrum', fontsize=11)
					ax2.set_xlabel(unitstr)
					ax2.set_ylabel('Number of stacks')
					ax2.hist(extra[1]*conv, bins=20, color='limegreen')
					ax2.set_xlim((extra[1]*conv).min(), (extra[1]*conv).max())
					xx = ax2.get_xticks()
					ll = ["%.3g" % a for a in xx]
					ax2.set_xticks(xx)
					ax2.set_xticklabels(ll)
					ax2.set_xlim((extra[1]*conv).min(), (extra[1]*conv).max())
				else:
					h = 0

			plt.tight_layout()

			if disp == False:
				plt.savefig(cat.outloc+'FluxDistribution_'+cat.runtime+'.pdf', format='pdf', bbox_inches='tight', pad_inches=0.2)
			else:
				h = 0

		if disp == True:
			plt.show()
		else:
			plt.close('all')
		logger.info('Saved plots.')
		return


	def analyse_data(self, other, cat, runno):
		try:
			self.runno = runno
			self.__fillInitial(other, cat, runno)
			cat = self.__callAnalysis(cat)
			self.intflux[7] = (np.nansum(self.spec[ cat.maskstart: cat.maskend ]))
			self.tflux = self.intflux[7] * other.nobj
			if (cat.stackunit == astun.Jy) & (cat.cluster == False):
				cat.avemass = (2.356E+05 * ( cat.mediandistance.value**2 ) * (self.intflux[7] * cat.restdv.value) ) / (1. + cat.medianz)
			if (cat.stackunit == astun.Jy) & (cat.cluster == True):
				cat.avemass = (2.356E+05 * ( cat.clusterDL.value**2 ) * (self.intflux[7] * cat.restdv.value) ) / (1. + cat.clusterZ)
			elif cat.stackunit == uf.gasfrac:
				cat.avemass = self.intflux[7] * (cat.avesm).to(uf.msun, equivalencies=uf.log10conv)
				self.tflux = self.intflux[7] * other.nobj
			else:
				cat.avemass = self.intflux[7]

			#print(cat.stackunit, cat.avemass, cat.mediandistance, cat.cluster, cat.restdv.value, self.intflux[7])

			if cat.uncert == 'n':
				logger.info('Saving data to ouput location.')
				self.saveResults(cat, False)
				if cat.rebinstatus == True:
					self.savePlots(cat, uncert=False, disp=False, allflux=None, spec=self.rbspec, spectral=self.rbspectralaxis, refspec=self.rbrefspec, specuncert=self.specuncert)
				else:
					self.savePlots(cat, uncert=False, disp=False, allflux=None, spec=self.spec, spectral=self.spectralaxis, refspec=self.refspec, specuncert=self.specuncert)
			else: 
				pass
			cat.rebinstatus = self.rebinstatus

			return cat
		except KeyboardInterrupt:
			uf.earlyexit(cat)
			raise sys.exit()
		except SystemExit:
			raise sys.exit()
		except:
			logger.error('Something happened :(', exc_info=True)
			uf.earlyexit(cat)
			raise sys.exit()

