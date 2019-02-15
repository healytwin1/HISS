import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as astfit
import astropy.io.ascii as astasc
import astropy.table as asttab
import astropy.modeling as astmod
import astropy.units as astun
import sys, time
import scipy.optimize as opt
import useful_functions as uf
from scipy.special import gammainc, erfinv, gammaincc
from scipy.stats import chisquare
from analyseData import anaData
import multiprocessing as multi
import _pickle as pck
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class uncertAnalysis(anaData):

	def __init__(self, cat, other, axis=None, allrefspec=[], allspec=[], allnoise=[], allrms=[], middle=[], allintflux=np.array([[],[],[],[],[],[],[],[]])):
		anaData.__init__(self)
		self.nobj = other.nobj
		self.averms1 = other.averms1
		self.averms2 = other.averms2
		self.averagenoise = other.averagenoise
		self.stackrms = None
		self.weightsum = other.weightsum
		self.allintflux = allintflux
		self.specuncert = None
		self.intflux = np.zeros((8,2))
		self.fitrms = np.zeros((8,2))
		self.fitparam = [None, None, None, None, None, None, None]
		self.fitparamuncert = [None, None, None, None, None, None, None]
		self.paramtable = None
		self.originalspec = None
		self.originalspectralaxis = None
		self.tflux = other.tflux
		self.noiseerrupp = None
		self.noiseerrlow = None
		self.noiseuncert = None
		self.allspec = allspec
		self.allrefspec = allrefspec
		self.allnoise = allnoise
		self.allrms = allrms
		self.otherparamtable = None
		self.middle = middle
		self.maskstart = cat.maskstart
		self.maskend = cat.maskend
		self.originalspec = other.spec
		self.originalspectralaxis = other.spectralaxis
		self.spectralaxis = axis
		self.originalrefspec = other.refspec
		self.count = 0



	def updateStackAnalysis(self, cat, n, other):
		if cat.rebinstatus == True:
			spectrum = other.rbspec
			axis = other.rbspectralaxis
			refspec = other.rbrefspec
		else:
			spectrum = other.spec
			axis = other.spectralaxis
			refspec = other.refspec

		self.spectralaxis = axis

		try:
			param = other.fitparam[0].array
			self.middle = param[1]
		except:
			self.middle = 0.
		self.allrms = np.std( np.array( list(spectrum[:self.maskstart]) + list(spectrum[self.maskend:]) ) )
		self.allspec = (np.array(spectrum)).reshape(len(spectrum), 1)
		self.allnoise =(np.array(other.stackrms)).reshape(len(other.stackrms), 1)
		self.allrefspec = (np.array(refspec)).reshape(len(refspec), 1)

		self.allintflux = other.intflux.reshape(8,1)
		self.count = n
		pck.dump( self, open('%sstackertemp/runninganalysis_%s_%04i.pkl'%(cat.outloc, cat.runtime, n), 'wb') )	
		return

	def load(self, n, cat):
		analysisload = pck.load( open('%sstackertemp/runninganalysis_%s_%04i.pkl'%(cat.outloc, cat.runtime, n), 'rb') )
		return analysisload

		
	
	def __processUncertainties(self, cat):
		logger.info('Processing uncertainties.')
		self.allspec = np.array(self.allspec)
		self.allnoise = np.array(self.allnoise)
		self.allrms = np.array(self.allrms)
		self.middle = np.array(self.middle)
		self.allrefspec = np.array(self.allrefspec)

		self.finalspec = np.mean(self.allspec, axis=1)
		self.stackrms = np.mean(self.allnoise, axis=1)
		self.finalrefspec = np.mean(self.allrefspec, axis=1)
		

		if cat.uncerttype == 'redshift':
			self.uppererror = np.absolute(np.percentile(self.allspec, 75, axis=1) - self.finalspec)
			self.lowererror = np.absolute(np.percentile(self.allspec, 25, axis=1) - self.finalspec)
			self.specuncert = np.mean(np.array([self.uppererror, self.lowererror]), axis=0)
			
			self.noiseerrupp = np.absolute(np.percentile(self.allnoise, 75, axis=1) - self.stackrms)
			self.noiseerrlow = np.absolute(np.percentile(self.allnoise, 25, axis=1) - self.stackrms)
			self.noiseuncert = np.mean(np.array([self.noiseerrupp, self.noiseerrlow]), axis=0)
		else:
			R = self.allspec.shape[1]
			newspec = np.empty((self.allspec.shape))
			newnois = np.empty((self.allnoise.shape))
			for i in range(R):
				newspec[:,i] = (self.allspec[:,i] - self.finalspec)**2
				newnois[:,i] = (self.allnoise[:,i] - self.stackrms)**2

			# self.uppererror = np.sqrt( (R-1)/R * np.sum(newspec, axis=1 ) )
			# self.lowererror = np.sqrt( (R-1)/R * np.sum(newspec, axis=1 ) )
			# self.specuncert = np.sqrt( (R-1)/R * np.sum(newspec, axis=1 ) )

			# self.noiseerrupp = np.sqrt( (R-1)/R * np.sum(newnois, axis=1 ) )
			# self.noiseerrlow = np.sqrt( (R-1)/R * np.sum(newnois, axis=1 ) )
			# self.noiseuncert = np.sqrt( (R-1)/R * np.sum(newnois, axis=1 ) )

			self.uppererror = np.absolute(np.percentile(self.allspec, 75, axis=1) - self.finalspec)
			self.lowererror = np.absolute(np.percentile(self.allspec, 25, axis=1) - self.finalspec)
			self.specuncert = np.mean(np.array([self.uppererror, self.lowererror]), axis=0)
			
			self.noiseerrupp = np.absolute(np.percentile(self.allnoise, 75, axis=1) - self.stackrms)
			self.noiseerrlow = np.absolute(np.percentile(self.allnoise, 25, axis=1) - self.stackrms)
			self.noiseuncert = np.mean(np.array([self.noiseerrupp, self.noiseerrlow]), axis=0)

		return


	def ProcessUncertainties(self, cat):
		self.__processUncertainties(cat)
		cen = len(self.finalspec)/2
		wid = cat.mask
		spec = self.finalspec
		refspec = self.finalrefspec
		axis = self.spectralaxis
		uncert = self.specuncert

		ind = np.where(uncert == 0.)[0]
		uncert[ind] = np.ones(len(ind))*np.mean(self.allrms)

		if '4' in cat.optnum or '4.' in cat.optnum:
			h = 0
		else:
			for i in cat.funcnum:
				self.callFunctions(i, cat, spec, axis, uncert, True, param=None, process='fit')

		self.intflux = np.zeros((8,2))
		self.intflux[:,0] = np.nanmean(self.allintflux, axis=1)
		self.intflux[:,1] = np.nanstd(self.allintflux, axis=1)
		logger.info('Saving uncertainty data to output location')

		self.saveResults(cat, True)
		self.savePlots(cat, uncert=True, allflux=self.allintflux, spec=self.finalspec, spectral=self.spectralaxis, refspec=self.finalrefspec, specuncert=self.specuncert, extra=[self.middle, self.allrms])
		return


	def __add__(self, other):
		if (np.array(self.allspec)).shape[0] == 0:
			allspec = (np.array(other.allspec)).reshape(len(other.allspec), 1)
			allrefspec = (np.array(other.allrefspec)).reshape(len(other.allrefspec), 1)
			allnoise = (np.array(other.allnoise)).reshape(len(other.allnoise), 1)
		else:
			allspec = np.append(np.array(self.allspec), (np.array(other.allspec)).reshape(len(other.allspec), 1), axis=1)
			allrefspec = np.append(np.array(self.allrefspec), (np.array(other.allrefspec)).reshape(len(other.allrefspec), 1), axis=1)
			allnoise = np.append( np.array(self.allnoise), (np.array(other.allnoise)).reshape(len(other.allnoise), 1), axis=1 )
		if type(self.allrms) == np.float64:
			incallrms = [self.allrms]
		else: 
			incallrms = list(self.allrms)
		allrms = incallrms + [other.allrms]
		middle = np.append(self.middle, other.middle)
		allintflux = np.append(np.array(self.allintflux), np.array(other.allintflux), axis=1)
		axis = other.spectralaxis
	
		return uncertAnalysis(other, other, axis=axis, allrefspec=allrefspec, allspec=allspec, allnoise=allnoise, allrms=allrms, middle=middle, allintflux=allintflux)

	def __radd__(self, other):
		if other.spec is None:
			return self
		else:
			return self.__add__(other)


			
