""" controls the stackData object """

import numpy as np
import astropy.constants as astcon
import astropy.cosmology as astcos
import astropy.modeling as astmod
import astropy.units as astun
import astropy.convolution as astcov
import useful_functions as uf
import matplotlib.pyplot as plt
from analyseData import anaData
import csv
import sys
import os
import timeit
import logging

logger = logging.getLogger(__name__)

Msun = astun.def_unit('Msun',astun.Jy*astun.Mpc**2*astun.km/(astun.s) )

c = astcon.c.to('km/s') 

msun = [
	(astun.Jy*astun.km*astun.Mpc*astun.Mpc/astun.s, astun.solMass, lambda x: 1*x, lambda x: 1*x)
]


log10 = [
	(astun.dex*astun.Msun, astun.Msun, lambda x: 10**x, lambda x: np.log10(x))
]

class objSpec():
	""" class contronlling each of the spectra,
	Inputs:
		z: redshift of the galaxy
		rz: random redshift
		spec: the spectrum's flux
		freq: the spectrum's frequency axis
		femit: the rest frequency of the emission/absorption line
		cosmology: an astropy object containting the comosmology information """
	def __init__(self):
		self.redshift = None
		self.randredshift = None
		self.origspec = None
		self.massspec = None
		self.refmassspec = None
		self.shiftorigspec = None
		self.shiftrefspec = None
		self.extendspec = None 
		self.extendrefspec = None
		self.spectralspec = None
		self.rms = None
		self.noisespec = None
		self.noisespecmass = None
		self.orignoisespec = None
		self.dl = None
		self.weight = None
		self.status = None
		self.file = None
		self.objid = None
		self.totalflux = None
		self.dv = None
		self.stellarmass = None
		self.dvkms = None

	def __smoothData(self, spec):
		windowsize = 3
		kernel = astcov.Box1DKernel(windowsize)
		sspec = astcov.convolve(spec, kernel)
		return sspec

	def __getSpectrumZOK(self, cat, n, runno):
		if runno > 1:
			if 'red' in cat.uncerttype:
				uz = cat.randuz[n][runno-2]
				z = cat.catalogue['Redshift'][n] + uz 
			else:
				z = cat.catalogue['Redshift'][n]
		else:
			z = cat.catalogue['Redshift'][n] 

		if cat.min_redshift < cat.catalogue['Redshift'][n] < cat.max_redshift: 
			self.redshift = z
			self.randredshift = cat.randz[runno][n]
			return True, cat
		else:
			self.status = 'incomplete'
			logger.warning('Spectrum ID: %s, REJECTED: Out of the redshift range'%str(cat.catalogue['Object ID'][n]))
			return False, cat

	def __getStellarMass(self, cat, n):
		if cat.catalogue['Stellar Mass'][n] == 0. or cat.catalogue['Stellar Mass'][n] =='masked':
			logger.warning('REJECTED %s - No Stellar Mass'%str(cat.catalogue['Object ID'][n]))
			return False, cat
		else:
			self.stellarmass = (cat.catalogue['Stellar Mass'][n] * cat.catalogue['Stellar Mass'].unit).to(astun.Msun, log10)
			logger.info('Logged stellar mass for %s'%str(cat.catalogue['Object ID'][n]))
			return True, cat


	def __getSpectrumDVOK(self, spec, cat, n):
		checkdv = np.mean(np.array([abs(spec[0]-spec[1]), abs(spec[-1]-spec[-2])]))*cat.spectralunit
		self.dv = checkdv
		z = cat.catalogue['Redshift'][n]
		if 'Hz' in cat.spectralunit.to_string():
			self.dvkms = uf.calc_dv_from_df(self.dv, z, cat.femit)
		else:
			self.dvkms = self.dv.to('km/s')

		if 0.95*cat.channelwidth < checkdv < 1.05*cat.channelwidth:
			return True, cat
		elif 0.9*cat.channelwidth < checkdv < 1.1*cat.channelwidth:
			logger.warning(r'Spectrum ID: %s, WARNING: Channel width is 5\%-10\% different from expected spectrum channel width'%str(cat.catalogue['Object ID'][n]))
			return True, cat
		else:
			self.status = 'incomplete'
			logger.warning(r'Spectrum ID: %s, REJECTED: Channel width is more than 5\%-10\% different from expected spectrum channel width'%str(cat.catalogue['Object ID'][n]))
			return False, cat
		

	def __getSpectrumlenOK(self, spec, cat, n):
		checklen = len(spec)
		if checklen > (int(cat.stackmask)):
			return True, cat
		else:
			self.status = 'incomplete'
			logger.warning(r'Spectrum ID: %s, REJECTED: spectrum is too short'%str(cat.catalogue['Object ID'][n]))
			return False, cat


	def __getSpectrum(self, cat, n, runno):
		self.objid = cat.catalogue['Object ID'][n]
		filexist = os.path.isfile(cat.specloc+cat.catalogue['Filename'][n])
		if filexist == True:
			try:
				if cat.rowdelim.lower() == "none":
					data = np.loadtxt(cat.specloc+cat.catalogue['Filename'][n], usecols=cat.speccol,unpack=True, skiprows=cat.rowstart)
				else:
					data = np.loadtxt(cat.specloc+cat.catalogue['Filename'][n], usecols=cat.speccol,unpack=True, skiprows=cat.rowstart, delimiter=cat.rowdelim)
				checkZ, cat = self.__getSpectrumZOK(cat, n, runno)
				checkDV, cat = self.__getSpectrumDVOK(data[0], cat, n)
				checklen, cat = self.__getSpectrumlenOK(data[1], cat, n)
				if cat.stackunit == uf.gasfrac:
					checkSM, cat = self.__getStellarMass(cat, n)
				else:
					checkSM = True
				if checkDV and checkZ and checklen and checkSM:
					################################
					# for testing puposes only
					#------------------------------------------------------#
					# random gaussian noise
					#------------------------------------------------------#
					# s = 0.6*np.max(data[1])#abs(np.random.normal(0,scale=np.max(data[1]), size=1))
					# data[1] += np.random.normal(0, s, size=len(data[1]))
					#------------------------------------------------------#
					# making absorption spectra
					#------------------------------------------------------#
					# data[1] = data[1]*(-1)
					#------------------------------------------------------#
					# coarsening resolution
					#------------------------------------------------------#
					# self.origspec = self.__smoothData(data[1])*cat.fluxunit/cat.convfactor
					################################
					spec = data[1]*cat.fluxunit/cat.convfactor
					self.origspec = spec.to(astun.Jy)
					if cat.veltype == 'optical':
						self.spectralspec = data[0]*cat.spectralunit 
					else: 
						self.spectralspec = data[0]*(1.+self.redshift)*cat.spectralunit 
					return cat
				else:
					self.status = 'incomplete'
					return cat
			except KeyboardInterrupt:
				logger.error('Encountered an exception:', exc_info=True)
				uf.earlyexit(self)
				raise KeyboardInterrupt
			except Exception as e:
				logger.error('Encountered an exception:', exc_info=True)
				self.status = 'incomplete'
			return cat
		else:
			logger.warning('Spectrum ID: %s has no corresponding spectrum file.'%str(cat.catalogue['Object ID'][n]))
			self.status = 'incomplete'
			return cat

		
	def __calcRestFrame(self, cat):
		""" convert the frequency spectrum from observed frame to rest frame """
		restaxis = self.spectralspec*(1+self.redshift)*self.redshift*astcon.c.to('km/s')/cat.femit
		restflux = self.origspec/(1+self.redshift)
		
		return restaxis, restflux


	def __calcNoiseRMS(self, spec, cat, spec2=None):
		if self.status == 'incomplete':
			return None, None
		else:
			""" create the noise spectrum and calculate the rms value of the spectrum """
			l = cat.noiselen
			s = len(spec)
			if spec2 is None:
				specv = spec.value
				noisespec = np.zeros(l)*spec.unit
				masked = np.append(specv[:s/2-cat.stackmask/2],specv[s/2+cat.stackmask/2:])*spec.unit
				for i in xrange(l/2):
					noisespec[i] = masked[i % len(masked)]
					noisespec[-(i+1)] = masked[-(i+1) % len(masked)]
				rms = np.std(noisespec)
				return rms, noisespec
			else:
				specv = spec.value
				specv2 = spec2.value
				noisespec = np.zeros(l)*spec.unit
				noisespec2 = np.zeros(l)*spec2.unit
				masked = np.append(specv[:s/2-cat.stackmask/2],specv[s/2+cat.stackmask/2:])*spec.unit
				masked2 =  np.append(specv2[:s/2-cat.stackmask/2],specv2[s/2+cat.stackmask/2:])*spec2.unit
				for i in xrange(l/2):
					noisespec[i] = masked[i % len(masked)]
					noisespec2[i] = masked2[i % len(masked)]
					noisespec[-(i+1)] = masked[-(i+1) % len(masked)]
					noisespec2[-(i+1)] = masked2[-(i+1) % len(masked)]
				rms = np.std(noisespec)
				return rms, noisespec, noisespec2		



	def __calcWeights(self, cat):
		if self.status == 'incomplete':
			logger.error("Error calculation on weight calculation for spectrum ID %s"%str(self.objid))
			return
		else:
			if cat.weighting == '1':
				self.weight = 1.
			elif cat.weighting == '2':
				if self.rms.value == 0.:
					self.weight = 1.
				else:
					self.weight = 1./(self.rms.value)
			elif cat.weighting == '3':
				if self.rms.value == 0.:
					self.weight = 1.
				else:
					self.weight = 1./(self.rms.value**2)
			elif cat.weighting == '4':
				if self.rms.value == 0.:
					self.weight = 1.
				else:
					self.weight = (self.dl.value)**2/(self.rms.value**2)


	def __calcShift(self, cat):
		"""shift the spectrum sunch that the galaxy location is centred"""
		if self.status == 'incomplete':
			logger.error("Error with shifting spectrum %s"%str(self.objid))
			return
		else:
			if 'Hz' in cat.spectralunit.to_string():
				axis, spec = self.__calcRestFrame(cat)
			else:
				axis, spec = self.spectralspec, self.origspec

			l = len(self.origspec)
			cen = int(l/2)
			self.shiftorigspec = np.zeros(l)*spec.unit
			self.shiftrefspec = np.zeros(l)*spec.unit
			
			cz = astcon.c.to('km/s')*self.redshift
			cz_ran = astcon.c.to('km/s')*self.randredshift
			galaxyloc = np.argmin(abs(axis - cz))
			galaxyloc_ran = np.argmin(abs(axis - cz_ran))
			for i in xrange(l):
				self.shiftorigspec[i] = spec[(galaxyloc-cen+i) % l]
				self.shiftrefspec[i] = spec[(galaxyloc_ran-cen+i) % l]

			return
			

	def __calcMassConversion(self, catalogue, spec, refspec):
		if self.status == 'incomplete':
			logger.error('Error on conversion to mass spectrum for spectrum %s'%str(self.objid))
			return
		else:
			try:
				massspec = (2.356E+05 * spec * self.dl**2 * self.dvkms/(1.+self.redshift)).to(astun.solMass, msun)
				refmassspec = (2.356E+05 * refspec * self.dl**2 * self.dvkms/(1.+self.randredshift)).to(astun.solMass, msun)
			except:
				self.status = 'incomplete'
				logger.error('Encountered an exception:', exc_info=True)
				return

			if catalogue.stackunit == uf.msun:
				return massspec, refmassspec
			elif catalogue.stackunit == uf.gasfrac:
				massspec, refmassspec = (massspec/self.stellarmass)*uf.gasfrac, (refmassspec/self.stellarmass)*uf.gasfrac
				return massspec, refmassspec


	def __calcExtendSpectrum(self, cat, spec, rspec, nspec):
		""" extend each spectrum such that they are all the same length """
		if self.status == 'incomplete':
			logger.error("Error with extending spectrum %s"%str(self.objid))
			return
		else:
			tl = cat.exspeclen
			extendspec = np.zeros(tl)*spec.unit
			extendrefspec = np.zeros(tl)*rspec.unit
			cent = int(tl/2)
			ol = len(spec)
			ceno = int(ol/2)
			if (tl-ol) <= 0:
				r = cent
			elif (tl-ol) > 0:
				r = ceno
			for i in xrange(r):
				extendspec[cent+i] = spec[ceno+i]
				extendspec[cent-(i+1)] = spec[ceno-(i+1)]
				extendrefspec[cent+i] = rspec[ceno+i]
				extendrefspec[cent-(i+1)] = rspec[ceno-(i+1)]
			bl = int((tl - ol)/2)
			nl = len(nspec)
			for i in xrange(bl):
				extendspec[i] = nspec[i % nl]
				extendspec[-(i+1)] = nspec[-(i+1) % nl]
				extendrefspec[i] = nspec[i % nl]
				extendrefspec[-(i+1)] = nspec[-(i+1) % nl]
			# extendspec[-1] = nspec[i%nl]
			# extendrefspec[-1] = nspec[i%nl]
			return extendspec, extendrefspec

	
	def __plot_process(self, axes, fig, cat):
		if axes==None or fig==None:
			return
		else:
			axes[0].plot(self.spectralspec.value, self.origspec.value)
			axes[1].plot(cat.spectralaxis.value, self.extendspec.value)
			fig.canvas.draw()

	
	def callModule(self, cat, n, runno, axes=None, fig=None):
		try:
			cat = self.__getSpectrum(cat, n, runno)
			if self.origspec is None:
				return cat
			else:
				self.__calcShift(cat)
				self.dl = cat.cosmology.luminosity_distance(self.redshift)
				if uf.msun == cat.stackunit:
					self.massspec, self.refmassspec = self.__calcMassConversion(cat, self.shiftorigspec, self.shiftrefspec)
					self.rms, self.noisespec, self.noisespecmass = self.__calcNoiseRMS(self.shiftorigspec, cat, self.massspec)
					self.extendspec, self.extendrefspec = self.__calcExtendSpectrum(cat, self.massspec, self.refmassspec, self.noisespecmass)
					self.__calcWeights(cat)
				elif astun.Jy == cat.stackunit: 
					self.rms, self.noisespec = self.__calcNoiseRMS(self.shiftorigspec, cat)
					self.extendspec, self.extendrefspec = self.__calcExtendSpectrum(cat, self.shiftorigspec, self.shiftrefspec, self.noisespec)
					self.__calcWeights(cat)
				elif uf.gasfrac == cat.stackunit:
					self.massspec, self.refmassspec = self.__calcMassConversion(cat, self.shiftorigspec, self.shiftrefspec)
					self.rms, self.noisespec, self.noisespecmass = self.__calcNoiseRMS(self.shiftorigspec, cat, self.massspec)
					self.extendspec, self.extendrefspec = self.__calcExtendSpectrum(cat, self.massspec, self.refmassspec, self.noisespecmass)
					self.__calcWeights(cat)
				else:
					logger.error("Something went wrong with deciding the stack unit and how to stack")
					uf.earlyexit(cat)
					raise sys.exit()

				startind = len(self.shiftorigspec)/2-cat.stackmask/2
				endind = len(self.shiftorigspec)/2+1+cat.stackmask/2

				intfluxspec = (self.shiftorigspec*self.dvkms).to(astun.Jy * astun.km/astun.s)
				intflux = np.sum( intfluxspec.value[startind:endind+1] )
				if runno == 0:
					cat.outcatalogue.add_row([ cat.catalogue['Bin Number'][n], self.objid, cat.catalogue['Filename'][n], cat.catalogue['Redshift'][n], cat.catalogue['Redshift Uncertainty'][n], cat.catalogue['Stellar Mass'][n], cat.catalogue['Other Data'][n],  intflux ])
				else:
					h = 0
		except KeyboardInterrupt:
			logger.error("Encountered exception.", exec_info=True)
			uf.earlyexit(cat)
			raise KeyboardInterrupt
		except SystemExit:
			logger.error("Encountered exception.", exec_info=True)
			uf.earlyexit(cat)
			raise sys.exit()
		return cat


class objStack(objSpec):
	def __init__(self, spec=[], ref=[], noise=[], rms=[], stackrms=[], dist=[], weight=0., z=[], nobj=0, spectral=[], sunits=None):
		self.spec = spec
		self.refspec = ref
		self.dl = dist
		self.noise = noise
		self.rms = rms
		self.stackrms = stackrms
		self.redshift = z
		self.nobj = nobj
		self.spectralaxis = spectral
		self.weightsum = weight
		self.stackunits = sunits

	def __add__(self, other):
		if other.extendspec is None or other.weight == 0.:
			return self
		else:
			spec = np.add(self.spec, other.extendspec*other.weight)
			refspec = np.add(self.refspec, other.extendrefspec*other.weight)
			weight = self.weightsum + other.weight
			nobj = self.nobj + 1
			dist = list(self.dl) + [other.dl]
			noise = np.add(self.noise, other.noisespec*other.weight)
			rms = (self.rms + [(np.std(other.noisespec.value*other.weight)/other.weight)])
			srms = self.stackrms+[(np.std(noise.value)/weight)]
			redshift = list(self.redshift) + [other.redshift]
			return objStack(spec=spec, ref=refspec, noise=noise, rms=rms,stackrms=srms, dist=dist,weight=weight, z=redshift, nobj=nobj,spectral=self.spectralaxis, sunits=self.stackunits)


	def __radd__(self, other):
		if other.spec is None:
			return self
		else:
			return self.__add__(other)



