""" controls the input data """
import numpy as np
import astropy.cosmology as astcos
import astropy.io.ascii as astasc
import astropy.constants as astcon
import astropy.units as astun
import astropy.table as asttab
import sys
import time
import os
import useful_functions as uf
import json
import logging

logging.disable(logging.DEBUG)
logger = logging.getLogger(__name__)

c = astcon.c.to('km/s')

class inputCatalogue(object):
	""" class to store all input information, the catalogue input information 
	Default values:
		minimum redshift: 0.1
		maximum redshift: 0.2
		Hubble constant (H0): 70 km/s/Mpc
		Omega matter: 0.3 """

	def __init__(self, time):
		self.catalogue = None
		self.randuz = None
		self.randz = None
		self.keys = None
		self.catfilename = None
		self.numcatspec = None
		self.max_redshift = 0
		self.min_redshift = 0
		self.cosmology = astcos.FlatLambdaCDM(70, 0.3)
		self.spectralunit = None
		self.veltype = "optical"
		self.fluxunit = None
		self.stackunit = None
		self.channelwidth = None
		self.restdv = None
		self.femit = 1420.40575177*astun.MHz
		self.specloc = ''
		self.rowstart = None
		self.rowdelim =','
		self.speccol = None
		self.noiselen = 0
		self.mask = 0
		self.stackmask = 0
		self.exspeclen = 0
		self.spectralaxis = None
		self.outloc = './output/'
		self.maxgalw = 450.*astun.km/astun.s
		self.medianz = 0.
		self.runtime = time
		self.weighting = '1'
		self.maxstackw = 2000.*astun.km/astun.s
		self.convfactor = 1.
		self.uncert = 'y'
		self.mc = 1000
		self.optnum = [] 
		self.funcnum = [] 
		self.rbnum = []
		self.linetype = 'emission'
		self.rebinstatus = False
		self.smoothalgorithms = {'1': 'HanningWindow', '2': 'Boxcar'}
		self.smoothtype = None
		self.smoothwin = None
		self.functions = ['Single Gaussian', 'Double Gaussian', 'Lorentzian Distribution Function', 'Voigt Profile', '3rd Order Gauss-Hermite Polynomial', 'Busy Function (with double horn)', 'Busy Function (without double horn)','Summed Flux of Stacked Spectrum']
		self.outcatalogue = asttab.Table(names=['Bin Number', 'Object ID', 'Filename', 'Redshift', 'Redshift Uncertainty', 'Stellar Mass', 'Other Data', 'Integrated Flux'], dtype=('i8','S30','S30','f8','f8','f8','f8','f8'), masked=True)
		self.maskstart = None
		self.maskend = None
		self.mean = 0.
		self.suppress = ''
		self.clean = ''
		self.progress = ''
		self.saveprogress = ''
		self.multiop = ''
		self.latex = ''
		self.uncerttype = 'redshift'
		self.R = 0.75
		self.mediandistance = 0.*astun.Mpc
		self.avemass = 0. 
		self.avesm = 1.
		self.w50 = 0 * astun.km/astun.s
		self.cluster = False
		self.config = False


	def updateconfig(self, config, maxgalw, rebinstatus, smoothtype, smoothwin, funcnum, optnum):
		config['IntegrateGalaxyWidth'] = maxgalw
		config['SmoothStatus'] = rebinstatus
		config['SmoothType'] = smoothtype
		config['SmoothWindow'] = smoothwin
		config['FittedFunctionNumbers'] = funcnum
		config['AnalysisOptions'] = optnum
		configfile = json.dumps(config, indent=4, sort_keys=False, separators=(',', ': ') )
		optfile = open(self.outloc+'ConfigFile_'+self.runtime, 'w')
		optfile.write(configfile)
		optfile.close()
		return

	def __fill_catalogue(self, table, cattable, colname, n, colnumlist=None):

		if colnumlist is None:
			colnum = input("Please enter the column number of %s (if this column isn't necessary, please leave blank): "%colname)
		else:
			if type(colnumlist[n]) != str:
				colnum = str(colnumlist[n])
			else:
				colnum = colnumlist[n]
		if len(colnum) == 0:
			newcol = asttab.MaskedColumn(name=colname, data=np.zeros(len(table)), mask=[True]*len(table))
			table.add_column(newcol)
			return table
		else:
			if colnum.isalnum():
				m = int(eval(colnum))
				cattablecolnames = cattable.colnames
				newcol = asttab.MaskedColumn(data=cattable[cattablecolnames[m]].data, name=colname, mask=[False]*len(table))
				table.add_column(newcol)
				return table
			else:
				print ("There was a problem understanding the column number, please try again.")
				return self.__fill_catalogue(table, colname, n, colnum)


	def __get_astropy_catalogue(self, config):
		print ("Initialising the catalogue file.")
		logger.info('Initialising the catalogue file.')

		if uf.checkkeys(config, 'CatalogueFilename'):
			self.catfilename = config['CatalogueFilename']
		else:
			self.catfilename = input('Enter location and filename of catalogue file: ')
		while not os.path.isfile(self.catfilename):
			print ('The catalogue filename and location you have given does not exist.\n')
			self.catfilename = input('Enter location and filename of catalogue file: ')
		try:
			filetable = astasc.read(self.catfilename) # [, format='csv'] took out 10/01/19 when line crashed with .dat on JD system
		except KeyboardInterrupt:
			uf.earlyexit(self)
			raise sys.exit()
		except:
			print ("Did not recognise the catalogue data type. Please check that your catalogue file is in the same format as the example catalogue file.")
			logger.critical("Did not recognise the catalogue file data type.")
			raise sys.exit()

		if uf.checkkeys(config, 'CatalogueColumnNumbers'):
			catcols = config['CatalogueColumnNumbers']
		else:
			catcols = None
			filetablenames = filetable.colnames
			print ('\nThe following columns are available:')
			for u in range(len(filetablenames)):
				print ('%i: %s'%(u, filetablenames[u]))
				
		colnames = ['Object ID', 'Filename', 'Redshift', 'Redshift Uncertainty', 'Stellar Mass', 'Other Data']
		catalogue = asttab.Table(names=['Dud'], dtype=['f8'], masked=True, data=[np.zeros(len(filetable))])

		for o in range(len(colnames)):
			try:
				self.__fill_catalogue(catalogue, filetable, colnames[o], o, catcols)
			except KeyboardInterrupt:
				uf.earlyexit(self)
			except SystemExit():
				uf.earlyexit(self)
				raise sys.exit()
			except:
				print ('There was problem with entering the column number, please try again.\n')
				self.__fill_catalogue(catalogue, filetable, colnames[o], o, None)
		catalogue.remove_column('Dud')

		## check units of the stellar mass
		if True in catalogue['Stellar Mass'].mask:
			if self.stackunit == uf.gasfrac:
				print ("\nTo stack in gas fraction, the catalogue file must include the Stellar Mass for each profile.")
				logger.info('To stack in gas fraction, the catalogue file must include the Stellar Mass for each profile.')
				uf.earlyexit(self)
		else:
			checkval = str(int(catalogue['Stellar Mass'][0]))
			if len(checkval) < 3:
				catalogue['Stellar Mass'].unit = astun.dex*uf.msun
			else:
				catalogue['Stellar Mass'].unit = uf.msun

			self.avesm = np.mean( catalogue['Stellar Mass'].data ) * catalogue['Stellar Mass'].unit


		uniquecatalogue = asttab.unique(catalogue, keys='Object ID')
		self.catalogue = uniquecatalogue
		self.medianz = np.median( self.catalogue['Redshift'].data )
		logger.info('Catalogue has been read in.')

		return


### the get methods for the "Critical User Inputs"
	def __getOutputLocation(self, config):
		try:
			if uf.checkkeys(config, 'OutputLocation'):
				self.outloc = config['OutputLocation']
			else:
				self.outloc = input('Enter the full location to where you would like the output saved: ')
			if self.outloc[-1] != '/':
				self.outloc +='/'
				uf.checkpath(self.outloc)
			else:
				uf.checkpath(self.outloc)
			logger.info('Output location: %s'%self.outloc)
		except KeyboardInterrupt:
			uf.exit(self)
		except SystemExit:
			uf.exit(self)
		except:
			print ('There was a problem with entering the output location. Please try again.\n')
			self.__getOutputLocation(config=False, data=None)



	def __getMaxRedshift(self, config):
		try:
			if uf.checkkeys(config, 'z_max'):
				self.max_redshift = config["z_max"]
			else:
				self.max_redshift = input('\nPlease enter the maximum redshift value of your sample: ')
		except KeyboardInterrupt:
			uf.earlyexit(self)
		except SystemExit:
			uf.exit(self)
		except:
			logger.critical("There was an error with the maximum redshift value.", exc_info=True)
			uf.earlyexit(self)

	def __getMinRedshift(self, config):
		try:
			if uf.checkkeys(config, 'z_min'):
				self.min_redshift = config["z_min"]
			else:
				self.min_redshift = input('\nPlease enter the minimum redshift value of your sample: ')
		except KeyboardInterrupt:
			raise KeyboardInterrupt
		except SystemExit:
			uf.exit(self)
		except:
			logger.critical("There was an error with the minimum redshift value.", exc_info=True)
			uf.earlyexit(self)
		return


	def __getSpectralUnits(self, config):
		try: 
			if uf.checkkeys(config, 'SpectralAxisUnit') and uf.checkkeys(config, 'VelocityType'):
				choice = config['SpectralAxisUnit']
				self.veltype = config['VelocityType']
			else:
				print ("\nWhat are the units of spectral axis of your spectra?\n\t1. Hertz (Hz)\n\t2. Kilo Hertz (kHz)\n\t3. Mega Hertz (MHz)\n\t4. Metres/Second (m/s)\n\t5. Kilometres/Second (km/s)\n\t6. Radio velocity (km/s)")
				choice = input('Please enter the number of the applicable units: ')
				if int(choice) in [4,5]:
					self.veltype = input('Please enter the velocity type [optical/radio]: ').lower()
			options = [astun.Hz, astun.kHz, astun.MHz, astun.m/astun.s, astun.km/astun.s]
			self.spectralunit = options[int(choice)-1]

		except KeyboardInterrupt:
			raise KeyboardInterrupt
		except SystemExit:
			uf.exit(self)
		except:
			logger.error('There was an input error with your unit selection.', exc_info=True)
			uf.earlyexit(self)
		return

	def __getFluxUnits(self, config):
		try:
			options = [astun.Jy, astun.mJy, astun.uJy]
			if uf.checkkeys(config, 'SpectrumFluxUnit'):
				choice = config['SpectrumFluxUnit']
			else:
				print ("\nWhat are the flux denisty units of your spectra?\n\t1. Jansky (Jy)\n\t2. Milli Jansky (mJy)\n\t3. Micro Jansky (uJy)\n\t4. Jansky/beam (Jy/beam)\n\t5. Milli Jansky/beam (mJy/beam)\n\t6. Micro Jansky/beam (uJy/beam)")
				choice = input('Please enter the number of the applicable units: ')
			if type(choice) != int:
				print ('Please only select one unit.')
				self.__getFluxUnits(None)
			elif choice in [1,2,3]:
				self.fluxunit = options[int(choice-1)]
				self.convfactor = 1
			else:
				self.fluxunit = options[int(choice-4)]
				if uf.checkkeys(config, 'BeamSize') and uf.checkkeys(config, 'PixelSize'):
					pxsize = config['PixelSize']
					beamsize = config['BeamSize']
				else:
					pxsize = input('Please enter pixel size in arcseconds: ')
					beamsize = input('Please enter fwhm of beam (in arcseconds): ')
				self.convfactor = ( 2.*np.pi*beamsize*beamsize/( 2.*np.sqrt(2.*np.log(2) ) )**2  )/pxsize**2
		except KeyboardInterrupt:
			raise KeyboardInterrupt
		except SystemExit:
			uf.exit(self)
		except:
			logger.error('There was an input error with your unit selection', exc_info=True)
			uf.earlyexit(self)
		return

	def __getStackUnits(self, config):
		try:
			if uf.checkkeys(config, 'StackFluxUnit'):
				choice = config['StackFluxUnit']
			else:
				print("\nWhat units do you want your final stacked profile to have?\n\t1. Jansky (Jy)\n\t2. Solar Masses (Msun)\n\t3. Gas Fraction (Msun/Msun)\n\t4. Solar Masses (Msun), stack in Jy [recommend for clusters]")
				choice = eval(input('Please enter the number of the applicable units: '))
				print(type(choice))
			if (type(choice) != int) :
				print ('Please only select one unit.')
				self.__getStackUnits(None)
			else:
				options = [astun.Jy, uf.msun, uf.gasfrac, astun.Jy]
				self.stackunit = options[int(choice-1)]
				if choice == 4:
					self.cluster = True
				else:
					pass
		except KeyboardInterrupt:
			raise uf.exit(self)
		except SystemExit:
			raise uf.exit(self)
		except:
			logger.error('There was an input error with your unit selection.', exc_info=True)
			uf.earlyexit(self)
		return


	def __getChannelWidth(self, config):
		try:
			if uf.checkkeys(config, 'ChannelWidth'):
				self.channelwidth = config['ChannelWidth'] * self.spectralunit
			else:
				self.channelwidth = input('Please enter the channel width (in the same units as the spectral axis): ') * self.spectralunit
		except KeyboardInterrupt:
			raise uf.exit(self)
		except SystemExit:
			uf.exit(self)
		except:
			logger.error('There was an input error with your channel width.', exc_info=True)
			uf.earlyexit(self)

	def __getSpectraLocation(self, config):
		try:
			if uf.checkkeys(config, 'SpectrumLocation'):
				self.specloc = config['SpectrumLocation']
			else:
				self.specloc = input('Please enter the full path of the spectra location: ')
			if self.specloc[-1] != '/':
				self.specloc +='/'
			else:
				h = 0
			if not os.path.exists(self.specloc):
				print ('\nThe spectrum location you have entered does not exist. Please try again.\n')
				self.__getSpectraLocation(config=None)
			else:
				print ('\nThe spectrum location exists :).\n')
		except KeyboardInterrupt:
			raise uf.exit(self)
		except SystemExit:
			uf.exit(self)
		except:
			logger.error('There was an issue with the entered spectrum folder path.', exc_info=True)
			uf.earlyexit(self)

	def __getSpectrumRowStart(self, config):
		try:
			if uf.checkkeys(config, 'FirstRowInSpectrum'):
				self.rowstart = int(config['FirstRowInSpectrum'])
			else:
				self.rowstart = input('Please enter the row number corresponding to the start of the data in the spectrum files (this number should be the same for every spectrum file).\n(Please note: row numbering starts at 1 - the first row of information is row 1): ')
				self.rowstart = int(self.rowstart)
		except KeyboardInterrupt:
			raise uf.exit(self)
		except SystemExit:
			uf.exit(self)
		except:
			logger.error('There was an error with the number of the first row in the spectrum files.', exc_info=True)
			uf.earlyexit(self)

	def __getSpectrumDelimiter(self, config):
		try: 
			if uf.checkkeys(config, 'RowDelimiter'): 
				self.rowdelim = config['RowDelimiter']
			else:
				self.rowdelim = input('Please enter the delimiter used in the spectra files (this must be the same for all spectra): ')
		except KeyboardInterrupt:
			raise uf.exit(self)
		except SystemExit:
			uf.exit(self)
		except:
			logger.error('There was an error with the row delimiter.', exc_info=True)
			uf.earlyexit(self)

	def __getSpectrumColumns(self, config):
		try:
			if uf.checkkeys(config, 'SpectrumColumns'):
				self.speccol = config['SpectrumColumns']
			else:
				self.speccol = input("Please enter the column numbers of the spectral axis and flux (the column number of the first column is 0).\nThe numbers should be separated by a ',' e.g. 0,1: ")
			if len(self.speccol) != 2:
				print ('You need to enter two column numbers. Please try again.\n')
				self.__getSpectrumColumns(config=False, data=None)
		except KeyboardInterrupt:
			raise uf.exit(self)
		except SystemExit():
			raise uf.exit(self)
		except:
			logger.error('There was an error with the spectrum column numbers.', exc_info=True)
			uf.earlyexit(self)


### the get methods for the "Optional User Inputs"
	def __getCosmology(self, config):
		try:
			if uf.checkkeys(config, 'H0') and uf.checkkeys(config, 'Om0'):
				H0 = config['H0']
				Om0 = config['Om0']
			else:
				print ('\nWe assume a Flat Lambda CDM universe')
				H0 = input('Please enter the Hubble constant (H0) in units of km/s/Mpc: ')
				Om0 = input('Please enter the Omega matter (Om0): ')
			self.cosmology = astcos.FlatLambdaCDM(H0, Om0)
			logger.info('Cosmology: H0 = %.1f %s, Om0 = %.3f'%(self.cosmology.H0.value, self.cosmology.H0.unit.to_string(), self.cosmology.Om0))
		except KeyboardInterrupt:
			raise uf.exit(self)
		except SystemExit:
			raise uf.exit(self)
		except:
			logger.error('There was a problem with entering the cosmology.', exc_info=True)
			uf.earlyexit(self)
		return


	# def __getRestFrequency(self, config):
	# 	try:
	# 		restfreq = input('Enter the rest frequency of the emission line in MHz: ') * astun.MHz
	# 		self.femit = restfreq
	# 	except KeyboardInterrupt:
	# 		raise KeyboardInterrupt
	# 	except SystemExit:
	# 		uf.exit(self)
	# 	except:
	# 		logger.error('Rest frequency not interpreted.', exc_info=True)
	# 		uf.earlyexit(self)


	def __getMaxGalaxyWidth(self, config):
		try:
			if uf.checkkeys(config, 'GalaxyWidth'):
				self.maxgalw = config['GalaxyWidth'] * astun.km/astun.s
			else:
				self.maxgalw = input('Enter maximum galaxy width in km/s: ') * astun.km/astun.s
			logger.info('Maximum galaxy width = %i %s'%(self.maxgalw.value, self.maxgalw.unit.to_string()))
		except KeyboardInterrupt:
			raise uf.exit(self)
		except SystemExit:
			raise uf.exit(self)
		except:
			logger.error("Did not understand the maximum galaxy width.", exc_info=True)
			uf.earlyexit(self)

	def __getWeightingScheme(self, config):
		try:
			if uf.checkkeys(config, 'WeightOption'):
				self.weighting = str(config['WeightOption'])
			else:
				print ('\nThe following are options can be used to weight each spectrum:\n\t1. w = 1. [default]\n\t2. w = 1/rms \n\t3. w = 1/rms^2 \n\t4. w = dl^2/rms^2')
				## TODO: add in option for individual weights from catalogue as well as custom weighting function
				weights = input('Please enter the number of the scheme you would like to use: ')
				self.weighting = weights[0]
			wghts = {'1': 'w = 1', '2': 'w = 1/rms', '3': 'w = 1/rms^2', '4': 'w = dl^2/rms^2'}
			logger.info('Weighting factor: %s'%wghts[self.weighting])
		except KeyboardInterrupt:
			raise uf.exit(self)
		except SystemExit:
			raise uf.exit(self)
		except:
			logger.error('Could not identify the weighting scheme.', exc_info=True)
			uf.earlyexit(self)

	def __getMaxLenSpectrum(self, config):
		try:
			if uf.checkkeys(config, 'StackedSpectrumLength'):
				maxstackw = config['StackedSpectrumLength'] * astun.km/astun.s
			else:
				maxstackw = input('Please enter the maximum length of the stacked spectrum in km/s: ') * astun.km/astun.s
			if maxstackw < 2.5*self.maxgalw:
				print ('The stacked spectrum length is too short - we will not be able to have very good noise calculations, etc. Please try again.\n')
				self.__getMaxLenSpectrum(config=None)
			else:
				self.maxstackw = maxstackw
			logger.info('Length of stacked spectrum: %i %s'%(self.maxstackw.value, self.maxstackw.unit.to_string()))
		except (KeyboardInterrupt, SystemExit):
			uf.exit(self)
		except: 
			logger.error('Could not determine the maximum length of the stacked spectrum.', exc_info=True)
			uf.earlyexit(self)


	def __getUncertAn(self, config):
		try:
			if uf.checkkeys(config, 'UncertaintyYN'):
				self.uncert = config['UncertaintyYN']
			else:
				self.uncert = input('Should I calculate uncertainties? ')

			if self.uncert.lower() == 'n':
				self.mc = 1
			else:
				self.mc = self.mc
		except KeyboardInterrupt:
			raise uf.exit(self)
		except SystemExit:
			raise uf.exit(self)
		except: 
			logger.error('There was a problem determining whether or not to perform an uncertainty analysis.')
			uf.earlyexit(self)


	def __getUncertType(self, config):
		try:
			if uf.checkkeys(config, 'UncertaintyMethod'):
				self.uncerttype = config['UncertaintyMethod']
			else:
				untyp = input('Which method of redshift uncertainties should I use? [dagjk/redshift] ')
				uncertop = { 'r': 'redshift', 'd': 'dagjk' }
				self.uncerttype = uncertop[(untyp.lower())[0]]
		except KeyboardInterrupt:
			raise uf.exit(self)
		except SystemExit:
			raise uf.exit(self)
		except: 
			logger.error('There was a problem determining whether or not to perform an uncertainty analysis', exc_info=True)
			uf.earlyexit(self)

	## calculating the private variables: 
	def __calcPrivate(self):
		self.mean = 0.
		if 'Hz' in self.spectralunit.to_string():
			self.restdv = uf.calc_dv_from_df(self.channelwidth, self.medianz, self.femit)
		elif 'm / s' in self.spectralunit.to_string():
			self.restdv = self.channelwidth
		self.exspeclen = int((self.maxstackw/self.restdv).value) + (int((self.maxstackw/self.restdv).value)+1)%2
		self.spectralaxis = np.arange(self.mean-(self.exspeclen/2)*self.restdv.value, self.mean+(self.exspeclen/2)*self.restdv.value+self.restdv.value, self.restdv.value)*self.restdv.unit
		self.maskstart = np.argmin(abs(self.spectralaxis.value - self.mean + self.maxgalw.value/2.))
		self.maskend = np.argmin(abs(self.spectralaxis.value - self.mean - self.maxgalw.value/2.)) + 1
		self.exspeclen += (len(self.spectralaxis) - self.exspeclen)
		self.mask = int(((self.maxgalw)/self.restdv).value) + int(((self.maxgalw)/self.restdv).value+1)%2
		self.stackmask = int(((self.maxgalw)/self.restdv).value) + int(((self.maxgalw)/self.restdv).value+1)%2
		self.noiselen = self.exspeclen - self.mask
		self.keys = self.catalogue['Object ID'].data
		self.numcatspec = len(self.keys)
		self.mediandistance = self.cosmology.luminosity_distance(self.medianz)
		randinds = [np.random.choice(len(self.catalogue), len(self.catalogue), replace=False) for j in range(self.mc+1)]
		self.randz = np.array( [ self.catalogue['Redshift'][randinds[i]] for i in range(self.mc+1)] )
		if True in self.catalogue['Redshift Uncertainty'].mask:
			self.randuz = 0.0002*np.random.randn(len(self.catalogue), self.mc)
		else:
			unc = self.catalogue['Redshift Uncertainty'].data.reshape(len(self.catalogue),1)
			self.randuz = np.multiply(unc, np.random.randn(len(self.catalogue), self.mc))

		logger.info('Median redshift: %f'%self.medianz)
		logger.info('Rest channel width: %.3g %s'%(self.restdv.value, self.restdv.unit.to_string()))
		logger.info('Length of spectrum mask: %i'%self.mask)
		logger.info('Number of channels in stacked spectrum: %i'%self.exspeclen)
		logger.info('Number of catalogue entries: %i'%self.numcatspec)
		return


	def __checkConfig(self, config):
		if uf.checkkeys(config, 'IntegrateGalaxyWidth'):
			intwid = True
			self.maxgalw = config['IntegrateGalaxyWidth']*astun.km/astun.s
		else: 
			intwid = False

		if uf.checkkeys(config, 'SmoothStatus'):
			self.rebinstatus = config['SmoothStatus']
			rebin = True
		else: 
			rebin = False

		if uf.checkkeys(config, 'SmoothType'):
			self.smoothtype = config['SmoothType']
		else: pass

		if uf.checkkeys(config, 'SmoothWindow'):
			self.smoothwin = config['SmoothWindow']
		else: pass

		if uf.checkkeys(config, 'FittedFunctionNumbers'):
			self.funcnum = config['FittedFunctionNumbers']
		else: pass

		if uf.checkkeys(config, 'AnalysisOptions'):
			self.optnum = config['AnalysisOptions']
		else: pass

		if (intwid == True) and (rebin == True):
			self.config = True
		else:
			self.config = False

		return


	def __callOptionalUser(self, config):

		self.__getCosmology(config)
		self.__getMaxGalaxyWidth(config)
		self.__getMaxLenSpectrum(config)
		self.__getWeightingScheme(config)
		
		return


	def __callCriticalUser(self, config):
		self.__getOutputLocation(config)
		self.__getStackUnits(config)
		self.__getUncertAn(config)
		if self.uncert.lower() == 'y':
			self.__getUncertType(config)
		else: pass
		logger.info("Stacked spectrum units: %s"%(self.stackunit.to_string()))
		logger.info("Uncertainty calculation type: %s"%self.uncerttype)

		while bool(self.catalogue) == False:
			self.__get_astropy_catalogue(config)

		while not self.min_redshift < self.max_redshift:
			self.__getMaxRedshift(config)
			self.__getMinRedshift(config)

		self.__getFluxUnits(config)
		self.__getSpectralUnits(config)
		self.__getChannelWidth(config)
		logger.info('Spectrum flux units: %s'%self.fluxunit.to_string())
		logger.info('Spectrum axis units: %s'%self.spectralunit.to_string())
		logger.info('Spectrum channel width: %.3g %s'%(self.channelwidth.value, self.channelwidth.unit.to_string()))

		if config is None:
			print ('\nThe following settings relate to the spectra to be stacked.')
		else:
			pass
		self.__getSpectraLocation(config)
		self.__getSpectrumRowStart(config)
		self.__getSpectrumColumns(config)
		self.__getSpectrumDelimiter(config)


	def runInput(self, config):

		try:
			self.__callCriticalUser(config)
			self.__callOptionalUser(config)
			self.__calcPrivate()
			self.__checkConfig(config)
		except (SystemExit, KeyboardInterrupt):
			uf.exit(self)
		except:
			logger.error('Encountered an exception:', exc_info=True)
			uf.earlyexit(self)


	