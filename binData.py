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
from inputData import inputCatalogue
import decimal
import logging

logger = logging.getLogger(__name__)



class binnedCatalogue(inputCatalogue):

	def __init__(self, other):
		self.binnedtable = None
		self.smoothalgorithms = other.smoothalgorithms
		self.binpartindicies = None
		self.fullcatalogue = other.catalogue
		self.catalogue = None
		self.origruntime = other.runtime
		self.fullranduz = other.randuz
		self.randuz = None
		self.randz = other.randz
		self.keys = other.keys
		self.catfilename = other.catfilename
		self.numcatspec = None
		self.max_redshift = other.max_redshift
		self.min_redshift = other.min_redshift
		self.cosmology = other.cosmology
		self.spectralunit = other.spectralunit
		self.veltype = other.veltype
		self.fluxunit = other.fluxunit
		self.stackunit = other.stackunit
		self.channelwidth = other.channelwidth
		self.femit = other.femit
		self.specloc = other.specloc
		self.rowstart = other.rowstart
		self.rowdelim = other.rowdelim
		self.speccol = other.speccol
		self.noiselen = other.noiselen
		self.origmask  = other.mask
		self.mask = None
		self.exspeclen = other.exspeclen
		self.spectralaxis = other.spectralaxis
		self.outloc = other.outloc
		self.origmaxgalw = other.maxgalw
		self.maxgalw = None
		self.runtime = None
		self.weighting = other.weighting
		self.maxstackw = []
		self.convfactor = other.convfactor
		self.uncert = other.uncert
		self.mc = other.mc
		self.optnum = []
		self.funcnum = []
		self.rbnum = []
		self.linetype = other.linetype	
		self.outcatalogue = other.outcatalogue
		self.origrebinstatus = False
		self.rebinstatus = False
		self.smoothtype = other.smoothtype
		self.smoothwin = other.smoothwin
		self.maskstart = None
		self.maskend = None
		self.bins = None
		self.mean = other.mean
		self.fullmaxgalw = []
		self.fullmask = []
		self.fullrebinstatus = []
		self.fullmaskstart = []
		self.fullmaskend = []
		self.fullsmoothtype = []
		self.fullsmoothwin = []
		self.fullfuncnum = []
		self.fullrbnum = []
		self.fulloptnum = []
		self.fullw50 = []
		self.w50 = other.w50
		self.suppress = other.suppress
		self.clean = other.clean
		self.progress = other.progress
		self.saveprogress = other.saveprogress
		self.multiop = other.multiop
		self.latex = other.latex
		self.uncerttype = other.uncerttype
		self.R = 0.75
		self.stackmask = other.stackmask
		self.restdv = other.restdv
		self.mediandistance = other.mediandistance
		self.avemass = other.avemass
		self.avesm = other.avesm
		self.medianz = other.medianz
		self.cluster = other.cluster
		self.clusterZ = other.clusterZ
		self.clusterDL = other.clusterDL
		self.config = other.config
		



	def determineBins(self, config):

		if self.config == True:
			self.fullfuncnum = config['FittedFunctionNumbers']
			self.fulloptnum = config['AnalysisOptions']
			self.fullsmoothtype = config['SmoothType']
			self.fullsmoothwin = config['SmoothWindow']
			self.fullmaxgalw = config['IntegrateGalaxyWidth']
			self.fullrebinstatus = config['SmoothStatus']
		else:
			pass

		if uf.checkkeys(config, 'BinInfo') and uf.checkkeys(config, 'Bins'):
			colname, binwidth, binstart, binend = config["BinInfo"]
			self.bins = config['Bins']			
			self.binnedtable, self.binpartindicies = self.__bintable(self.fullcatalogue, colname, binwidth, binstart, binend)
			return
		else:
			print ('\n%s\t%s\t%s\t%s\t%s\n'%(self.fullcatalogue.colnames[0], self.fullcatalogue.colnames[1], self.fullcatalogue.colnames[2], self.fullcatalogue.colnames[3], self.fullcatalogue.colnames[4]))
			colname = raw_input('Please enter the exact Column name for the column containing the data to be binned: ')
			binwidth = input('Please enter the bin width (in same units as the data to be binned): ')
			binstart = input('Please enter the start of the bin range (in same units as the data to be binned): ')
			binend = input('Please enter the end of the bin range (in same units as the data to be binned): ')
			self.binnedtable, self.binpartindicies = self.__bintable(self.fullcatalogue, colname, binwidth, binstart, binend)
			return

	def __bintable(self, tableorig, colnam, binwidth, binrangestart, binrangeend):
		table = tableorig
		# print(table)
		if self.bins is None or self.bins == 'none':
			bins = np.arange( binrangestart, binrangeend+binwidth, binwidth )
		else:
			bins = self.bins
		if colnam == 'Stellar Mass':
			coldata = table[colnam]
			# coldata = (table[colnam].data*table[colnam].unit).to(uf.msundex, uf.log10conv)
		else:
			coldata = table[colnam].data
		bindata = np.digitize(coldata, bins, right=True)
		newcol = asttab.Column(name='Bin Number', data=bindata)
		if 'Bin Number' in table.colnames:
			table.remove_column('Bin Number')
			table.add_column(newcol)
		else:
			table.add_column(newcol)
		table = table.group_by('Bin Number')

		if 'quantity' in str(type(coldata)):
			columndata = coldata.value
		else:
			columndata = coldata

		if 0 in bindata:
			if np.max(columndata) > binrangeend:
				bindices = table.groups.indices[1:-1]
			else:
				bindices = table.groups.indices[1:]
		else:
			if np.max(columndata) > binrangeend:
				bindices = table.groups.indices[:-1]
			elif len(table.groups.indices) > (len(bins)+1):
				bindices = table.groups.indices[:-1]
			else:
				bindices = table.groups.indices

		return table, bindices


	def createCatalogue(self, binindices, table, n, counter):
		
		if (counter == 0) and (self.config == True):

			self.optnum = self.fulloptnum[n]
			self.maxgalw = self.fullmaxgalw[n]*astun.km/astun.s
			self.rebinstatus = self.fullrebinstatus[n]

			if len(self.fullfuncnum) == 0:
				self.funcnum =[]
			else:
				if self.optnum == ["4"]:
					pass
				else:
					self.funcnum = self.fullfuncnum[n]

			if self.rebinstatus is True:
				self.smoothtype = self.fullsmoothtype[n]
				self.smoothwin = self.fullsmoothwin[n]
			else:
				pass

		else: pass

		self.catalogue = None
		self.numcatspec = None
		self.runtime = None
		cat = table[binindices[n]:binindices[n+1]]
		# print(cat)
		self.randuz = self.fullranduz[binindices[n]:binindices[n+1]]
		self.runtime = self.origruntime + '_bin%i'%table['Bin Number'][binindices[n]]
		self.catalogue = cat
		self.numcatspec = len(cat)
		
		if counter > 0:
			self.rebinstatus = self.fullrebinstatus[n]
			self.rbnum = self.fullrbnum[n]
			self.optnum = self.fulloptnum[n]
			if self.optnum == ["4"]:
				pass
			else:
				self.funcnum = self.fullfuncnum[n]
				
			if self.rebinstatus is True:
				self.smoothtype = self.fullsmoothtype[n]
				self.smoothwin = self.fullsmoothwin[n]
			else: pass

			self.maskstart = self.fullmaskstart[n]
			self.maskend = self.fullmaskend[n]
			self.maxgalw = self.fullmaxgalw[n]*self.spectralunit
			self.mask = self.fullmask[n]
			self.w50 = self.fullw50[n]

		else:
			if self.config == False:
				self.maxgalw = self.origmaxgalw
				self.mask = self.origmask
				self.rebinstatus = self.origrebinstatus
				self.optnum = []
			else:
				self.mask = self.origmask

		self.maskstart = np.argmin(abs(self.spectralaxis.value - self.mean + self.maxgalw.value/2.))
		self.maskend = np.argmin(abs(self.spectralaxis.value - self.mean - self.maxgalw.value/2.)) + 1
		logger.info('Created catalogue for bin %i'%(n+1))
		# print(self.catalogue)
		return 
	

	def __add__(self, other):
		if other.config == False:
			self.fullrbnum.append(other.rbnum)
			self.fullfuncnum.append(other.funcnum)
			self.fulloptnum.append(other.optnum)
			self.fullsmoothtype.append(other.smoothtype)
			self.fullsmoothwin.append(other.smoothwin)
			self.fullmaskstart.append(other.maskstart)
			self.fullmaskend.append(other.maskend)
			self.fullmask.append(other.mask)
			self.fullmaxgalw.append(other.maxgalw.value)
			self.fullrebinstatus.append(other.rebinstatus)
			self.fullw50.append(other.w50)
		else: 
			self.fullrbnum.append(other.rbnum)
			self.fullmaskstart.append(other.maskstart)
			self.fullmaskend.append(other.maskend)
			self.fullmask.append(other.mask)
			self.fullw50.append(other.w50)
		return self

	def __radd__(self, other):
		logger.info('Updated full catalogue')
		return self.__add__(other)


