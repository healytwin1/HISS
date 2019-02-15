import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as astfit
import astropy.io.ascii as astasc
import astropy.table as asttab
import useful_functions as uf
import astropy.modeling as astmod
import astropy.units as astun
import matplotlib.gridspec as gridspec
from scipy.special import gammainc, erfinv, gammaincc
from scipy.stats import chisquare
from analyseData import anaData
import logging

logger = logging.getLogger(__name__)


class dispData(anaData):

	def __init__(self, cat):
		anaData.__init__(self)
		self.datafilename = cat.outloc+'OutputData_'+cat.runtime+'.FITS'
		self.massfilename = cat.outloc+'IntegratedFlux_'+cat.runtime+'.csv'
		self.stackrms = None
		self.spec = None
		self.specuncert = 0.
		self.refspec = 0.
		self.spectralaxis = None
		self.nobj = 0
		self.totalmass = 0.
		self.averms1 = 0
		self.averms2 = 0
		self.averagenoise = 0
		self.fitparam = None
		self.massdata = None
		self.specunit = None
		self.stackunit = cat.stackunit
		self.spectralunit = cat.spectralunit.to_string()
		self.fittedfunctions = None
		self.paramtable = None
		self.noiseuncert = None
		self.optnum = cat.optnum
		self.funcnum = cat.funcnum
		self.rbnum = cat.rbnum
		self.mask = cat.mask
		self.maxgalw = cat.maxgalw
		self.mean = cat.mean
		self.avemass = (cat.avemass)


	def __read_in_data(self, cat):
		logger.info('Reading in spectrum data for display.')
		dHDU = astfit.open(self.datafilename)
		header = dHDU[0].header
		self.nobj = header['N_OBJ']
		self.totalmass = header['TFlux']
		spectable = dHDU[1].data
		self.paramtable = dHDU[3]
		self.spectralaxis = spectable['Spectral Axis']
		if cat.clean == 'clean':
			self.refspec = 0.
		else:
			noisetable = dHDU[2].data
			self.stackrms = noisetable['Stacked RMS Noise'][np.where(noisetable['Stacked RMS Noise'] != 0.)[0]]
			self.averms1, self.averms2 = header['AveRMS_1'], header['AveRMS_2']
			self.averagenoise = header['AveSig']
			self.refspec = spectable['Reference Spectrum']
		self.spec = spectable['Stacked Spectrum']
		self.fittedfunctions = [col for col in self.paramtable.columns.names if 'Uncertainty' not in col]
		if cat.uncert == 'y':
			self.specuncert = spectable['Stacked Spectrum Uncertainty']
			self.noiseuncert = noisetable['Stacked RMS Noise Uncertainty']
		return


	def __readTextData(self):
		logger.info('Reading in integrated flux data for display.')
		self.massdata = asttab.Table.read(self.massfilename, format='ascii.ecsv')
		return


	def printTable(self, table):
		formattedtable = table.pformat(show_name=True, show_unit=True)
		return formattedtable
	

	def __getstatistics(self, spec, specuncert, spectral, cat):
		#########################################
		## get the stats
		#########################################
		if (type(specuncert) == float) or (type(specuncert) == int) or (type(specuncert) is None):
			sigma = np.ones(len(spec))
			noise = (np.percentile( np.array( list(spec[:cat.maskstart])+list(spec[cat.maskend:] ) ), 75) - np.percentile( np.array( list(spec[:cat.maskstart])+list(spec[cat.maskend:] ) ), 25))/(2*0.67)
		else:
			sigma = specuncert
			noise = (np.percentile( np.array( list(spec[:cat.maskstart])+list(spec[cat.maskend:] ) ), 75) - np.percentile( np.array( list(spec[:cat.maskstart])+list(spec[cat.maskend:] ) ), 25))/(2*0.67)

		sgauss, cat = self.fit_one_gauss(spec, spectral, cat, sigma)
		standard_snr, alfalfa_snr, integrated_snr, cat = self.calcSignalNoise(spec, spectral, sgauss, cat, noise)
		pval, sig = self.calcPvalue(sgauss, spec, sigma)
		#########################################
		return pval, sig, standard_snr, alfalfa_snr, integrated_snr, cat


	def __printStatistics(self, uncert, cat, spec, specuncert, spectralaxis):
		if type(self.avemass) != np.float_:
			self.avemass = (self.avemass).value
		else:
			pass
		self.massdata['Integrated Flux'].format = '7.2g'
		if '4' in cat.optnum or '4.' in cat.optnum:
			h=0
		else:
			self.massdata['Fit RMS'].format = '7.2g'
			self.massdata['Fit ChiSquare'].format = '7.2g'

		if uncert == True:
			self.massdata['Uncertainty Integrated Flux'].format = '7.2g'
			tl = 110
		else:
			tl = 100
		unit = self.massdata['Integrated Flux'].unit
		masstable = self.printTable(self.massdata)
		if (cat.stackunit == astun.Jy) and (cat.cluster == False):
			fd = 'flux density'
		elif (cat.stackunit == uf.msun) or (cat.cluster == True):
			fd = 'HI mass'
		else:
			fd = 'gas fraction '
		t = 'Total %s of sample = %.2g %s'%(fd, self.totalmass, unit)

		am = 'Average HI mass of sample = %.2g %s'%(self.avemass, 'Msun')
		n = 'Stacked N = %i profiles'%self.nobj
		masstable = self.printTable(self.massdata)

		pval, sig, standard_snr, alfalfa_snr, integrated_snr, cat = self.__getstatistics(spec, specuncert, spectralaxis, cat)

		snr = r'The peak signal-to-noise ratio: %5.2g'%standard_snr
		if sig == 8.2:
			ex = r'>'
		else:
			ex =''
		p = 'The significance of the peak: %s%3.2g sig (p-value = %.2g)'%(ex, sig, pval)
		sn = 'Peak signal to noise: %.1f'%standard_snr
		asn = 'ALFALFA signal to noise: %.1f'%alfalfa_snr
		isn = 'Integrated signal to noise: %.1f'%integrated_snr

		line0 = '\n   |'+'-'*tl+'|'
		line1 = '   |'+' '*tl+'|'
		line2 = '   |'+' '*5+n+' '*(tl-5-len(n))+'|'
		line3 = '   |'+' '*5+am+' '*(tl-5-len(am))+'|'
		line4 = '   |'+' '*5+t+' '*(tl-5-len(t))+'|'
		line41 = '   |'+' '*5+p+' '*(tl-5-len(p))+'|'
		line42 = '   |'+' '*5+sn+' '*(tl-5-len(sn))+'|'
		line43 = '   |'+' '*5+asn+' '*(tl-5-len(asn))+'|'
		line44 = '   |'+' '*5+isn+' '*(tl-5-len(isn))+'|'
		line5 = '   |'+' '*tl+'|'
		linet = ''
		for i in range(len(masstable)):
			linet += '   |'+'   '+masstable[i]+' '*(tl-3-len(masstable[i]))+'|' + '\n'
		line6 = '   |'+' '*tl+'|'
		line7 = '   |'+'-'*tl+'|'

		displaytext = line0 + '\n' + line1 + '\n' + line2 + '\n' + line3 + '\n' + line4 + '\n'  + line41 + '\n'  +line42 + '\n'  +line43 + '\n'  +line44 + '\n'  + line5 + '\n' + linet + line6 + '\n' + line7 + '\n' 
		
		logger.info('Final results:'+displaytext)
		if cat.suppress == 'hide':
			print (displaytext)
		else:
			h = 0		
		return




	def __displayplots(self, cat, uncert, allflux=None, spec=None, spectral=None, refspec=0., specuncert=0., extra=None):
		
		plt.close('all')
		fig = plt.figure(figsize=( 12,8.5 ))
		gs = gridspec.GridSpec(9,4)
		ax1 = fig.add_subplot(gs[:5,:2]) ## stacked spectrum
		ax2 = fig.add_subplot(gs[:5,2:]) ## stacked noise
		ax3 = fig.add_subplot(gs[5:,:]) ## bottom panel, containing tabled data


		if (astun.Jy == cat.stackunit) and (cat.cluster == False):
			if 1E-3 < max(spec) < 1E-1:
				conv = 1E3
				unitstr = r'Average Stacked Flux Density ($\mathrm{mJy}$)'
			elif max(spec) < 1E-3:
				conv = 1E6
				unitstr = r'Average Stacked Flux Density ($\mathrm{\mu Jy}$)'
			else:
				conv = 1.
				unitstr = 'Average Stacked Flux Density ($\mathrm{Jy}$)'
			t = r'Total integrated flux of sample = %5.2g %s'%(self.totalmass, 'Jy km/s')
		elif (uf.msun == cat.stackunit) or (cat.cluster == True):
			masstext = '%.3E'%np.max(spec)
			masi = masstext.split('E')[0]
			masexp = masstext.split('E')[1]
			conv = 1./eval('1E%s'%masexp)
			unitstr = r'Average Stacked Mass ($10^{%i} \, \mathrm{M_\odot/chan}$)'%float(masexp)
			t = r'Total HI mass of sample = %s %s'%(uf.latexexponent(self.totalmass), r'M$_\odot$')
		elif uf.gasfrac == cat.stackunit:
			conv = 1.
			unitstr = r'Average Stacked Gas Fraction (M$_\mathrm{HI}$/M$_\ast$)'
			t = ''
		else:
			conv = 1.
			unitstr = 'Average Stacked Quantity'
		

		if cat.clean == 'clean':
			ax2.annotate(r'-----There is no noise analysis.-----', xy=(0.5, 0.5), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center')
		else:
			X = range(1,len(self.stackrms)+1,1)
			ax2.set_xlabel(r'Number of stacked spectra')
			ax2.errorbar(X, self.stackrms, yerr=(self.noiseuncert), ls='', ecolor='k', mec='gray', mfc='gray', marker='o', ms=4)
			ax2.plot(X, self.averagenoise/np.sqrt(X),'g-')
			ax2.set_ylabel(r'Stacked Average Noise ($\mathrm{Jy}$)')
			ax2.set_xscale('log')
			ax2.set_yscale('log')
			ax2.set_xlim(0.9, self.nobj+0.1*self.nobj)
			ax2.set_ylim(0.9*min(self.stackrms),1.2*max(self.stackrms))
			ax1.plot(spectral, refspec*conv, color='grey',ls='-', marker='.', label='Reference Spectrum')
			
		## stacked spectrum
		x = np.linspace(spectral[0],spectral[-1],10000)
		ax1.set_ylabel(unitstr)
		ax1.set_xlabel(r'Relative Velocity (km/s)')
		ax1.set_xlim([spectral[0],spectral[-1]])
		xx = ax1.get_xticks()
		ll = ["%5.1f" % a for a in xx]
		ax1.set_xticks(xx)
		ax1.set_xticklabels(ll)
		ax1.set_xlim([spectral[0],spectral[-1]])
		ax1.axhline(0, color='#2E2E2E', ls='--')
		ax1.errorbar(spectral, spec*conv, yerr=specuncert*conv, ecolor='k', ls='-', color='k', marker='.', label='Stacked Spectrum')
		if '4' in cat.optnum or '4.' in cat.optnum:			
			h=0
		else:
			for num in self.funcnum:
				fname = self.functions[int(eval(num)-1)]
				param = self.paramtable.data[fname]
				func = self.callFunctions(num, cat, None, x, None, uncert=False, param=param, process='plot')
				ax1.plot(x, func*conv, color=self.fitcolors[fname], label=fname)
		ax1.legend(loc='upper left',fontsize=8, numpoints=1)

		pval, sig, standard_snr, alfalfa_snr, integrated_snr, cat = self.__getstatistics(spec, specuncert, spectral, cat)

		snr = r'The peak signal-to-noise ratio: %5.2g'%standard_snr
		if sig == 8.2:
			ex = r'>'
		else:
			ex =''
		p = r'The significance of the peak: $%s %3.2g\sigma$ (p-value = %s)'%(ex, sig, uf.latexexponent(pval))
		am = 'Average HI mass of sample = %s %s'%(uf.latexexponent(self.avemass), r'M$_\odot$')
		n = r'Number of galaxy profiles included in Stacked Spectrum: %i'%self.nobj

		if cat.latex == 'latex':
			tabstrinner = r'%s \\ %s \\ %s \\ %s \\ %s \\ %s'%(n, am, t, snr, p, r'  ')
			tabstr1 = r'\begin{tabular}{l} %s \end{tabular}'%tabstrinner
		
			tabstr = uf.latextablestr(self.massdata, cat)
			colstr = r"%s"%("|c"*len(self.massdata.colnames)+"|")
			table = r"\vspace{1cm}\begin{tabular}{%s}\hline %s \end{tabular}"%( colstr, tabstr)
			plotstr = tabstr1 + r'\\'+ table

			ax3.text(s=plotstr, x=0.5, y=0.5, horizontalalignment='center', verticalalignment='center', color='k')
		else:
			tabstrinner = '%s \n%s \n%s \n%s \n%s \n%s'%(n, am, t, snr, p, r'  ')
			ax3.text(s=tabstrinner, x=0.5, y=0.98, horizontalalignment='center', verticalalignment='top', color='k', fontsize=12)
			self.massdata.rename_column('Integrated Flux', 'Int. Flux Dens.')
			massdata = uf.convertable(self.massdata)
			thetable = ax3.table(cellText= massdata.as_array(), colLabels=massdata.colnames, loc='lower center', fontsize=12)
			colwidths = tuple([i for i in range(-1, len(massdata.colnames), 1)])
			thetable.auto_set_column_width(colwidths)

		ax3.xaxis.set_visible(False)
		ax3.yaxis.set_visible(False)
		ax3.axis('off')
	
		plt.tight_layout(rect=[0,0,0.95,0.98])
		plt.savefig(cat.outloc+'DisplayWindow_%s.pdf'%cat.runtime, format='pdf', bbox_inches='tight', pad_inches=0.2)

		## Histogram of integrated flux
		if uncert == True:
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
					grid[0][int(n/2.)].set_xlabel(unitstr, fontsize=11)
					grid[0][int(n/2.)].set_ylabel('Number of stacks', fontsize=11)
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
						grid[1][int(n/2.)].set_xlabel(unitstr, fontsize=11)
						grid[1][int(n/2.)].set_ylabel('Number of stacks', fontsize=11)
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
					ax1.set_title(r'Mean of Single Gaussian Fit', fontsize=11)
					ax1.set_xlabel('km/s', fontsize=11)
					ax1.set_ylabel('Number of stacks', fontsize=11)
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
					ax2.set_title(r'RMS of Stacked Spectrum', fontsize=11)
					ax2.set_xlabel(unitstr, fontsize=11)
					ax2.set_ylabel('Number of stacks', fontsize=11)
					ax2.hist(extra[1]*conv, bins=20, color='limegreen')
					ax2.set_xlim((extra[1]*conv).min(), (extra[1]*conv).max())
					xx = ax2.get_xticks()
					ll = ["%.3g" % a for a in xx]
					ax2.set_xticks(xx)
					ax2.set_xticklabels(ll)
					ax2.set_xlim((extra[1]*conv).min(), (extra[1]*conv).max())
				else:
					h = 0

			plt.tight_layout(pad=0.25, rect=[0.01, 0.01, 0.99, 0.99])
		return plt.show()


	def display(self, cat, allflux, extra=None, hide=None):
		if cat.uncert == 'y':
			uncert = True
		else:
			uncert = False
		try:
			self.__read_in_data(cat)
		except (SystemExit, KeyboardInterrupt):
			uf.exit(cat)
		except:
			logger.critical('Encountered an error reading in the FITS data.', exc_info=True)
			uf.earlyexit(cat)

		try:
			self.__readTextData()
		except (SystemExit, KeyboardInterrupt):
			uf.exit(cat)
		except:
			logger.critical('Encountered an error reading in the text data.', exc_info=True)
			uf.earlyexit(cat)
		
		try:
			self.__printStatistics(uncert, cat, self.spec, self.specuncert, self.spectralaxis)
		except (SystemExit, KeyboardInterrupt):
			uf.exit(cat)
		except:
			logger.critical('Encountered an error printing the results.', exc_info=True)
			uf.earlyexit(cat)
		
		if cat.suppress == 'hide':
			logger.info('Final plots are not displayed to screen. Check saved data.')
			return
		else:
			try:
				self.__displayplots(cat, uncert, allflux=allflux, spec=self.spec, spectral=self.spectralaxis, refspec=self.refspec, specuncert=self.specuncert, extra=extra)
			except (SystemExit, KeyboardInterrupt):
				uf.exit(cat)
			except:
				logger.critical('Encountered an error trying to display the plots:', exc_info=True)
				uf.earlyexit(cat)
		return















               