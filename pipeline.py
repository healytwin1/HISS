# from __future__ import print_function
import sys
h = sys.version_info
if (sys.version_info < (2, 7)):
	print('Warning: Some functions may not work.')
	
else:
	pass




import argparse
parser = argparse.ArgumentParser(description='Initialise and run HI line stacker.')
parser.add_argument('-f','--file', required=False, help='Filename of configuration file.', action='store', nargs=1, default=None, metavar='<filepath+filename>', dest='config')
parser.add_argument('-m','--multi', help='Option to use multiprocessing module to run the uncertainty calculation. Need at least 8Gb of RAM to run this mode.', action='store_const', const='multi', default=None, required=False, metavar='MultiProcessing', dest='multiop')
parser.add_argument('-d','--display', help='Option to display progress window during the stacking process.', action='store_const', const='progress', default=None, required=False, metavar='Progress Window', dest='progress')
parser.add_argument('-p','--saveprogress', help='Option to save progress window during the stacking process.', action='store_const', const='save', default=None, required=False, metavar='Progress Window', dest='saveprogress')
parser.add_argument('-s','--suppress', help='Use this flag to suppress all output windows. Note that [suppress] and [progress] cannot be used simultaneously.', action='store_const', const='hide', default=None, required=False, metavar='Suppress Output', dest='suppress')
parser.add_argument('-c','--clean', help='Use [clean] for testing purposes and for stacking noiseless spectra as this option will bypass any noise-related functions and actions.', action='store_const', const='clean', default=None, required=False, metavar='Clean Stack', dest='clean')
parser.add_argument('-l','--latex', help='This option enables to the use of latex formatting in the plots.', action='store_const', const='latex', default=None, required=False, metavar='Latex', dest='latex')

args = parser.parse_args()
config = args.config


import matplotlib
if args.suppress == 'hide':
	matplotlib.use('Agg')	
	progress = None
else:
	pass
	# matplotlib.use('TkAgg')

if args.latex == 'latex':
	matplotlib.rc('text', usetex=True)
	matplotlib.rc('text.latex', unicode=True, preamble=[r'''\usepackage{amsmath}''', r'''\usepackage{color, colortbl}''', r'''\usepackage[dvipsnames]{xcolor}'''])

import pkg_resources as pk
import logging 
import os, time

runtime = time.strftime("%d-%m-%Y")+'_'+time.strftime("%H-%M")
logging.basicConfig(format='(%(asctime)s) [%(name)-17s] %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S", filename='hiss_%s.log'%runtime,level=logging.DEBUG)
logging.captureWarnings(True)

modules = ['astropy', 'matplotlib', 'scipy', 'numpy', 'yaml']
versions = ['3.0.0','1.5', '0.17','1.09.1','3.11']
def checkmod(modname, version):
	try:
		if modname == 'yaml':
			__import__(modname)# this will fail if the module doesn't exist on the system
			return True
		else:
			__import__(modname)# this will fail if the module doesn't exist on the system
			currentversion = pk.get_distribution(modname).version
			if currentversion < version:
				print ('Please update %s, you have version %s installed and this package needs to be at least version %s.'%(modname, currentversion, version))
				status = False
				
				return status
			else:
				status = True
				return status
	except ImportError:
		print ('This package needs %s, please install it and try again.'%modname)
		status = False
		return status

status = True
for m in range(5):
	newstatus = checkmod(modules[m], versions[m])
	if newstatus == False:
		status = False

if status == False:
	sys.exit()
else:
	pass


import inputData as ID
import stackData as SD
import analyseData as AD
import uncertAnalysis as UA
import displayData as DD
import binData as BD
import useful_functions as uf
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gr
import astropy.units as astun
import astropy.constants as astcon
import astropy.io.ascii as astasc
import astropy.table as asttab
import astropy.units as astun
import multiprocessing as multi
import warnings
import astropy
import json
import platform
import gc
import _pickle as pck
from functools import partial
import logging

logging.disable(logging.DEBUG)
logger = logging.getLogger(__name__)

# os.system("taskset -p 0xff %d" % os.getpid())

warnings.filterwarnings('ignore')

font = {'family': 'serif', 'size': 12}

matplotlib.rc('font', **font)
matplotlib.rc('figure', facecolor='w')
gc.enable()

c = astcon.c.to('km/s').value

if config is not None:
	configfile = args.config[0]
	data = json.load(open(configfile))
	config = True
else:
	data = None


def axisunit(cat):
	if astun.Jy == cat.stackunit:
		labstr2 = 'Flux density per\ngalaxy (mJy)'
		labstr3 = 'Total Stacked Flux density (mJy)'
	elif uf.msun == cat.stackunit:
		labstr2 = 'Flux density per\ngalaxy ($\mathrm{M_\odot/chan}$)'
		labstr3 = 'Total Stacked Flux density ($10^8 \mathrm{M_\odot/chan}$)'
	else:
		labstr2 = r'Gas Fraction per\ngalaxy (M$_{\mathrm{HI}}$/M$_\star$)'
		labstr3 = r'Total Stacked Gas Fraction (M$_{\mathrm{HI}}$/M$_\star$)'

	return labstr2, labstr3


def startfigure(cat):
	if cat.progress == 'progress' or cat.saveprogress == 'save':
		fig = plt.figure(figsize=(11.7,8))
		gs = gr.GridSpec(3,4)
		ax1 = fig.add_subplot(gs[0,0:2])
		ax2 = fig.add_subplot(gs[1,0:2])
		ax3 = fig.add_subplot(gs[:,2:])
		ax4 = fig.add_subplot(gs[2,0:2])
		plt.subplots_adjust(left=0.09,right=0.97,top=0.97,bottom=0.09, wspace=0.5, hspace=0.3)
		# plt.suptitle("Progress Window", y=1.01)
		labstr2, labstr3 = axisunit(cat)

		ax2.set_ylabel(labstr2)
		ax3.set_ylabel(labstr3)
		ax3.set_xlim(cat.spectralaxis.value[0],cat.spectralaxis.value[-1])
		ax2.set_xlim(cat.spectralaxis.value[0],cat.spectralaxis.value[-1])
		xx = ax2.get_xticks()
		lab = ["%i" % a for a in xx]
		ax2.set_xticks(xx)
		ax2.axhline(0, color='gray', ls='--')
		ax2.set_xticklabels(lab)
		ax3.set_xticks(xx)
		ax3.set_xticklabels(lab)
		ax3.set_xlim(cat.spectralaxis.value[0],cat.spectralaxis.value[-1])
		ax1.set_ylabel('Flux density per \ngalaxy (mJy)')
		ax1.axhline(0, color='gray', ls='--')
		
		ax2.set_xlabel('Relative Velocity (km/s)')
		ax3.set_xlabel('Relative Velocity (km/s)')

		ax2.set_xlim(cat.spectralaxis.value[0],cat.spectralaxis.value[-1])
		ax3.set_xlim(cat.spectralaxis.value[0],cat.spectralaxis.value[-1])

		if 'Hz' in cat.spectralunit.to_string():
			xs = (cat.femit/(1+cat.min_redshift-0.001)).value
			xe = (cat.femit/(1+cat.max_redshift)).value
			ax1.set_xlim(xs,xe)
			ax1.set_xlabel('Observed Frequency ('+cat.spectralunit.to_string()+')')
		else:
			xs = (cat.min_redshift-0.001)*c
			xe = (cat.max_redshift)*c
			ax1.set_xlim(xs,xe)
			yy = ax1.get_xticks()
			lab = ["%i" % a for a in yy]
			ax1.set_xticks(yy)
			ax1.set_xticklabels(lab)
			ax1.set_xlim(xs,xe)
			ax1.set_xlabel('Velocity ('+cat.spectralunit.to_string()+')')
		return fig, ax1, ax2, ax3, ax4
	else:
		return None, None, None, None, None


def plotprogress(cat, stackobj, col, stackspectrum, spectrum, fig, ax1, ax2, ax3, ax4, n):
	if fig is None:
		return None
	else:
		yvals = (stackspectrum.spec/stackspectrum.weightsum) * stackspectrum.nobj
		xvals = stackspectrum.spectralaxis.value
		svals = spectrum.extendspec.value

		if astun.Jy == cat.stackunit:
			yvals = yvals.value*1E3
			svals = svals*1E3
		elif uf.msun == cat.stackunit:
			yvals = yvals.value/1E8
			svals = svals/1E8

		
		ax3.cla()
		txt = ax3.annotate('N = %i'%(stackspectrum.nobj),xy=(0.5,0.97), xycoords='axes fraction', horizontalalignment='center', fontsize=12, color='k')

		ax3.set_ylim(1.05*np.min(yvals),1.08*np.max(yvals))
		labstr2, labstr3 = axisunit(cat)

		stackednoise = np.array(stackspectrum.stackrms)*1E6
		objects = range(1,len(stackednoise)+1)
		ax4.cla()
		ax4.set_xlabel('Number of stacked spectra')
		ax4.set_ylabel('Stacked Noise ($\mu$Jy)')
		ax4.set_xscale('log')
		ax4.set_yscale('log')
		ax4.plot(objects, stackednoise, 'ko')
		ax4.set_xlim(0.95, stackobj+10)

		ax1.plot(spectrum.spectralspec.value, spectrum.origspec.value*1E3, color=col)

		ax2.cla()
		if cat.stackunit == astun.Jy:
			spec = spectrum.extendspec.to('mJy').value
		else:
			spec = spectrum.extendspec.value
		ax2.set_xlim(cat.spectralaxis.value[0],cat.spectralaxis.value[-1])
		xx = ax2.get_xticks()
		lab = ["%i" % a for a in xx]
		ax2.set_xticks(xx)
		ax2.axhline(0, color='gray', ls='--')
		ax2.set_xticklabels(lab)
		ax2.axvspan(xmin=-cat.maxgalw.value/2., xmax=cat.maxgalw.value/2., color='lavender')
		ax2.plot(cat.spectralaxis.value, spec, color=col)
		ax2.set_xlabel('Relative Velocity ('+cat.spectralunit.to_string()+')')
		labstr2, labstr3 = axisunit(cat)
		ax2.set_ylabel(labstr2)
		ax2.set_xlim(cat.spectralaxis.value[0],cat.spectralaxis.value[-1])

		ax3.axvspan(xmin=-cat.maxgalw.value/2., xmax=cat.maxgalw.value/2., color='lavender')
		ax3.plot(xvals, yvals, color='k', label='Total Stacked Spectrum')
		ax3.plot(xvals, svals, color=col, label='Spectrum ID: %s'%str(cat.catalogue['Object ID'][n]))
		ax3.axhline(0, color='gray',ls='--')
		ax3.set_xlabel('Relative Velocity ('+cat.spectralunit.to_string()+')')
		ax3.set_ylabel(labstr3)
		ax3.set_xlim(xvals[0],xvals[-1])
		ax3.legend(loc='lower left', fontsize=10, numpoints=1)

		fig.canvas.draw()
		return fig


def stackeverything(pbar,cat, stackobj, binstatus, callno, binno, mccount, fig, ax1, ax2, ax3, ax4, analysisfinal=None):
	stackspectrum = SD.objStack(spec=np.zeros(cat.exspeclen)*cat.stackunit,ref=np.zeros(cat.exspeclen)*cat.stackunit, noise=np.zeros(cat.noiselen), weight=0.,spectral=cat.spectralaxis, sunits=cat.stackunit)
	colours = uf.colourgen(stackobj)
	## loop through all of spectra listed in the catalogue
	######################################################################################
	## Stack the catalogue
	######################################################################################

	for n in range(0, stackobj):
		spectrum = SD.objSpec()
		col = colours[n]
		cat = spectrum.callModule(cat, n, mccount, axes=[ax1,ax2], fig=fig)
		if spectrum.origspec is None:
			continue
		else:
			stackspectrum = stackspectrum + spectrum
			fig = plotprogress(cat, stackobj, col, stackspectrum, spectrum, fig, ax1, ax2, ax3, ax4, n)
			if mccount == 0:
				frac = int((n+1.)/stackobj * 100.)
				if frac < 2:
					frac = 0
				else: 
					h = 0
				pbar = uf.progressbar(frac,pbar)
				if cat.saveprogress == 'save':
					outloc = cat.outloc+'progressplots/'
					uf.checkpath(outloc)
					if binstatus == 'bin':
						plt.savefig(outloc+'bin%i_window_%i_source_%s.png'%(binno, n, str(cat.catalogue['Object ID'][n])), bbox_inches='tight', pad_inches=0.2)
					else:
						plt.savefig(outloc+'window_%i_source_%s.png'%(n, str(cat.catalogue['Object ID'][n])), bbox_inches='tight', pad_inches=0.2)
				else:
					h = 0
			else:
				h = 0
	######################################################################################
	######################################################################################
	# close the progress window before moving on to analysis
	if cat.progress == 'progress' and mccount == 0:
		time.sleep(5)
		plt.close('all')
	elif cat.saveprogress == 'save':
		plt.close('all')
	else:
		cat.progress = None

	######################################################################################
	## Abort process if the stack is empty
	######################################################################################
	if len(stackspectrum.stackrms) == 0 and len(cat.outcatalogue) == 0 and cat.uncert == 'n':
		print ("\nThere were no spectra to stack or something went wrong. I am going to abort and you need to check that there are spectrum files in the correct folder and that the spectrum file names are correctly entered in to the the catalogue.")
		logging.critical('There were no spectra to stack, or something else went wrong. Please check that the spectrum files in specified folder.')
		uf.earlyexit(cat)
	elif len(stackspectrum.stackrms) == 0 and len(cat.outcatalogue) != 0 and cat.uncert == 'n':
		print ("\nThere were no spectra to stack or something went wrong. I am going to abort and you need to check that there are spectrum files in the correct folder and that the spectrum file names are correctly entered in to the the catalogue.")
		return cat
	elif len(stackspectrum.stackrms) == 0 and len(cat.outcatalogue) == 0 and cat.uncert == 'y':

		return cat, None
	######################################################################################
	## Continue on to analysis
	######################################################################################
	else:
		analysis = AD.anaData()
		cat = analysis.analyse_data(stackspectrum, cat, mccount)
		##################################################################################
		## If Uncertainty needs to run..
		##################################################################################
		stackspectrum, spectrum = None, None
		if cat.uncert == 'n' or callno == 'firstcall':
			analysisfinal = UA.uncertAnalysis(cat, analysis, allspec=[], allnoise=[], allrms=[], middle=[], allintflux=np.array([[],[],[],[],[],[],[],[]]))
			return cat, analysisfinal
		else:
			analysisfinal.updateStackAnalysis(cat, mccount, analysis)
			analysis = None
			return cat
	return cat



def pipeline(cat, config, data, binstatus, binno=None, mccount=0):
	disp, spectrum, stackspectrum, analysis, analysisfinal = None, None, None, None, None

	fig, ax1, ax2, ax3, ax4 = startfigure(cat)

	if cat.progress == 'progress':
		plt.show(block=False)
	else:
		pass

	## creating the stack objects
	xs = cat.spectralaxis.value[0]
	xe = cat.spectralaxis.value[-1]

	if uf.checkkeys(data, 'StackEntireCatalogueYN'):
		stacknoYN = data['StackEntireCatalogueYN']
	else:
		stacknoYN = raw_input('Do you want to stack your entire catalogue? [y/n] ')

	if stacknoYN.lower() == 'y':
		stackobj = len(cat.catalogue)
	else:
		if uf.checkkeys(data, 'NumberCatalogueObjects'):
			stackobj = int(data['NumberCatalogueObjects'])
		else:
			stackobj = None
			while type(stackobj) != int:
				stackobj = int(input('Please enter the number of profiles you want to stack: '))
		if stackobj > cat.numcatspec:
			stackobj = len(cat.catalogue)
		else:
			h = 0
		logging.info('Number of objects going into the stack: %i'%stackobj)

	if cat.uncert == 'y' and mccount == 0:
		if binstatus == 'bin':
			print ('\nProgress of first round of Bin %i: '%binno)
			logging.info('Starting first round of Bin %i.'%binno)
		else:
			print ('\nProgress of first round: ')
			logging.info('Starting stacking first round.')
	elif cat.uncert == 'n':
		if binstatus == 'bin':
			print('\nStacking progress of Bin %i: '%binno)
			logging.info('Starting stacking of Bin %i.'%binno)
		else:
			print('\nStacking progress: ')
			logging.info('Starting stacking.')
	else:
		h = 0

	pbar = ''
	if ((mccount == 0) or ((mccount == 1)) and (binstatus == 'bin')):
		cat, analysisfinal = stackeverything(pbar,cat, stackobj, binstatus, 'firstcall', binno, mccount, fig, ax1, ax2, ax3, ax4)
	else:
		h = 0

	if mccount == 0 and cat.uncert == 'y' and binstatus == 'bin':
		logging.info('Finished stacking first round of Bin %i.'%binno)
		return cat
	else:
		disp = DD.dispData(cat)
		mccount += 1
		if cat.uncert == 'y':
			if binstatus == 'bin':
				print ('\n\nUncertainty calculations of Bin %i in progress:'%binno)
			else:
				print ('\n\nUncertainty calculations in progress:')
			pbar = ''
			if cat.multiop == 'multi':
				results = []
				cpu_count = int(multi.cpu_count()/2)
				logging.info('Spawning %i processes to run uncertainty calculations.'%cpu_count)
				pool = multi.Pool(processes=cpu_count, maxtasksperchild=1)
			else:
				h = 0
			analysisempty = UA.uncertAnalysis(cat, analysisfinal, allspec=[], allnoise=[], allrms=[], middle=[], allintflux=np.array([[],[],[],[],[],[],[],[]]))
			mccount = 1
			suppress = cat.suppress
			cat.suppress = 'hide'
			fullcatfile = cat.catalogue
			logging.info('Starting uncertainty calculations...')
			while mccount < (cat.mc+1):
				if cat.uncerttype == 'dagjk':
					indices = np.random.randint(low=0, high=(len(fullcatfile)-1), size=int(cat.R*len(fullcatfile)) )
					cat.catalogue = fullcatfile[indices]
					if stackobj > len(cat.catalogue):
						stackobj = len(cat.catalogue)
					else:
						h = 0
				else:
					h = 0
				if mccount > 0 and cat.multiop != 'multi':
					frac = int((mccount)/(cat.mc*2.) * 100.)
					if frac < 2:
						frac = 0
					else:
						h = 0
					pbar = uf.progressbar(frac,pbar)
					cat = stackeverything(pbar,cat, stackobj, binstatus, 'uncertcall', binno, mccount, fig, ax1, ax2, ax3, ax4, analysisfinal=analysisfinal)
				else:
					result = pool.apply_async(stackeverything, args=(pbar,cat, stackobj, binstatus, 'uncertcall', binno, mccount, fig, ax1, ax2, ax3, ax4 , analysisfinal))
					results.append(result)
				mccount += 1
			if cat.multiop == 'multi':
				pool.close()
				pbar = ''
				for n in range(cat.mc):
					frac = int((n+1.)/cat.mc * 100.)
					if frac < 2:
						frac = 0
					else: 
						h = 0
					pbar = uf.progressbar(frac,pbar)
					results[n].get()
					analysisfinal = analysisfinal + analysisempty.load(n+1, cat)
			else:
				for n in range(1, cat.mc+1, 1):
					frac = int( float(cat.mc + n)/(cat.mc*2.) * 100. )
					pbar = uf.progressbar(frac, pbar)
					analysisfinal = analysisfinal + analysisempty.load(n, cat)

			cat.suppress = suppress
			analysisfinal.ProcessUncertainties(cat)
			pck.dump( analysisfinal, open('%sstackedresultsobj_%s.pkl'%(cat.outloc, cat.runtime), 'wb') )
			disp.display(cat, analysisfinal.allintflux,[analysisfinal.middle, analysisfinal.allrms], args.suppress)
		else:
			disp.display(cat, None, cat.suppress)
		return cat


def main():

	fullcat = ID.inputCatalogue(runtime)

	fullcat.runInput(data)
	fullcat.clean = args.clean
	fullcat.suppress = args.suppress
	fullcat.progress = args.progress
	fullcat.saveprogress = args.saveprogress
	fullcat.multiop = args.multiop
	fullcat.latex = args.latex

	## create a temp file for all object files
	uf.checkpath('%sstackertemp/'%fullcat.outloc)
	if uf.checkkeys(data, 'BinYN'):
		binyn = data["BinYN"]
	else:
		binyn = raw_input('\nWould you like to stack in various bins? [y/n] ').lower()

	if binyn == 'y':
		bincatalogue = BD.binnedCatalogue(fullcat)
		try:
			bincatalogue.determineBins(data)
		except (SystemExit, KeyboardInterrupt):
			logger.error('Early exit', exc_info=False)
			uf.exit(fullcat)
		except:
			logging.warning("Couldn't determine the catalogue for the different bins:", exc_info=True)
			uf.earlyexit(fullcat)

		if fullcat.uncert == 'y':
			repeat = 2
		else:
			repeat = 1

		counter = 0
		# print(bincatalogue.catalogue)
		while counter < repeat:
			mccount = counter
			for m in range(0,len(bincatalogue.binpartindicies)-1):
				if counter > 0:
					bincatalogue = pck.load( open('%sstackertemp/catalogue_obj.pkl'%fullcat.outloc, 'rb') )
				else:
					h = 0
				try:
					bincatalogue.createCatalogue(bincatalogue.binpartindicies, bincatalogue.binnedtable, m, counter)
				except KeyboardInterrupt:
					logger.error('Early exit', exc_info=False)
					uf.earlyexit(fullcat)
					raise sys.exit()
				except SystemExit:
					logger.error('Early exit', exc_info=False)
					uf.earlyexit(fullcat)
					raise sys.exit()
				except:
					logger.error('Exception occurred trying to create the binned catalogue.', exc_info=True)
					uf.earlyexit(fullcat)
					raise sys.exit()
				
				if len(bincatalogue.catalogue) < 2:
					logger.info('Bin %i has too few objects'%(m+1))
					pass
				else:
					try:
						catalogue = pipeline(bincatalogue, config, data, 'bin', m+1, mccount)
						if counter == 0:
							bincatalogue = bincatalogue + catalogue
						else:
							pass
					except TypeError:
						logger.error('Problem with bin %i. Skipping'%(m+1))
			if (counter == 0):
				bincatalogue.outcatalogue['Stellar Mass'].unit = bincatalogue.fullcatalogue['Stellar Mass'].unit
				bincatalogue.outcatalogue['Other Data'].unit = bincatalogue.fullcatalogue['Other Data'].unit
				bincatalogue.outcatalogue['Integrated Flux'].unit = astun.Jy*astun.km/astun.s

				pck.dump( bincatalogue, open('%sstackertemp/catalogue_obj.pkl'%bincatalogue.outloc, 'wb') )

				catalogue = uf.maskTable(bincatalogue.outcatalogue)
				if len(catalogue) > 1:
					catalogue = asttab.unique(catalogue, keys='Object ID')
				else:
					pass
				catalogue.sort('Bin Number')
				astasc.write(catalogue, bincatalogue.outloc+'Stacked_Catalogue_%s.csv'%bincatalogue.origruntime, format='ecsv')
				fullcat.updateconfig(data, bincatalogue.fullmaxgalw, bincatalogue.fullrebinstatus, bincatalogue.fullsmoothtype, bincatalogue.fullsmoothwin, bincatalogue.fullfuncnum, bincatalogue.fulloptnum)
			else:
				pass
			saveprogress = None
			counter += 1
	else:
		fullcat.catalogue.add_column(asttab.Column(name='Bin Number', data=np.ones(len(fullcat.catalogue), dtype=int)))
		fullcat = pipeline(fullcat, config, data, 'normal')
		try:
			fullcat.updateconfig(data, fullcat.maxgalw.value, fullcat.rebinstatus, fullcat.smoothtype, fullcat.smoothwin, fullcat.funcnum, fullcat.optnum)
			fullcat.outcatalogue['Stellar Mass'].unit = fullcat.catalogue['Stellar Mass'].unit
			fullcat.outcatalogue['Other Data'].unit = fullcat.catalogue['Other Data'].unit
			fullcat.outcatalogue['Integrated Flux'].unit = astun.Jy*astun.km/astun.s
			catalogue = uf.maskTable(fullcat.outcatalogue)
			catalogue = asttab.unique(catalogue, keys='Object ID')
			catalogue.sort('Bin Number')
			astasc.write(catalogue, fullcat.outloc+'Stacked_Catalogue_%s.csv'%fullcat.runtime, format='ecsv')
			logging.info('Written Stacked Catalogue to file.')
		except (SystemExit, KeyboardInterrupt):
			logger.error('Early exit', exc_info=False)
			uf.exit(fullcat)
		except:
			logging.warning("Struggled to save the catalogue files.", exc_info=True)
			uf.earlyexit(fullcat)

	## inform the user that the stacking has finished
	if fullcat.suppress != 'hide':
		outloc = uf.bashfriendlypath(fullcat.outloc)
		os.system('open '+outloc)
	else:
		pass

	print( '\nStacker has finished.\n')
	uf.exit(fullcat)
	return


if __name__ == '__main__':
	try:
		main()
	except SystemExit: pass
	except:
		logger.error('Early exit', exc_info=True)
		raise sys.exit()
else:
	main()




