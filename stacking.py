""" use this to wrap around HISS """


import astropy.io.ascii as astasc
import astropy.table as asttab
import astropy.constants as astcon
import astropy.io.fits as astfit
import astropy.wcs as astwcs
import astropy.units as astun
import astropy.coordinates as astcoo
import astropy.cosmology as astcos

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family': 'serif', 'size': 14}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', facecolor='w', figsize=(8,6.5))


import os, sys
import useful_functions as uf
import glob

c = (astcon.c.to('km/s')).value
cosmo = astcos.FlatLambdaCDM(70, 0.3)


fullcat = astasc.read('../Data/Coma/StackingCatalogues/Coma_fullsample_19October2019.csv')
fullcat = fullcat[fullcat['hidata'] == 1]


fullcat['T'][fullcat['Morphs'] == 'S0'] = 0
fullcat['T'][fullcat['Morphs'] == 'L'] = 1
fullcat['T'][fullcat['Morphs'] == 'E'] = 2
fullcat['T'][fullcat['Morphs'] == 'U'] = 3
fullcat['T'][(fullcat['Morphs'] == 'I') & (fullcat['gmag']-fullcat['rmag'] > 0.6)] = 4
fullcat['T'][(fullcat['Morphs'] == 'I') & (fullcat['gmag']-fullcat['rmag'] <= 0.6)] = 5

config = """{
"#": "json does not have comments, so the '#' entries are comment placeholders",
"CatalogueFilename": "/Users/juliahealy/Documents/OneDrive - University of Cape Town/PhD/Stacking/Coma/%s_cat.csv",
"CatalogueColumnNumbers": [0,30,10,11,%s,%s], 		"#": "column numbers for [Object ID, Filename, Redshift, Redshift Uncertainty, Stellar Mass, Other Data], Other Data is other data which is used to bin the sample - if one of the columns is not needed leave the column entry as an empty string",
"z_max": 0.037, 		"#": "maximum redshift value",
"z_min": 0.01, 		"#": "minimum redshift value",
"SpectrumFluxUnit": 1, 		"#": "1-Jy, 2-mJy, 3-uJy, 4-Jy/beam, 5-mJy/beam, 6-uJy/beam",
"StackFluxUnit": %i, 		"#": "1-Jy, 2-Msun",
"SpectralAxisUnit": 5, 		"#": "1-Hz, 2-kHz, 3-MHz, 4-m/s, 5-km/s",
"VelocityType": "radio",	"#": "[optical/radio] - Only change this if you know that the spectra use radio velocity",
"ChannelWidth": 8.2, 		"#": "in same units as spectralunit",
"SpectrumLocation": "/Users/juliahealy/Documents/OneDrive - University of Cape Town/PhD/Data/Coma/hispectra/",		"#": "Location of spectra",
"FirstRowInSpectrum": 1, 		"#": "first row of data in spectrum file (numbering starts at 1)",
"RowDelimiter": ",",		"#": "spectra file data delimiter - if the delimiter is whitespace, use none",
"SpectrumColumns": [0,%i], 		"#": "column numbers for [freq/vel, flux]",
"cosmo": "y", 		"#": "Change default cosmology [y/n]",
"H0": 70, 		"#": "Hubble constant (H0) in units of km/Mpc/s (default is 70 km/Mpc/s)",
"Om0": 0.3, 		"#": "Omega matter (Om0) (default is 0.3)",
"changeoutloc": "y", 		"#": "Change output location [y/n] default is ./output/",
"OutputLocation": "/Users/juliahealy/Documents/OneDrive - University of Cape Town/PhD/Stacking/Coma/%s",
"chanlen": "y", 		"#": "Change default length of stacked spectrum [y/n]",
"GalaxyWidth": 400, 		"#": "default galaxy width is 450km/s, this option must use same units as spectralunit",
"StackedSpectrumLength": 1500, 		"#": "default stacked spectrum length is 2000km/s, this option must use same units as 'spectralunit'",
"weight": "y", 		"#": "Use a weighting scheme [y/n]",
"WeightOption": 1, 		"#": "1-w=1.[default], 2-w=1/rms, 3-w=1/rms^2, 4-w=dl^2/rms^2",
"StackEntireCatalogueYN": "y", 		"#": "stack entire catalogue [y/n]",
"NumberCatalogueObjects": 10, 		"#": "number of spectra to stack if entcat=y",
"UncertaintyYN": "n", 		"#": "Perform uncertainty analysis, [y/n]",
"UncertaintyMethod": "redshift", 	"#": "Uncertainty calculation method [redshift/dagjk]",
"BinYN": "%s", 		"#": "Bin the catalogue sample",
"BinInfo": %s, 		"#": "[Column name of data to bin, Bin width, Start of Bin range, End of Bin range]",
"Bins": %s,
"IntegrateGalaxyWidth": %s,
"SmoothStatus": %s,
"SmoothType": null,
"SmoothWindow": null,
"FittedFunctionNumbers": [],
"AnalysisOptions": %s
}
	"""

units = {
	1: 'flux',
	3: 'gas',
}


subsamples = {
	# 'nondetections_all': ['detected', 0, '20', '""', 1, 'n', '""', '""', '["4"]', '400.0', 'false'], ## column name to cut catalogue | value to cut | stellar mass column | other data column | stack unit | bin YN | BinInfo | Bins | analysis options | max width | rebinstatus
	# 'morphologies': ['detected', 0, '20', '33', 1, 'y', '["Other Data", 1, -0.5, 5.5]', '[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]', '[["4"], ["4"], ["4"], ["4"], ["4"], ["4"]]', '[400.0, 400.0, 400.0, 400.0, 400.0, 400.0]' , '[false, false, false, false, false, false]' ],
	# 'stellarmass': ['detected', 0, '20', '""', 1, 'y', '["Stellar Mass", 1, 6, 12]', '[6, 7, 8, 9, 10, 11, 12]', '[["4"], ["4"], ["4"], ["4"], ["4"], ["4"]]', '[400.0, 400.0, 400.0, 400.0, 400.0, 400.0]' , '[false, false, false, false, false, false]' ],
	# 'stellarmassfullcat': ['hidata', 1, '20', '""', 1, 'y', '["Stellar Mass", 1, 6, 12]', '[6, 7, 8, 9, 10, 11, 12]', '[["4"], ["4"], ["4"], ["4"], ["4"], ["4"]]', '[400.0, 400.0, 400.0, 400.0, 400.0, 400.0]' , '[false, false, false, false, false, false]' ],
	# 'hideficiency': ['detected', 0, '31', '20', 3, 'y', '["Other Data", 1, 6, 12]', '[6, 7, 8, 9, 10, 11, 12]', '[["4"], ["4"], ["4"], ["4"], ["4"], ["4"]]', '[400.0, 400.0, 400.0, 400.0, 400.0, 400.0]' , '[false, false, false, false, false, false]' ],,
	# 'substructure': ['detected', 0, '31', '18', 1, 'y', '["Other Data", 1, 0.5, 15.5]', '[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]', '[["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"]]', '[400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]' , '[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]' ],
	# 'substructureFC': ['hidata', 1, '31', '18', 1, 'y', '["Other Data", 1, 0.5, 15.5]', '[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]', '[["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"]]', '[400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]' , '[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]' ]

	# 'xray': ['detected', 0, '31', '40', 3, 'y', '["Other Data", 1e-4, 0, 0.002]', '[0, 3e-5, 5e-5, 7e-5, 1e-4, 0.0002, 0.0004, 0.001, 0.002]', '[["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"]]', '[400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]' , '[false, false, false, false, false, false, false, false]' ],
	# 'size': ['detected', 0, '31', '13', 3, 'y', '["Other Data", 5, 0, 100]', '[1, 5, 10, 15, 20, 25, 30, 35, 100]', '[["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"]]', '[400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]' , '[false, false, false, false, false, false, false, false]' ],
	# 'substructure': ['detected', 0, '31', '18', 3, 'y', '["Other Data", 1, 0.5, 15.5]', '[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]', '[["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"]]', '[400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]' , '[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]' ],
	# 'substructureFC': ['hidata', 1, '31', '18', 3, 'y', '["Other Data", 1, 0.5, 15.5]', '[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]', '[["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"], ["4"]]', '[400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]' , '[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]' ]
	# 'morphologies': ['detected', 0, '31', '33', 3, 'y', '["Other Data", 1, -0.5, 5.5]', '[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]', '[["4"], ["4"], ["4"], ["4"], ["4"], ["4"]]', '[400.0, 400.0, 400.0, 400.0, 400.0, 400.0]' , '[false, false, false, false, false, false]' ]
}

for sample in subsamples.keys():
	tempcat = fullcat.copy()
	tempcat = tempcat[ tempcat[subsamples[sample][0]] == subsamples[sample][1]]

	tempcat.write('/Users/juliahealy/Documents/OneDrive - University of Cape Town/PhD/Stacking/Coma/%s_cat.csv'%sample, overwrite=True)
	outloc = sample+'/'+units[subsamples[sample][4]]+'/target'

	configfile = config%(sample, subsamples[sample][2], subsamples[sample][3], subsamples[sample][4], 1, outloc, subsamples[sample][5], subsamples[sample][6], subsamples[sample][7], subsamples[sample][9], subsamples[sample][10], subsamples[sample][8] )

	outfile = open('%s.json'%sample, 'w')
	outfile.write(configfile)
	outfile.close()

	os.system('python ../../Stacker/pipeline.py -s -f %s.json '%sample)

	# os.system('mail -s "message from Serenity" juliahealyza@gmail.com <<< "Finished stacking %s"'%(sample))
	os.system('mv  %s.json Coma/%s/'%(sample, outloc))
	# break
	for i in range(1, 26):

		outloc = sample+'/'+units[subsamples[sample][4]]+'/ref_%i'%i
		configfile = config%(sample, subsamples[sample][2], subsamples[sample][3], subsamples[sample][4], 1+i, outloc, subsamples[sample][5], subsamples[sample][6], subsamples[sample][7], subsamples[sample][9], subsamples[sample][10], subsamples[sample][8] )

		outfile = open('%s_ref_%i.json'%(sample, i), 'w')
		outfile.write(configfile)
		outfile.close()

		os.system('python ../../Stacker/pipeline.py -s -f %s_ref_%i.json '%(sample, i))
		# os.system('mail -s "message from Serenity" juliahealyza@gmail.com <<< "Finished stacking %s_ref_%i"'%(sample, i))
		os.system('mv  %s_ref_%i.json Coma/%s/'%(sample, i, outloc))






