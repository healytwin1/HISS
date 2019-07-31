import sys
h = sys.version_info
if (sys.version_info < (3, 0)):
    import Tkinter as tk
    import tkFileDialog as tkf
else:
    import tkinter as tk
    import tkinter.filedialog as tkf 


import sys, os, linecache
import json, io
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import astropy.io.ascii as astasc
import useful_functions as uf
font = {'family': 'serif', 'size': 25}
matplotlib.rc('font', **font)
matplotlib.rc('figure', facecolor='w', figsize=(8,5))


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


class hiss(tk.Frame):

    def __init__(self):
        self.root = tk.Tk()
        tk.Frame.__init__(self, self.root, width=1025)

        ## necessary variables
        self.catalogue = None
        self.CatalogueFilenameVar = tk.StringVar(self, value="")
        self.OutputLocationVar = tk.StringVar(self, value="")
        self.SpectrumLocationVar = tk.StringVar(self, value="")
        self.CatalogueFilenameVar = tk.StringVar(self, value="")
        self.SpectrumFluxUnitVar = tk.IntVar(self, value=0)
        self.PixelSizeVar = tk.DoubleVar(self, value=0.0)
        self.BeamSizeVar = tk.DoubleVar(self, value=0.0)
        self.SpectralAxisUnitVar = tk.IntVar(self, value=0)
        self.VelocityTypeVar = tk.StringVar(self, value="optical")
        self.ChannelWidthVar = tk.DoubleVar(self, value=0.0)
        self.FirstRowInSpectrumVar = tk.IntVar(self, value=0)
        self.RowDelimiterVar = tk.StringVar(self, "")
        self.FluxColumnVar = tk.IntVar(self, value=0) 
        self.AxisColumnVar = tk.IntVar(self, value=0)
        self.StackedSpectrumLengthVar = tk.DoubleVar(self, value=2000)
        self.GalaxyWidthVar = tk.DoubleVar(self, value=450)
        self.ZminVar = tk.DoubleVar(self, value=0.0)
        self.ZmaxVar = tk.DoubleVar(self, value=0.0)
        self.StackFluxUnitVar = tk.IntVar(self, value=0)
        self.H0Var = tk.DoubleVar(self, value=70.0)
        self.Om0Var = tk.DoubleVar(self, value=0.3)
        self.WeightOptionVar = tk.IntVar(self, value=1)
        self.StackEntireCatalogueYNVar = tk.StringVar(self, value='y')
        self.NumberCatalogueObjectsVar = tk.IntVar(self, value=0)
        self.UncertaintyYNVar = tk.StringVar(self, value='n')
        self.UncertaintyMethodVar = tk.StringVar(self, value=0)
        self.BinYNVar = tk.StringVar(self, value='n')
        self.BinColumnVar = tk.StringVar(self, value='Select column to bin')
        self.ObjectIDVar = tk.IntVar()
        self.FilenameVar = tk.IntVar()
        self.RedshiftVar = tk.IntVar()
        self.RedshiftErrorVar = tk.IntVar()
        self.StellarmassVar = tk.IntVar()
        self.OtherDataVar = tk.IntVar()
        self.columnlist = []
        self.BinStartVar = tk.DoubleVar(self, 0.0)
        self.BinEndVar = tk.DoubleVar(self, 1.0)
        self.BinSizeVar = tk.DoubleVar(self, 0.5)
        self.BinsVar = tk.StringVar(self, str(list(np.arange(0.,1.1, 0.5))))
        self.bsy = 0
        self.bey = 0
        self.bssy = 0

        ## variables to be saved to config file
        self.CatalogueFilename = None
        self.CatalogueColumnNumbers = [-1, -1, -1, -1, -1, -1]
        self.CatalogueColumnNumbersUpdated = ["", "", "", "", ""]
        self.OutputLocation = './'
        self.SpectrumLocation = None
        self.CatalogueFilename = None
        self.SpectrumFluxUnit = None
        self.PixelSize = None
        self.BeamSize = None
        self.SpectralAxisUnit = None
        self.VelocityType = None
        self.ChannelWidth = None
        self.FirstRowInSpectrum = None
        self.RowDelimiter = ""
        self.SpectrumColumns = [None, None]
        self.StackedSpectrumLength = None
        self.GalaxyWidth = None
        self.Zmin = None
        self.Zmax = None
        self.StackFluxUnit = None
        self.H0 = None
        self.Om0 = None
        self.WeightOption = None
        self.StackEntireCatalogueYN = None
        self.NumberCatalogueObjects = None
        self.UncertaintyYN = None
        self.UncertaintyMethod = None
        self.BinYN = None
        self.BinInfo = [None, None, None, None]
        self.Bins = None


        ## set up the master frame
        self.master.title('HI Stacking Software')
        self.master.iconname('HISS')
        tk.Label(self.root, text='Welcome to HI Stacking Software', font=(None, 24, "bold")).pack(side=tk.TOP)
        tk.Label(self.root, text='\nThis interface allows you (the user) to enter values required for stacking. Once you have entered a value into the grey boxes, hit <Return> - if the value is accepted, \nthe box will turn green. When you are ready, click on the Stack button at the bottom right of the window. If there are any missing values critical for stacking,\nthe relevant boxes will be highlighted in red.\n', font=(None, 14, "italic"), fg='darkviolet', bg='whitesmoke').pack(side=tk.TOP, pady=5)
        self.mastercanvas = tk.Canvas(self.root, width=1000, height=610, background="#ffffff", bd=2)
        self.frame = tk.Frame(self.mastercanvas, background="#ffffff", height=1000, width=900)
        self.vsb = tk.Scrollbar(self.root, orient="vertical", command=self.mastercanvas.yview)
        self.mastercanvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.mastercanvas.pack(side="top", fill="x", expand=True)
        self.mastercanvas.create_window((0,0), window=self.frame, anchor="nw", tags="self.frame")
    

        ## locations frame
        self.locationframe = tk.Frame(self.mastercanvas, bd=2, relief=tk.GROOVE)
        tk.Label(self.locationframe, text='Output location:', anchor="w").grid(sticky="W", in_=self.locationframe, row=1, column=1)
        self.outlocentry = tk.Entry(self.locationframe, textvariable=self.OutputLocationVar, width=95, background='beige')
        self.outlocentry.grid(in_=self.locationframe, row=1, column=2)
        tk.Button(self.locationframe, text='Browse', command=self.__getOutputLocation, width=8).grid(in_=self.locationframe, row=1, column=3)
        tk.Label(self.locationframe, text='Spectra location:', anchor="w").grid(sticky="W", in_=self.locationframe, row=2, column=1)
        self.speclocentry = tk.Entry(self.locationframe, textvariable=self.SpectrumLocationVar, width=95, background='beige')
        self.speclocentry.grid(in_=self.locationframe, row=2, column=2)
        tk.Button(self.locationframe, text='Browse', command=self.__getSpectrumLocation, width=8).grid(in_=self.locationframe, row=2, column=3)
        
        tk.Label(self.locationframe, text='Catalogue file:', anchor="w").grid(sticky="W", in_=self.locationframe, row=3, column=1)
        self.CatalogueFilenameEntry = tk.Entry(self.locationframe, textvariable=self.CatalogueFilenameVar, width=95, background='beige')
        self.CatalogueFilenameEntry.grid(in_=self.locationframe, row=3, column=2)
        tk.Button(self.locationframe, text='Browse', command=self.__getCatalalogueFilename, width=8).grid(in_=self.locationframe, column=3, row=3, columnspan=1)

        ## catalogue frame
        self.catalogueframe = tk.Frame(master=self.mastercanvas, bd=2, relief=tk.GROOVE)
        tk.Label(self.catalogueframe, text='Catalogue Information:', anchor=tk.W, font=(None, 14, "bold underline")).grid(sticky="W", in_=self.catalogueframe, column=1, row=1, columnspan=6)
        self.catalogueobjid = tk.Entry(self.catalogueframe, width=2, state=tk.DISABLED)
        self.cataloguefilename = tk.Entry(self.catalogueframe, width=2, state=tk.DISABLED)
        self.catalogueredshift = tk.Entry(self.catalogueframe, width=2, state=tk.DISABLED)
        self.catalogueredshifterr = tk.Entry(self.catalogueframe, width=2, state=tk.DISABLED)
        self.cataloguestellarmass = tk.Entry(self.catalogueframe, width=2, state=tk.DISABLED)
        self.catalogueotherdata = tk.Entry(self.catalogueframe, width=2, state=tk.DISABLED)

        tk.Label(self.catalogueframe, text='Enter the column numbers in the entry boxes and hit <Return>.', fg='darkblue').grid(sticky="W", in_=self.catalogueframe, row=2, column=1, columnspan=13)
        tk.Label(self.catalogueframe, text='Columns:', anchor="w").grid(sticky="W", in_=self.catalogueframe, row=3, column=1)
        tk.Label(self.catalogueframe, text='Object ID', anchor="w").grid(sticky="W", in_=self.catalogueframe, row=3, column=2)
        tk.Label(self.catalogueframe, text='Filename', anchor="w").grid(sticky="W", in_=self.catalogueframe, row=3, column=4)
        tk.Label(self.catalogueframe, text='Redshift', anchor="w").grid(sticky="W", in_=self.catalogueframe, row=3, column=6)
        tk.Label(self.catalogueframe, text='Redshift Error', anchor="w").grid(sticky="W", in_=self.catalogueframe, row=3, column=8)
        tk.Label(self.catalogueframe, text='Stellar Mass', anchor="w").grid(sticky="W", in_=self.catalogueframe, row=3, column=10)
        tk.Label(self.catalogueframe, text='Other Data', anchor="w").grid(sticky="W", in_=self.catalogueframe, row=3, column=12)
        self.catalogueobjid.grid(sticky="W", in_=self.catalogueframe, row=3, column=3)
        self.cataloguefilename.grid(sticky="W", in_=self.catalogueframe, row=3, column=5)
        self.catalogueredshift.grid(sticky="W", in_=self.catalogueframe, row=3, column=7)
        self.catalogueredshifterr.grid(sticky="W", in_=self.catalogueframe, row=3, column=9)
        self.cataloguestellarmass.grid(sticky="W", in_=self.catalogueframe, row=3, column=11)
        self.catalogueotherdata.grid(sticky="W", in_=self.catalogueframe, row=3, column=13)

        self.fig = plt.figure(figsize=(11,2))
        self.ax = self.fig.add_axes([0.01, 0.05, 0.97, 0.9])
        self.ax.annotate('Sample Catalogue', xy=(0.5,0.5), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center')
        self.ax.plot([0.1, 0.1, 0.99, 0.99, 0.1], [0.45, 0.99, 0.99, 0.45, 0.45], 'k-')
        self.ax.axis('off')
        self.canvasplot = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.fig, master=self.mastercanvas)
        self.canvasplot.draw()
        self.canvasplot.get_tk_widget().grid(sticky="n",in_=self.catalogueframe, row=4,column=1, columnspan=13)
         

        ## Spectra frame
        #### Input spectra info
        self.spectraframe = tk.Frame(master=self.mastercanvas, bd=2, relief=tk.GROOVE, width=950)
        tk.Label(self.spectraframe, text='Input Spectra:', anchor=tk.W, font=(None, 14, "bold underline")).grid(sticky="W", in_=self.spectraframe, row=1, column=1, columnspan=2)
        self.spectrumfluxunitlabel = tk.Label(self.spectraframe, text='Please select the spectrum flux density units:', anchor="w")
        self.spectrumfluxunitlabel.grid(sticky="W", in_=self.spectraframe, row=2, column=1, columnspan=6)
        tk.Radiobutton(self.spectraframe, text='Jy', variable=self.SpectrumFluxUnitVar, value=1, command=self.__getSpectrumFluxUnit).grid(sticky="W", in_=self.spectraframe, row=3, column=1)
        tk.Radiobutton(self.spectraframe, text='mJy', variable=self.SpectrumFluxUnitVar, value=2, command=self.__getSpectrumFluxUnit).grid(sticky="W", in_=self.spectraframe, row=3, column=2)
        tk.Radiobutton(self.spectraframe, text='uJy', variable=self.SpectrumFluxUnitVar, value=3, command=self.__getSpectrumFluxUnit).grid(sticky="W", in_=self.spectraframe, row=3, column=3)
        tk.Radiobutton(self.spectraframe, text='Jy/beam', variable=self.SpectrumFluxUnitVar, value=4, command=self.__getSpectrumFluxUnit).grid(sticky="W", in_=self.spectraframe, row=3, column=4)
        tk.Radiobutton(self.spectraframe, text='mJy/beam', variable=self.SpectrumFluxUnitVar, value=5, command=self.__getSpectrumFluxUnit).grid(sticky="W", in_=self.spectraframe, row=3, column=5)
        tk.Radiobutton(self.spectraframe, text='uJy/beam', variable=self.SpectrumFluxUnitVar, value=6, command=self.__getSpectrumFluxUnit).grid(sticky="W", in_=self.spectraframe, row=3, column=6)

        self.pixsizelabel = tk.Label(self.spectraframe, text='Pixel size (arcsec):', anchor=tk.W)
        self.beamsizelabel = tk.Label(self.spectraframe, text='FWHM of beam (arcsec):', anchor=tk.W)
        self.pixsizeentry = tk.Entry(self.spectraframe, textvariable=self.PixelSizeVar, width=5, background='lightgray')
        self.beamsizeentry = tk.Entry(self.spectraframe, textvariable=self.BeamSizeVar, width=5, background='lightgray')
        self.pixsizeentry.bind("<Return>", self.__getPixelsize)
        self.beamsizeentry.bind("<Return>", self.__getBeamsize)
       
        self.spectralaxisunitslabel = tk.Label(self.spectraframe, text='Please select the spectrum axis units:', anchor="w")
        self.spectralaxisunitslabel.grid(sticky="W", in_=self.spectraframe, row=5, column=1, columnspan=6)
        tk.Radiobutton(self.spectraframe, text='Hz', variable=self.SpectralAxisUnitVar, value=1, command=self.__getSpectrumAxisUnit).grid(sticky="W", in_=self.spectraframe, row=6, column=1)
        tk.Radiobutton(self.spectraframe, text='kHz', variable=self.SpectralAxisUnitVar, value=2, command=self.__getSpectrumAxisUnit).grid(sticky="W", in_=self.spectraframe, row=6, column=2)
        tk.Radiobutton(self.spectraframe, text='MHz', variable=self.SpectralAxisUnitVar, value=3, command=self.__getSpectrumAxisUnit).grid(sticky="W", in_=self.spectraframe, row=6, column=3)
        tk.Radiobutton(self.spectraframe, text='m/s', variable=self.SpectralAxisUnitVar, value=4, command=self.__getSpectrumAxisUnit).grid(sticky="W", in_=self.spectraframe, row=6, column=4)
        tk.Radiobutton(self.spectraframe, text='km/s', variable=self.SpectralAxisUnitVar, value=5, command=self.__getSpectrumAxisUnit).grid(sticky="W", in_=self.spectraframe, row=6, column=5)
        self.velocitydeflab = tk.Label(self.spectraframe, text='Velocity definition:', anchor="w")
        self.veliciydefopt = tk.Radiobutton(self.spectraframe, text='Optical', variable=self.VelocityTypeVar, value="optical", command=self.__getVelocitydef)
        self.veliciydefrad = tk.Radiobutton(self.spectraframe, text='Radio', variable=self.VelocityTypeVar, value="radio", command=self.__getVelocitydef)

        tk.Label(self.spectraframe, text='Please enter the input spectra channelwidth:', anchor="w").grid(sticky="W", in_=self.spectraframe, column=1, row=8, columnspan=4)
        self.chanwidthentry = tk.Entry(self.spectraframe, textvariable=self.ChannelWidthVar, width=8, background='lightgray')
        self.chanwidthlabel = tk.Label(self.spectraframe, text='units', anchor="w")
        self.chanwidthentry.grid(sticky="W", in_=self.spectraframe, row=8, column=5)
        self.chanwidthlabel.grid(sticky="W", in_=self.spectraframe, row=8, column=6)
        self.chanwidthentry.bind("<Return>", self.__getChannelWidth)


        self.canvas = tk.Canvas( master=self.mastercanvas, width=50)
        self.canvas.create_line(40,0,40,180)
        self.canvas.grid(in_=self.spectraframe, row=2, column=7, rowspan=7)

        #### Spectra file info
        tk.Label(self.spectraframe, text='Input spectra file info:', anchor="w", font=(None, 14, "underline")).grid(sticky="W", in_=self.spectraframe, row=2, column=8, columnspan=3)
        tk.Label(self.spectraframe, text='First row of data:').grid(sticky="W", in_=self.spectraframe, row=3, column=8, columnspan=2)
        self.firstrowentry = tk.Entry(self.spectraframe, textvariable=self.FirstRowInSpectrumVar, width=3, background='lightgray')
        self.firstrowentry.grid(sticky="W", in_=self.spectraframe, row=3, column=10, columnspan=1)
        self.firstrowentry.bind("<Return>", self.__getFirstRow)
        tk.Label(self.spectraframe, text='Spectrum Data Columns:').grid(sticky="W", in_=self.spectraframe, row=4, column=8, columnspan=4)
        tk.Label(self.spectraframe, text='Axis:').grid(sticky="W", in_=self.spectraframe, row=5, column=8, columnspan=1)
        tk.Label(self.spectraframe, text='Flux:').grid(sticky="W", in_=self.spectraframe, row=5, column=10, columnspan=1)
        self.axiscolumnentry = tk.Entry(self.spectraframe, textvariable=self.AxisColumnVar, width=3, bg='lightgray')
        self.axiscolumnentry.grid(sticky="W", in_=self.spectraframe, row=5, column=9, columnspan=1)
        self.axiscolumnentry.bind('<Return>', self.__getAxisColumn)
        self.fluxcolumnentry = tk.Entry(self.spectraframe, textvariable=self.FluxColumnVar, width=3, bg='lightgray')
        self.fluxcolumnentry.grid(sticky="W", in_=self.spectraframe, row=5, column=11, columnspan=1)
        self.fluxcolumnentry.bind('<Return>', self.__getFluxColumn)
        tk.Label(self.spectraframe, text='Data delimiter:').grid(sticky="W", in_=self.spectraframe, row=6, column=8, columnspan=2)
        self.rowdelimentry = tk.Entry(self.spectraframe, textvariable=self.RowDelimiterVar, width=3, bg='lightgray')
        self.rowdelimentry.grid(sticky="W", in_=self.spectraframe, row=6, column=10)
        self.rowdelimentry.bind('<Return>', self.__getRowDelimiter)

        tk.Label(self.spectraframe, text='NOTE: the spectrum files should be in text or csv format.', fg='seagreen', font=(None, 14, "")).grid(sticky="NW", in_=self.spectraframe, row=3, column=14)
        


        ## Stacked Spectrum frame
        self.stackframe = tk.Frame(master=self.mastercanvas, bd=2, relief=tk.GROOVE)
        tk.Label(self.stackframe, text='Stacked Spectrum Information:', anchor=tk.W, font=(None, 14, "bold underline")).grid(sticky="W", in_=self.stackframe, row=1, column=1, columnspan=4)
        self.stackunitlabel = tk.Label(self.stackframe, text='Please select units in which to stack:')
        self.stackunitlabel.grid(sticky="W", in_=self.stackframe, column=1, row=2, columnspan=4)
        tk.Radiobutton(self.stackframe, text='Jy', variable=self.StackFluxUnitVar, value=1, command=self.__getStackFluxUnit).grid(sticky="W", in_=self.stackframe, column=5, row=2, columnspan=1)
        tk.Radiobutton(self.stackframe, text='M_HI', variable=self.StackFluxUnitVar, value=2, command=self.__getStackFluxUnit).grid(sticky="W", in_=self.stackframe, column=6, row=2, columnspan=1)
        tk.Radiobutton(self.stackframe, text='M_HI/M_star', variable=self.StackFluxUnitVar, value=3, command=self.__getStackFluxUnit).grid(sticky="W", in_=self.stackframe, column=7, row=2, columnspan=1)

        tk.Label(self.stackframe, text='Velocity width of spectrum:').grid(sticky="W", in_=self.stackframe, column=1, row=3, columnspan=2)
        self.stackspeclenentry = tk.Entry(self.stackframe, textvariable=self.StackedSpectrumLengthVar, width=5, bg='lightgray')
        self.stackspeclenentry.grid(sticky="W", in_=self.stackframe, column=3, row=3)
        self.stackspeclenentry.bind('<Return>', self.__getStackSpectrumLength)
        tk.Label(self.stackframe, text='km/s').grid(sticky="W", in_=self.stackframe, column=4, row=3)
        tk.Label(self.stackframe, text='Max galaxy velocity width of sample:').grid(sticky="W", in_=self.stackframe, column=1, row=4, columnspan=3)
        self.galaxywidthentry = tk.Entry(self.stackframe, textvariable=self.GalaxyWidthVar, width=5, bg='lightgray')
        self.galaxywidthentry.grid(sticky="W", in_=self.stackframe, column=4, row=4)
        self.galaxywidthentry.bind('<Return>', self.__getGalaxyWidth)
        tk.Label(self.stackframe, text='km/s').grid(sticky="W", in_=self.stackframe, column=5, row=4)

        tk.Label(self.stackframe, text='Cosmology (Flat Lambda CDM):').grid(sticky="W", in_=self.stackframe, column=1, row=5, columnspan=3)
        tk.Label(self.stackframe, text='H0 =').grid(sticky='W', in_=self.stackframe, column=4, row=5)
        tk.Label(self.stackframe, text='km/s/Mpc').grid(sticky="W", in_=self.stackframe, column=6, row=5)
        self.h0entry = tk.Entry(self.stackframe, textvariable=self.H0Var, width=5, bg='lightgray')
        self.h0entry.grid(sticky="W", in_=self.stackframe, column=5, row=5)
        self.h0entry.bind('<Return>', self.__getH0)
        tk.Label(self.stackframe, text='Om0 =').grid(sticky='W', in_=self.stackframe, column=4, row=6)
        self.om0entry = tk.Entry(self.stackframe, textvariable=self.Om0Var, width=5, bg='lightgray')
        self.om0entry.grid(sticky="W", in_=self.stackframe, column=5, row=6)
        self.om0entry.bind('<Return>', self.__getOm0)

        self.redshiftlabel = tk.Label(self.stackframe, text='Redshift range of sample:')
        self.redshiftminentry = tk.Entry(self.stackframe, textvariable=self.ZminVar, width=5, bg='lightgray')
        self.redshiftlabel2 = tk.Label(self.stackframe, text='< z <')
        self.redshiftmaxentry = tk.Entry(self.stackframe, textvariable=self.ZmaxVar, width=5, bg='lightgray')
        self.redshiftlabel.grid(sticky="W", in_=self.stackframe, column=1, row=7, columnspan=2)
        self.redshiftminentry.grid(sticky="W", in_=self.stackframe, column=3, row=7)
        self.redshiftlabel2.grid(in_=self.stackframe, column=4, row=7)
        self.redshiftmaxentry.grid(sticky="W", in_=self.stackframe, column=5, row=7)
        self.redshiftmaxentry.bind('<Return>', self.__getRedshifts)
        self.redshiftminentry.bind('<Return>', self.__getRedshifts)

        self.weightoptionlabel = tk.Label(self.stackframe, text='Please select weighting function:')
        self.weightoptionlabel.grid(sticky="W", in_=self.stackframe, column=1, row=8, columnspan=3)
        tk.Radiobutton(self.stackframe, text='W = 1', value=1, variable=self.WeightOptionVar, command=self.__getWeightOption).grid(sticky="W", in_=self.stackframe, column=4, row=8)
        tk.Radiobutton(self.stackframe, text='W = 1/rms', value=2, variable=self.WeightOptionVar, command=self.__getWeightOption).grid(sticky="W", in_=self.stackframe, column=5, row=8)
        tk.Radiobutton(self.stackframe, text='W = 1/rms^2', value=3, variable=self.WeightOptionVar, command=self.__getWeightOption).grid(sticky="W", in_=self.stackframe, column=6, row=8)
        tk.Radiobutton(self.stackframe, text='W = 1/(rms Dl^2)^2', value=4, variable=self.WeightOptionVar, command=self.__getWeightOption).grid(sticky="W", in_=self.stackframe, column=7, row=8)

        tk.Label(self.stackframe, text='Stack entire catalogue?').grid(sticky='W', in_=self.stackframe, row=9, column=1, columnspan=2)
        self.stackentirecatY = tk.Radiobutton(self.stackframe,text='Y', variable=self.StackEntireCatalogueYNVar, value='y', state=tk.DISABLED, command=self.__getNumberStackObjects)
        self.stackentirecatN = tk.Radiobutton(self.stackframe,text='N', variable=self.StackEntireCatalogueYNVar, value='n', state=tk.DISABLED, command=self.__getNumberStackObjects)
        self.stackentirecatLabel = tk.Label(self.stackframe, text='How many objects?')
        self.stackentirecatSpinbox = tk.Spinbox(self.stackframe, from_=5, to=100, increment=1, textvariable=self.NumberCatalogueObjectsVar, width=3, command=self.__getNumberObjects)

        self.stackentirecatY.grid(sticky='W', in_=self.stackframe, row=9, column=3)
        self.stackentirecatN.grid(sticky='W', in_=self.stackframe, row=9, column=4)


        tk.Label(self.stackframe, text='Stack in different bins?').grid(sticky='W', in_=self.stackframe, row=10, column=1, columnspan=2)
        self.bincatY = tk.Radiobutton(self.stackframe, text='Y', variable=self.BinYNVar, value='y', state=tk.DISABLED, command=self.__getBinYN)
        self.bincatN = tk.Radiobutton(self.stackframe, text='N', variable=self.BinYNVar, value='n', state=tk.DISABLED, command=self.__getBinYN)
        self.bincatY.grid(sticky='W', in_=self.stackframe, row=10, column=3)
        self.bincatN.grid(sticky='W', in_=self.stackframe, row=10, column=4)

        self.bincanvas = tk.Canvas( master=self.mastercanvas, width=15)
        self.bincanvas.create_line(10,0,10,300)
        self.listbox = tk.OptionMenu(self.stackframe, self.BinColumnVar, self.columnlist, command=self.__getBinColumn)
        self.bininfolabel1 = tk.Label(self.stackframe, text='Bin Information:')
        self.bininfolabel2 = tk.Label(self.stackframe, text='Select bin column:')
        self.binstartlabel = tk.Label(self.stackframe, text='Enter start of bin range:')
        self.binendlabel = tk.Label(self.stackframe, text='Enter end of bin range:')
        self.binsizelabel = tk.Label(self.stackframe, text='Enter bin width:')
        self.displaybinlabel = tk.Label(self.stackframe, text='These are the bins (you may edit them):')
        self.binstartentry = tk.Entry(self.stackframe, textvariable=self.BinStartVar, width=5, bg='lightgray')
        self.binendentry = tk.Entry(self.stackframe, textvariable=self.BinEndVar, width=5, bg='lightgray')
        self.binsizeentry = tk.Entry(self.stackframe, textvariable=self.BinSizeVar, width=5, bg='lightgray')
        self.bindisplay = tk.Entry(self.stackframe, textvariable=self.BinsVar, width=30, bg='lightgray', state=tk.DISABLED)
        self.binstartentry.bind('<Return>', self.__getBinStart)
        self.binendentry.bind('<Return>', self.__getBinEnd)
        self.binsizeentry.bind('<Return>', self.__getBinSize)
        self.bindisplay.bind('<Return>', self.__getBins)
        
        tk.Label(self.stackframe, text='Calculate Uncertainties?').grid(sticky='W', in_=self.stackframe, row=11, column=1, columnspan=2)
        self.uncertY = tk.Radiobutton(self.stackframe, text='Y', variable=self.UncertaintyYNVar, value='y', command=self.__getUncertainty)
        self.uncertN = tk.Radiobutton(self.stackframe, text='N', variable=self.UncertaintyYNVar, value='n', command=self.__getUncertainty)
        self.uncertMethodL = tk.Label(self.stackframe, text='Please select uncertainty calculation method:')
        self.uncertMethodR = tk.Radiobutton(self.stackframe, text='Redshift', variable=self.UncertaintyMethodVar, value='redshift', command=self.__getUncertaintyMethod)
        self.uncertMethodD = tk.Radiobutton(self.stackframe, text='Delete A Group Jackknife', variable=self.UncertaintyMethodVar, value='dagjk', command=self.__getUncertaintyMethod)
        self.uncertY.grid(sticky='W', in_=self.stackframe, row=11, column=3)
        self.uncertN.grid(sticky='W', in_=self.stackframe, row=11, column=4)


        ## placing everything  
        self.locationframe.pack(side=tk.TOP, expand=tk.NO, fill=tk.X, pady=5, padx=10, in_=self.frame)
        self.catalogueframe.pack(side=tk.TOP, expand=tk.NO, fill=tk.X, pady=5, padx=10, in_=self.frame)
        self.spectraframe.pack(side=tk.TOP, expand=tk.NO, fill=tk.X, pady=5, padx=10, in_=self.frame)
        self.stackframe.pack(side=tk.TOP, expand=tk.NO, fill=tk.X, pady=5, padx=10, in_=self.frame)


        self.frame.bind("<Configure>", self.mastercanvas.configure(scrollregion=self.mastercanvas.bbox("all")))
        tk.Button(self.root, text='Abort', background='orangered', width=5, command=sys.exit).pack(side=tk.RIGHT)
        tk.Button(self.root, text='Stack', background='limegreen', width=5, command=self.__checkvalues).pack(side=tk.RIGHT)
        self.pack(expand=tk.YES, fill=tk.BOTH) 
        return

    
    def __getOutputLocation(self):
        try:
            self.OutputLocation = tkf.askdirectory(initialdir='./')
            if len(self.OutputLocation) > 1:
                self.outlocentry.config(bg='lightgreen')
                self.OutputLocationVar.set(self.OutputLocation)
                if self.OutputLocation[-1] != '/':
                    self.OutputLocation += '/'
                else:
                    h = 0
            else:
                self.outlocentry.config(bg='tomato')
                self.OutputLocation = './'
        except AttributeError:
            self.outlocentry.config(bg='tomato')
            self.OutputLocation = './'
        return

    def __getSpectrumLocation(self):
        try:
            self.SpectrumLocation = tkf.askdirectory(initialdir='./')
            if len(self.SpectrumLocation) > 1:
                self.speclocentry.config(bg='lightgreen')
                self.SpectrumLocationVar.set(self.SpectrumLocation)
            else:
                self.speclocentry.config(bg='tomato')
                self.SpectrumLocation = None
        except AttributeError:
            self.speclocentry.config(bg='tomato')
            self.SpectrumLocation = None
        return

    def __getCatalalogueFilename(self):
        if self.SpectrumLocation != None:
            initialdir = self.SpectrumLocation
        else:
            initialdir = './'
        self.CatalogueFilename = tkf.askopenfilename(title='Select catalogue file', initialdir=initialdir)
        try:
            self.catalogue = astasc.read(self.CatalogueFilename)
            if os.path.exists(self.CatalogueFilename) is True:
                self.CatalogueFilenameEntry.config(background='lightgreen')
                self.CatalogueFilenameVar.set(self.CatalogueFilename)
                
                ## activating the catalogue questions
                self.stackentirecatY.config(state=tk.NORMAL)
                self.stackentirecatN.config(state=tk.NORMAL)

                self.bincatY.config(state=tk.NORMAL)
                self.bincatN.config(state=tk.NORMAL)

                self.__showCatalogue()
                self.__getCatalogueColumnNumbers()
            else:
                self.CatalogueFilenameEntry.config(background='tomato')
                self.CatalogueFilename = None
                self.CatalogueFilenameVar.set('')
        except:
            self.CatalogueFilenameEntry.config(background='tomato')
            self.CatalogueFilename = None
            self.CatalogueFilenameVar.set('')
            PrintException()
        return

    def __getCatalogueColumnNumbers(self):
        """ This method checks that the column number is in the catalogue, highlights the 
        column in the display window and changes the cell to green if the column exists,
        otherwise changes to red. """
        self.catalogueobjid.config(state=tk.NORMAL, textvariable=self.ObjectIDVar)
        self.cataloguefilename.config(state=tk.NORMAL, textvariable=self.FilenameVar)
        self.catalogueredshift.config(state=tk.NORMAL, textvariable=self.RedshiftVar)
        self.catalogueredshifterr.config(state=tk.NORMAL, textvariable=self.RedshiftErrorVar)
        self.cataloguestellarmass.config(state=tk.NORMAL, textvariable=self.StellarmassVar)
        self.catalogueotherdata.config(state=tk.NORMAL, textvariable=self.OtherDataVar)

        self.catalogueobjid.bind("<Return>", self.__getValidateColumnNumbersObjID)
        self.cataloguefilename.bind("<Return>", self.__getValidateColumnNumbersFilename)
        self.catalogueredshift.bind("<Return>", self.__getValidateColumnNumbersRedshift)
        self.catalogueredshifterr.bind("<Return>", self.__getValidateColumnNumbersRedshiftErr)
        self.cataloguestellarmass.bind("<Return>", self.__getValidateColumnNumbersStellarMass)
        self.catalogueotherdata.bind("<Return>", self.__getValidateColumnNumbersOtherData)
        return

    def __getValidateColumnNumbersObjID(self, event):
        self.CatalogueColumnNumbers[0] = self.ObjectIDVar.get()-1
        if (self.CatalogueColumnNumbers[0] >= 0) and (self.CatalogueColumnNumbers[0] < len(self.catalogue.colnames)):
            self.catalogueobjid.config(background=r'plum')
            self.columnlist.append('Object ID')
            self.__showCatalogue()
        else:
            self.CatalogueColumnNumbers[0] = None
            self.catalogueobjid.config(background='tomato')
            self.__showCatalogue()
        return 

    def __getValidateColumnNumbersFilename(self, event):
        self.CatalogueColumnNumbers[1] = self.FilenameVar.get()-1
        if (self.CatalogueColumnNumbers[1] >= 0) and (self.CatalogueColumnNumbers[1] < len(self.catalogue.colnames)):
            self.cataloguefilename.config(background='paleturquoise')
            self.columnlist.append('Filename')
            self.__showCatalogue() 
        else:
            self.CatalogueColumnNumbers[1] = None
            self.cataloguefilename.config(background='tomato')
            self.__showCatalogue()
        return 

    def __getValidateColumnNumbersRedshift(self, event):
        self.CatalogueColumnNumbers[2] = self.RedshiftVar.get()-1
        if (self.CatalogueColumnNumbers[2] >= 0) and (self.CatalogueColumnNumbers[2] < len(self.catalogue.colnames)):
            self.catalogueredshift.config(background='hotpink')
            self.columnlist.append('Redshift')
            self.__showCatalogue() 
        else:
            self.CatalogueColumnNumbers[2] = None
            self.catalogueredshift.config(background='tomato')
            self.__showCatalogue()
        return 

    def __getValidateColumnNumbersRedshiftErr(self, event):
        self.CatalogueColumnNumbers[3] = self.RedshiftErrorVar.get()-1
        if (self.CatalogueColumnNumbers[3] >= 0) and (self.CatalogueColumnNumbers[3] < len(self.catalogue.colnames)):
            self.catalogueredshifterr.config(background='darkseagreen')
            self.columnlist.append('Redshift Error')
            self.__showCatalogue() 
        else:
            self.CatalogueColumnNumbers[3] = None
            self.catalogueredshifterr.config(background='tomato')
            self.__showCatalogue()
        return 

    def __getValidateColumnNumbersStellarMass(self, event):
        self.CatalogueColumnNumbers[4] = self.StellarmassVar.get()-1
        if (self.CatalogueColumnNumbers[4] > 0) and (self.CatalogueColumnNumbers[4] < len(self.catalogue.colnames)):
            self.cataloguestellarmass.config(background='cornsilk')
            self.columnlist.append('Stellar Mass')
            self.__showCatalogue() 
        else:
            self.CatalogueColumnNumbers[4] = -1
            self.cataloguestellarmass.config(background='tomato')
            self.__showCatalogue()
        return 

    def __getValidateColumnNumbersOtherData(self, event):
        self.CatalogueColumnNumbers[5] = self.OtherDataVar.get()-1
        if (self.CatalogueColumnNumbers[5] > 0) and (self.CatalogueColumnNumbers[5] < len(self.catalogue.colnames)):
            self.catalogueotherdata.config(background='skyblue')
            self.columnlist.append('Other Data')
            self.__showCatalogue() 
        else:
            self.CatalogueColumnNumbers[5] = -1
            self.catalogueotherdata.config(background='tomato')
            self.__showCatalogue()
        return 

    def __showCatalogue(self):
        tab = self.catalogue[:5]

        colours = [np.unicode_('plum'), np.unicode_('paleturquoise'), np.unicode_('hotpink'),np.unicode_('darkseagreen'), np.unicode_('cornsilk'), np.unicode_('skyblue')]
        tabcolors = [np.unicode_('w')]*len(self.catalogue.colnames)
        colind = np.where(np.array(self.CatalogueColumnNumbers) >= 0)[0]
        cellcolours = np.chararray((5, len(self.catalogue.colnames)), itemsize=20, unicode=True)
        cellcolours[:] = np.unicode_('white')
        for i in range(len(colind)):
            j = colind[i]
            k = self.CatalogueColumnNumbers[j]
            tabcolors[k] = np.unicode_(colours[j])
            for n in range(5):
                cellcolours[n,k] = np.unicode_(colours[j])
        try:
            plt.cla()           
            thetable = self.ax.table(cellText=tab.as_array(), colLabels=tab.colnames, colColours=tabcolors, cellColours=cellcolours, loc='center')#
            thetable.set_fontsize(10)
            thetable.auto_set_font_size(False)
            self.ax.axis('off')
            self.canvasplot.draw()
        except:
            # print(cellcolours)
            PrintException()
        return

    def __getPixelsize(self, event):
        try:
            self.PixelSize = self.PixelSizeVar.get()
            self.pixsizeentry.config(background='lightgreen')
        except ValueError:
            self.pixsizeentry.config(bg='tomato')
        return

    def __getBeamsize(self, event):
        try:
            self.BeamSize = self.BeamSizeVar.get()
            self.beamsizeentry.config(bg='lightgreen')
        except ValueError:
            self.beamsizeentry.config(bg='tomato')
        return

    def __getSpectrumFluxUnit(self):
        self.SpectrumFluxUnit = self.SpectrumFluxUnitVar.get()
        self.spectrumfluxunitlabel.config(bg='white')
        if self.SpectrumFluxUnit > 3:
            self.pixsizelabel.grid(sticky='W', in_=self.spectraframe, row=4, column=1, columnspan=2)
            self.beamsizelabel.grid(sticky='W', in_=self.spectraframe, row=4, column=4, columnspan=2)
            self.pixsizeentry.grid(sticky="W", in_=self.spectraframe, row=4, column=3)
            self.beamsizeentry.grid(sticky="W", in_=self.spectraframe, row=4, column=6)
        else:
            self.pixsizelabel.grid_forget()
            self.beamsizelabel.grid_forget()
            self.pixsizeentry.grid_forget()
            self.beamsizeentry.grid_forget()
        return

    def __getVelocitydef(self):
        self.VelocityType = self.VelocityTypeVar.get()
        return

    def __getChannelWidth(self, event):
        try:
            self.ChannelWidth = self.ChannelWidthVar.get()
            self.chanwidthentry.config(bg='lightgreen')
        except ValueError:
            self.chanwidthentry.config(bg='tomato')
        return

    def __getSpectrumAxisUnit(self):
        self.SpectralAxisUnit = self.SpectralAxisUnitVar.get()
        self.spectralaxisunitslabel.config(bg='white')
        if self.SpectralAxisUnit > 3:
            self.velocitydeflab.grid(sticky="W", in_=self.spectraframe, row=7, column=1, columnspan=2)
            self.veliciydefopt.grid(sticky="W", in_=self.spectraframe, row=7, column=3)
            self.veliciydefrad.grid(sticky="W", in_=self.spectraframe, row=7, column=4)
        else:
           self.velocitydeflab.grid_forget()
           self.veliciydefopt.grid_forget()
           self.veliciydefrad.grid_forget()
        units = {1: "Hz", 2: "kHz", 3: "MHz", 4: "m/s", 5: "km/s"}
        self.chanwidthlabel.config(text='%s'%units[self.SpectralAxisUnit])
        return

    def __getFirstRow(self, event):
        try:
            self.FirstRowInSpectrum = self.FirstRowInSpectrumVar.get()
            self.firstrowentry.config(bg='lightgreen')
        except ValueError:
            self.firstrowentry.config(bg='tomato')
        return

    def __getAxisColumn(self, event):
        try:
            self.SpectrumColumns[0] = self.AxisColumnVar.get()
            self.axiscolumnentry.config(bg='lightgreen')
        except ValueError:
            self.axiscolumnentry.config(bg='tomato')
        return

    def __getFluxColumn(self, event):
        try:
            self.SpectrumColumns[1] = self.FluxColumnVar.get()
            self.fluxcolumnentry.config(bg='lightgreen')
        except ValueError:
            self.fluxcolumnentry.config(bg='tomato')
        return

    def __getRowDelimiter(self, event):
        try:
            self.RowDelimiter = self.RowDelimiterVar.get()
            if len(self.RowDelimiter) == 0:
                self.RowDelimiter = "none"
            self.rowdelimentry.config(bg='lightgreen')
        except ValueError:
            self.rowdelimentry.config(bg='tomato')
        return

    def __getStackFluxUnit(self):
        """ This function will need to check that there is a stellar mass
        column in the catalogue, if none has been entered, the stellar mass
        entry must turn red. """
        self.StackFluxUnit = self.StackFluxUnitVar.get()
        self.stackunitlabel.config(bg='white')
        return

    def __getH0(self, event):
        try:
            self.H0 = self.H0Var.get()
            self.h0entry.config(bg='lightgreen')
        except ValueError:
            self.h0entry.config(bg='tomato')
        return

    def __getOm0(self, event):
        try:
            self.Om0 = self.Om0Var.get()
            self.om0entry.config(bg='lightgreen')
        except ValueError:
            self.om0entry.config(bg='tomato')
        return

    def __getGalaxyWidth(self, event):
        try:
            self.GalaxyWidth = self.GalaxyWidthVar.get()
            self.galaxywidthentry.config(bg='lightgreen')
        except ValueError:
            self.galaxywidthentry.config(bg='tomato')
        return

    def __getStackSpectrumLength(self, event):
        try:
            self.StackedSpectrumLength = self.StackedSpectrumLengthVar.get()
            if (self.StackedSpectrumLength < 2.5*self.GalaxyWidthVar.get()) and (self.StackedSpectrumLength > self.GalaxyWidthVar.get()):
                self.stackspeclenentry.config(bg='orange')
            elif (self.StackedSpectrumLength < 1.5*self.GalaxyWidthVar.get()):
                self.stackspeclenentry.config(bg='tomato')
            else:
                self.stackspeclenentry.config(bg='lightgreen')
        except ValueError:
            self.stackspeclenentry.config(bg='tomato')
        return

    def __getRedshifts(self, event):
        try:
            self.Zmin = self.ZminVar.get()
            self.Zmax = self.ZmaxVar.get()
            if self.Zmin > self.Zmax:
                self.redshiftmaxentry.config(bg='tomato')
                self.redshiftminentry.config(bg='orange')
            else:
                self.redshiftminentry.config(bg='lightgreen')
                self.redshiftmaxentry.config(bg='lightgreen')
        except ValueError:
            self.redshiftmaxentry.config(bg='tomato')
            self.redshiftminentry.config(bg='tomato')
        return

    def __getWeightOption(self):
        self.WeightOption = self.WeightOptionVar.get()
        return

    def __getNumberObjects(self):
        self.NumberCatalogueObjects = self.NumberCatalogueObjectsVar.get()
        return

    def __getNumberStackObjects(self):
        self.StackEntireCatalogueYN = self.StackEntireCatalogueYNVar.get()
        self.stackentirecatLabel.config(bg='white')
        if self.StackEntireCatalogueYN == 'n':
            self.stackentirecatSpinbox.config(to=len(self.catalogue))
            self.stackentirecatLabel.grid(sticky='W', in_=self.stackframe, row=9, column=5, columnspan=2)
            self.stackentirecatSpinbox.grid(sticky='W', in_=self.stackframe, row=9, column=7)
        else:
            self.stackentirecatLabel.grid_forget()
            self.stackentirecatSpinbox.grid_forget()
            self.NumberCatalogueObjects = len(self.catalogue)

    def __getBinStart(self, event):
        try:
            self.BinInfo[2] = self.BinStartVar.get()
            self.binstartentry.config(bg='lightgreen')
            self.bsy = 1
            if self.bsy == 1 and self.bey == 1 and self.bssy == 1:
                self.Bins = list(np.arange(self.BinInfo[2], self.BinInfo[3]+self.BinInfo[1], self.BinInfo[1]))
                self.BinsVar.set(str(self.Bins))
                self.bindisplay.config(state=tk.NORMAL, bg='lightblue')
            else:
                h = 0
        except ValueError:
            self.binstartentry.config(bg='tomato')
        return

    def __getBinEnd(self, event):
        try:
            self.BinInfo[3] = self.BinEndVar.get()
            self.binendentry.config(bg='lightgreen')
            self.bey = 1
            if self.bsy == 1 and self.bey == 1 and self.bssy == 1:
                self.Bins = list(np.arange(self.BinInfo[2], self.BinInfo[3]+self.BinInfo[1], self.BinInfo[1]))
                self.BinsVar.set(str(self.Bins))
                self.bindisplay.config(state=tk.NORMAL, bg='lightblue')
            else:
                h = 0
        except ValueError:
            self.binendentry.config(bg='tomato')
        return

    def __getBinSize(self, event):
        try:
            self.BinInfo[1] = self.BinSizeVar.get()
            self.binsizeentry.config(bg='lightgreen')
            self.bssy = 1
            if self.bsy == 1 and self.bey == 1 and self.bssy == 1:
                self.Bins = list(np.arange(self.BinInfo[2], self.BinInfo[3]+self.BinInfo[1], self.BinInfo[1]))
                self.BinsVar.set(str(self.Bins))
                self.bindisplay.config(state=tk.NORMAL, bg='lightblue')
            else:
                h = 0
        except ValueError:
            self.binsizeentry.config(bg='tomato')
        return

    def __getBins(self, event):
        try:
            self.Bins = eval(self.BinsVar.get())
            self.bindisplay.config(bg='lightgreen')
        except:
            self.bindisplay.config(bg='tomato')
        return

    def __getBinColumn(self, event):
        self.BinInfo[0] = self.BinColumnVar.get()
        return

    def __getBinYN(self):
        self.BinYN = self.BinYNVar.get()
        if self.BinYN == 'y':
            self.bincanvas.grid(in_=self.stackframe, row=2, column=8, rowspan=7)
            self.bininfolabel1.grid(sticky="W", in_=self.stackframe, row=2, column=9) # Bin Information
            self.bininfolabel2.grid(sticky="W", in_=self.stackframe, row=3, column=9) # Select bin column
            self.binstartlabel.grid(sticky="W", in_=self.stackframe, row=4, column=9) # Enter start of bin range
            self.binendlabel.grid(sticky="W", in_=self.stackframe, row=5, column=9) # Enter end of bin range
            self.binsizelabel.grid(sticky="W", in_=self.stackframe, row=6, column=9) # Enter bin width
            self.displaybinlabel.grid(sticky="W", in_=self.stackframe, row=7, column=9, columnspan=2) # These are the bins (you may edit them)

            self.columnlist = list(set(self.columnlist))
            self.listbox = tk.OptionMenu(self.stackframe, self.BinColumnVar, *self.columnlist, command=self.__getBinColumn)
            self.listbox.grid(sticky="W", in_=self.stackframe, row=3, column=10)

            self.binstartentry.grid(sticky="W", in_=self.stackframe, row=4, column=10)
            self.binendentry.grid(sticky="W", in_=self.stackframe, row=5, column=10)
            self.binsizeentry.grid(sticky="W", in_=self.stackframe, row=6, column=10)
            self.bindisplay.grid(sticky="W", in_=self.stackframe, row=8, column=9, columnspan=2)

        else:
            self.bincanvas.grid_forget()
            self.listbox.grid_forget()
            self.bininfolabel1.grid_forget()
            self.bininfolabel2.grid_forget()
            self.binstartlabel.grid_forget()
            self.binendlabel.grid_forget()
            self.binsizelabel.grid_forget()
            self.displaybinlabel.grid_forget()
            self.binstartentry.grid_forget()
            self.binendentry.grid_forget()
            self.binsizeentry.grid_forget()
            self.bindisplay.grid_forget()
        return
    
    def __getUncertainty(self):
        self.UncertaintyYN = self.UncertaintyYNVar.get()
        if self.UncertaintyYN == 'y':
            self.uncertMethodL.grid(sticky="W", in_=self.stackframe, row=12, column=2, columnspan=3)
            self.uncertMethodR.grid(sticky="W", in_=self.stackframe, row=12, column=5, columnspan=1)
            self.uncertMethodD.grid(sticky="W", in_=self.stackframe, row=12, column=6, columnspan=3)
        else:
            self.uncertMethodL.grid_forget()
            self.uncertMethodR.grid_forget()
            self.uncertMethodD.grid_forget()
        return

    def __getUncertaintyMethod(self):
        self.UncertaintyMethod = self.UncertaintyMethodVar.get()
        self.uncertMethodL.config(bg='white')
        return

  
    def __checkvalues(self):
        go = []
        if self.CatalogueFilename is None or self.CatalogueFilename == '':
            self.CatalogueFilenameEntry.config(background='tomato')
            go.append(False)
        else:
            go.append(True)

        if any(c is None for c in self.CatalogueColumnNumbers):
            if self.CatalogueColumnNumbers[0] is None:
                self.catalogueobjid.config(background='tomato')
                go.append(False)
            else:
                go.append(True)
            if self.CatalogueColumnNumbers[1] is None:
                self.cataloguefilename.config(background='tomato')
                go.append(False)
            else:
                go.append(True)
            if self.CatalogueColumnNumbers[2] is None:
                self.catalogueredshift.config(background='tomato')
                go.append(False)
            else:
                go.append(True)
            if self.CatalogueColumnNumbers[3] is None:
                self.CatalogueColumnNumbers[3] = ""
                go.append(True)
            else:
                go.append(True)
            if self.CatalogueColumnNumbers[4] is None:
                if self.StackFluxUnit == 3:
                    self.cataloguestellarmass.config(background='tomato')
                    go.append(False)
                else:
                    self.CatalogueColumnNumbers[4] = ""
                    go.append(True)
            else:
                go.append(True)
            if self.CatalogueColumnNumbers[5] is None:
                self.CatalogueColumnNumbers[5] = ""
                go.append(True)
            else:
                go.append(True)
        else:
            go.append(True)

        if self.SpectrumLocation is None:
            self.speclocentry.config(bg='tomato')
            go.append(False)
        else:
            go.append(True)

        if self.CatalogueFilename is None:
            self.CatalogueFilenameEntry.config(background='tomato')
            go.append(False)
        else:
            go.append(True)

        if self.SpectrumFluxUnit is None:
            self.spectrumfluxunitlabel.config(bg='tomato')
            go.append(False)
        else:
            if self.SpectrumFluxUnit > 3:
                if self.PixelSize is None:
                    go.append(False)
                    self.pixsizeentry.config(bg='tomato')
                else:
                    go.append(True)
                if self.BeamSize is None:
                    go.append(False)
                    self.beamsizeentry.config(bg='tomato')
                else:
                    go.append(True)
            else:
                go.append(True)

        if self.SpectralAxisUnit is None:
            self.spectralaxisunitslabel.config(bg='tomato')
            go.append(False)
        else:
            if self.SpectralAxisUnit > 3:
                if self.VelocityType is None:
                    self.__getVelocitydef()
                    go.append(True)
                else:
                    go.append(True)
            else:
                go.append(True)

        if self.ChannelWidth is None:
            self.chanwidthentry.config(bg='tomato')
            go.append(False)
        else:
            go.append(True)

        if self.FirstRowInSpectrum is None:
            go.append(False)
            self.firstrowentry.config(bg='tomato')
        else:
            go.append(True)

        if self.RowDelimiter == "":
            self.RowDelimiter = "none"
            go.append(True)
        else:
            go.append(True)

        if any(c is None for c in self.SpectrumColumns):
            go.append(False)
            if self.SpectrumColumns[0] is None:
                self.axiscolumnentry.config(bg='tomato')
            else:
                h = 0
            if self.SpectrumColumns[1] is None:
                self.fluxcolumnentry.config(bg='tomato')
            else:
                h = 0
        else:
            go.append(True)

        if self.StackedSpectrumLength is None:
            self.StackedSpectrumLength = self.StackedSpectrumLengthVar.get()
            go.append(True)
        else:
            go.append(True)

        if self.GalaxyWidth is None:
            self.GalaxyWidth = self.GalaxyWidthVar.get()
            go.append(True)
        else:
            go.append(True)

        if self.Zmin is None:
            self.redshiftminentry.config(bg='tomato')
            go.append(False)
        else:
            go.append(True)

        if self.Zmax is None:
            self.redshiftmaxentry.config(bg='tomato')
            go.append(False)
        else:
            go.append(True)

        if self.StackFluxUnit is None:
            self.stackunitlabel.config(bg='tomato')
            go.append(False)
        else:
            if (self.StackFluxUnit == 3) and (self.CatalogueColumnNumbers[4] is None):
                self.cataloguestellarmass.config(background='tomato')
                go.append(False)
            else:
                go.append(True)

        if self.H0 is None:
            self.H0 = self.H0Var.get()
            go.append(True)
        else:
            go.append(True)

        if self.Om0 is None:
            self.Om0 = self.Om0Var.get()
            go.append(True)
        else:
            go.append(True)

        if self.WeightOption is None:
            self.WeightOption = self.WeightOptionVar.get()
            go.append(True)
        else:
            go.append(True)

        if self.StackEntireCatalogueYN is None:
            self.StackEntireCatalogueYN = self.StackEntireCatalogueYNVar.get()
            if self.StackEntireCatalogueYN == 'n':
                if self.NumberCatalogueObjects is None:
                    self.stackentirecatLabel.config(bg='tomato')
                    go.append(False)
                else:
                    go.append(True)
            else:
                go.append(True)
        else:
            if self.StackEntireCatalogueYN == 'n':
                if self.NumberCatalogueObjects is None:
                    self.stackentirecatLabel.config(bg='tomato')
                    go.append(False)
                else:
                    go.append(True)
            else:
                go.append(True)


        if self.UncertaintyYN is None:
            self.UncertaintyYN = self.UncertaintyYNVar.get()
            if self.UncertaintyYN == 'y':
                if self.UncertaintyMethod is None:
                    self.uncertMethodL.config(bg='tomato')
                    go.append(False)
                else:
                    go.append(True)
            else:
                go.append(True)
        else:
            if self.UncertaintyYN == 'y':
                if self.UncertaintyMethod is None:
                    self.uncertMethodL.config(bg='tomato')
                    go.append(False)
                else:
                    go.append(True)
            else:
                go.append(True)


        if self.BinYN is None:
            self.BinYN = self.BinYNVar.get()
            if self.BinYN == 'y':
                if self.BinInfo[0] is None:
                    self.bininfolabel2.config(bg='tomato')
                    go.append(False)
                elif self.BinInfo[1] is None:
                    self.binsizeentry.config(bg='tomato')
                    go.append(False)
                elif self.BinInfo[2] is None:
                    self.binstartentry.config(bg='tomato')
                    go.append(False)
                elif self.BinInfo[4] is None:
                    self.binendentry.config(bg='tomato')
                    go.append(False)
                else:
                    go.append(True)
        
            else:
                go.append(True)
        else:
            if self.BinYN == 'y':
                if self.BinInfo[0] is None:
                    self.bininfolabel2.config(bg='tomato')
                    go.append(False)
                elif self.BinInfo[1] is None:
                    self.binsizeentry.config(bg='tomato')
                    go.append(False)
                elif self.BinInfo[2] is None:
                    self.binstartentry.config(bg='tomato')
                    go.append(False)
                elif self.BinInfo[4] is None:
                    self.binendentry.config(bg='tomato')
                    go.append(False)
                else:
                    go.append(True)
        
            else:
                go.append(True)

        if any(c==False for c in go):
            return
        else:
            return self.stack()


    def __dumptofile(self):

        self.CatalogueColumnNumbersUpdated = [ e for e in self.CatalogueColumnNumbers if e > 0 ]

        config = {
            'CatalogueFilename': self.CatalogueFilename,
            'CatalogueColumnNumbers': self.CatalogueColumnNumbers,
            'OutputLocation': self.OutputLocation,
            'SpectrumLocation': self.SpectrumLocation,
            'SpectrumColumns': self.SpectrumColumns,
            'FirstRowInSpectrum': self.FirstRowInSpectrum,
            'RowDelimiter': self.RowDelimiter,
            'SpectrumFluxUnit': self.SpectrumFluxUnit,
            'PixelSize': self.PixelSize,
            'BeamSize': self.BeamSize,
            'SpectralAxisUnit': self.SpectralAxisUnit,
            'VelocityType': self.VelocityType,
            'ChannelWidth': self.ChannelWidth,
            'StackedSpectrumLength': self.StackedSpectrumLength,
            'GalaxyWidth': self.GalaxyWidth,
            'StackFluxUnit': self.StackFluxUnit,
            'z_min': self.Zmin,
            'z_max': self.Zmax,
            'H0': self.H0,
            'Om0': self.Om0,
            'WeightOption': self.WeightOption,
            'StackEntireCatalogueYN': self.StackEntireCatalogueYN,
            'NumberCatalogueObjects': self.NumberCatalogueObjects,
            'UncertaintyYN': self.UncertaintyYN,
            'UncertaintyMethod': self.UncertaintyMethod,
            'BinYN': self.BinYN,
            'BinInfo': self.BinInfo,
            'Bins': self.Bins,
            }
                
        outfile =  open(self.OutputLocation+'inputinfo.json', 'w')
        configfile = json.dumps(config,
              indent=4, sort_keys=False,
              separators=(',', ': '))
        print(configfile, file=outfile)
        outfile.close()
        
        return

    def stack(self):
        self.__dumptofile()
        self.master.destroy()
        setupwindow = preparewindow(self.OutputLocation)

        return

class preparewindow(tk.Frame):
    def __init__(self, outputlocation):
        self.root = tk.Tk()
        tk.Frame.__init__(self, self.root, width=800)

        self.configfile = uf.bashfriendlypath(outputlocation+'inputinfo.json')
        self.shortcall = 'python pipeline.py -f inputinfo.json'
        self.callsign = 'python pipeline.py -f %s'%self.configfile

        self.callsignVar = tk.StringVar(self, value=self.shortcall)
        self.multiVar = tk.StringVar(self, value='')
        self.displayVar = tk.StringVar(self, value='')
        self.progressVar = tk.StringVar(self, value='')
        self.hideVar = tk.StringVar(self, value='')
        self.cleanVar = tk.StringVar(self, value='')
        self.latexVar = tk.StringVar(self, value='')
        self.var = tk.StringVar(self, value='')


        self.newframe = tk.Frame(self.root)
        self.master.title('HI Stacking Software')
        self.master.iconname('HISS')
        tk.Label(self.root, text='Preparing to stack', font=(None, 20, "bold")).pack(side=tk.TOP)

        tk.Label(self.root, text='\nPlease select any additional options. The command-line call shows the command used to call HISS should you wish call \nHISS from the command-line. Click on Go when you are ready to proceed, all further dialogue will appear \nin the command-line.\n', font=(None, 14, "italic"), fg='darkviolet', bg='whitesmoke').pack(side=tk.TOP, pady=5)

        self.instructionframe = tk.Frame(self.root, width=800, bd=2, relief=tk.GROOVE)
        label = tk.Label(self.instructionframe, text='Command-line call:')
        self.callentry = tk.Entry(self.instructionframe, width=80, bg='lightgray', textvariable=self.callsignVar)
        self.multi = tk.Checkbutton(self.instructionframe, text="Use multiprocessing module to run the uncertainty calculation.", variable=self.multiVar, onvalue='-m', offvalue='', command=self.__add_m, state=tk.DISABLED)
        self.display = tk.Checkbutton(self.instructionframe, text="Display progress window during the stacking process.", variable=self.displayVar, onvalue='-d', offvalue='', command=self.__add_d)
        self.progress = tk.Checkbutton(self.instructionframe, text="Save progress window during the stacking process.", variable=self.progressVar, onvalue='-p', offvalue='', command=self.__add_p)
        self.hide = tk.Checkbutton(self.instructionframe, text='Suppress all output windows.', variable=self.hideVar, onvalue='-s', offvalue='', command=self.__add_s)
        self.latex = tk.Checkbutton(self.instructionframe, text='Enable the use of latex formatting in the plots.', variable=self.latexVar, onvalue='-l', offvalue='', command=self.__add_l)
        self.clean = tk.Checkbutton(self.instructionframe, text='Test noiseless spectra.', variable=self.cleanVar, onvalue='-c', offvalue='', command=self.__add_c)


        label.grid(sticky="W", in_=self.instructionframe, row=1, column=1)
        self.callentry.grid(sticky="W", in_=self.instructionframe, row=1, column=3)
        self.multi.grid(sticky="W", in_=self.instructionframe, row=2, column=2, columnspan=2)
        self.display.grid(sticky="W", in_=self.instructionframe, row=3, column=2, columnspan=2)
        self.progress.grid(sticky="W", in_=self.instructionframe, row=4, column=2, columnspan=2)
        self.hide.grid(sticky="W", in_=self.instructionframe, row=5, column=2, columnspan=2)
        self.latex.grid(sticky="W", in_=self.instructionframe, row=6, column=2, columnspan=2)
        self.clean.grid(sticky="W", in_=self.instructionframe, row=7, column=2, columnspan=2)
        self.instructionframe.pack(side=tk.TOP, expand=tk.NO, fill=tk.X, pady=5, padx=10)


        tk.Button(self.root, text='Abort', background='orangered', width=5, command=sys.exit).pack(side=tk.RIGHT)
        tk.Button(self.root, text='Go', background='limegreen', width=5, command=self.callhiss).pack(side=tk.RIGHT)
        self.pack(fill=tk.X)
        return

    def __add_m(self):
        excess = self.multiVar.get()
        if excess == '':
            self.shortcall = self.shortcall.replace(" -m", "")
            self.callsign = self.callsign.replace(" -m", "")
        else:
            self.shortcall += ' %s'%excess
            self.callsign += ' %s'%excess

        self.callsignVar.set(self.shortcall)
        return

    def __add_d(self):
        excess = self.displayVar.get()
        if excess == '':
            self.shortcall = self.shortcall.replace(" -d", "")
            self.callsign = self.callsign.replace(" -d", "")
        else:
            self.shortcall += ' %s'%excess
            self.callsign += ' %s'%excess

        self.callsignVar.set(self.shortcall)
        return

    def __add_p(self):
        excess = self.progressVar.get()
        if excess == '':
            self.shortcall = self.shortcall.replace(" -p", "")
            self.callsign = self.callsign.replace(" -p", "")
        else:
            self.shortcall += ' %s'%excess
            self.callsign += ' %s'%excess

        self.callsignVar.set(self.shortcall)
        return

    def __add_s(self):
        excess = self.hideVar.get()
        if excess == '':
            self.display.config(state=tk.NORMAL)
            self.shortcall = self.shortcall.replace(" -s", "")
            self.callsign = self.callsign.replace(" -s", "")
        else:
            self.shortcall += ' %s'%excess
            self.callsign += ' %s'%excess
            self.display.config(state=tk.DISABLED)
            self.displayVar.set("")
            self.__add_d()

        self.callsignVar.set(self.shortcall)
        return

    def __add_l(self):
        excess = self.latexVar.get()
        if excess == '':
            self.shortcall = self.shortcall.replace(" -l", "")
            self.callsign = self.callsign.replace(" -l", "")
        else:
            self.shortcall += ' %s'%excess
            self.callsign += ' %s'%excess

        self.callsignVar.set(self.shortcall)
        return

    def __add_c(self):
        excess = self.cleanVar.get()
        if excess == '':
            self.shortcall = self.shortcall.replace(" -c", "")
            self.callsign = self.callsign.replace(" -c", "")
        else:
            self.shortcall += ' %s'%excess
            self.callsign += ' %s'%excess

        self.callsignVar.set(self.shortcall)
        return


    def callhiss(self):
        self.master.withdraw()
        os.system(self.callsign)
        sys.exit()
        return


if __name__ == '__main__':
    hiss().mainloop()
