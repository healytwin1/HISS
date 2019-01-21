""" This module conatains all useful functions that don't belong to a specific module or class """

from __future__ import print_function
import numpy as np
import numpy.random as npr
import astropy.constants as astcon
import os 
import astropy.modeling as astmod
from scipy.special import wofz, erf, erfc
import sys
import astropy.units as astun
import astropy.table as asttab
import logging

logger = logging.getLogger(__name__)

msun = astun.def_unit(
    s = 'msun',
    represents=astun.Msun,
    doc='Nothing yet',
    format={'latex': r'M$_\odot$'},
    prefixes=True
    )


gasfrac = astun.def_unit(
    s = 'gas fraction',
    represents=astun.Msun/astun.Msun,
    doc='Nothing yet',
    format={'latex': r'M$_\odot$/M$_\ast$'}
    )

msundex = astun.def_unit(
    s = 'log msun',
    represents = astun.Msun*astun.dex,
    doc = 'Nothing yet',
    format={'latex': r'$\log (\mathrm{M}_\odot)$'}
    )

msunconv = [
    (astun.Jy*astun.km*astun.Mpc*astun.Mpc/astun.s, astun.solMass, lambda x: 1*x, lambda x: 1*x)
]


log10conv = [
    (astun.dex*astun.Msun, astun.Msun, lambda x: 10**x, lambda x: np.log10(x))
]

def maskTable(table):
    cols = table.colnames
    for i in range(len(cols)):
        indices = np.where(table[cols[i]].data == 0.)[0]
        mask = np.array([False]*len(table))
        mask[indices] = np.array([True]*len(indices))
        table[cols[i]].mask = mask
    return table

def checkkeys(config, key):
    if config != None:
        keys = config.keys()
        if key in keys:
            return True
        else:
            return False
    else:
        return False


def convertable(table):
    colnames = table.colnames
    for i in range(1, len(colnames)):
        firstentry = table[colnames[i]][0]
        if (type(firstentry) == np.str):
            continue
        else:
            column = (table[colnames[i]].data).astype(float)
            ref = np.min(column)
        if ('RMS' in colnames[i]) or ('chi' in colnames[i]) or ('FWHM' in colnames[i]):
            for j in range(len(table)):
                table[colnames[i]][j] = round(table[colnames[i]][j], 2)
        elif (len(str(ref).split('.')[0]) > 3 and ref) > 1 or (len(str(ref).split('.')[1]) > 3 and ref < 1):
            refstr = '%.2E'%ref
            conv = eval('1E%i'%float(refstr.split('E')[1]))
            for j in range(len(table)):
                table[colnames[i]][j] = round(table[colnames[i]][j]/conv, 2)
            colnames[i] += ' (x 10^%i)'%float(refstr.split('E')[1])
        elif ('Fitted' in colnames[i]):
            for j in range(len(table)):
                table[colnames[i]][j] = np.unicode(table[colnames[i]][j])
        else:
            for j in range(len(table)):
                table[colnames[i]][j] = round(table[colnames[i]][j], 2)
        
    newtab = asttab.Table(names=colnames, data=table.as_array())
    return newtab


def bashfriendlypath(path):
    char = [' ', '-']
    if '\\' in path:
        return path
    else:
        for i in range(len(char)):
            path = path.replace(char[i], '\\%s'%char[i])
        return path


def exit(cat):
    outloc = bashfriendlypath(cat.outloc)
    if os.path.exists('%sstackertemp/'%cat.outloc):
        os.system('rm -r %sstackertemp/'%outloc)
    else:
        h = 0
    if os.path.exists('hiss_%s.log'%(cat.runtime)):
        logger.info('Stacker has exited')
        os.system('mv hiss_%s.log %shiss_%s.log'%(cat.runtime, outloc, cat.runtime))
    else:
        h = 0
    sys.exit()
    return


def earlyexit(cat):
    logger.critical('Exited early.')
    outloc = bashfriendlypath(cat.outloc)
    if os.path.exists('%sstackertemp/'%cat.outloc):
        os.system('rm -r %sstackertemp/'%outloc)
    else:
        h = 0
    if os.path.exists('hiss_%s.log'%(cat.runtime)):
        os.system('mv hiss_%s.log %shiss_%s.log'%(cat.runtime, outloc, cat.runtime))
    else:
        h = 0
    print("\nExiting early due to an error, please check the \nlog file in the output location for more information\n")
    sys.exit()
    return
    

def latexexponent(value):
    masstext = '%.2E'%value
    masslist = masstext.split('E')
    if len(masslist) > 1:
        masexp = masstext.split('E')[1]
    else:
        masexp = '0'
    masb = masstext.split('E')[0]
    lstr = r'$%s \times 10^{%i}$'%(masb, float(masexp))
    return lstr


def latextablestr(table, cat):
    tablestr = r' '
    for colnm in range(len(table.colnames)):
        tablestr += r'\textbf{%s}'% table.colnames[colnm]
        if colnm == len(table.colnames)-1:
            tablestr += r' \\'
        else:
            tablestr += r' & '
    for colnm in range(len(table.colnames)):
        if str(table[table.colnames[colnm]].unit) == 'None':
            unit = ''
        elif 'Flux' in table.colnames[colnm]:
            if cat.stackunit == msun:
                unit = r'(M$_\odot$)'
            elif cat.stackunit == gasfrac:
                unit = r'(M$_\mathrm{HI}$/M$_\ast$)'
            else:
                unit = '(Jy km/s)'
        elif np.unicode(table.colnames[colnm]) == np.unicode('FWHM'):
            unit = '(km/s)'
        else:
            unit = ''
        tablestr += r'%s'%unit
        if colnm == len(table.colnames)-1:
            tablestr += r' \\\hline '
        else:
            tablestr += r' & '
    for rownm in range(len(table)):
        for colnm in range(len(table.colnames)):
            if (type(table[table.colnames[colnm]][rownm]) == str) or (type(table[table.colnames[colnm]][rownm]) == np.str_):
                tablestr += r'%s'% np.unicode(table[table.colnames[colnm]][rownm])
            else:
                if (type(table[table.colnames[colnm]][rownm]) == int) or (type(table[table.colnames[colnm]][rownm]) == np.int_) :
                    tablestr += r'%i'% table[table.colnames[colnm]][rownm]
                elif ('FWHM' in table.colnames[colnm]) or ('chi' in table.colnames[colnm]):
                    numstr = r'%.1f'%round(table[table.colnames[colnm]][rownm],1)
                    tablestr += numstr
                else:
                    numstr = r'%10.2g'%table[table.colnames[colnm]][rownm]
                    if 'E' in numstr or 'e' in numstr:
                        numstr = latexexponent(table[table.colnames[colnm]][rownm])
                        tablestr += numstr
                    else:
                        tablestr += numstr                    
            if colnm == len(table.colnames)-1:
                tablestr += r' \\\hline '
            else:
                tablestr += r' & '
    return tablestr


def colourgen(num=1):
    count = 0
    randtuple = []
    while count < num:
        rtuple = list(set(npr.random_integers(0,255,size=3)))
        if len(rtuple) == 3 and (np.mean(rtuple) < 190 and np.mean(rtuple) > 30):
            colour = '#%02X%02X%02X'%(rtuple[0], rtuple[1], rtuple[2])
            if colour not in randtuple:
                randtuple.append(colour)
                count += 1

    return randtuple


def progressbar(frac, pbar):
    if len(pbar) == 0:
        l = '0%'
        pbar = '0%'
        print( l, end='', sep='', file=sys.stdout, flush=True )
        # sys.stdout.write(l)
        # sys.stdout.flush()
        return pbar
    elif (frac%10 == 0) and (frac%2 == 0):
        if (str(frac)+'%') in pbar:
            return pbar
        else:
            dts=''
            if (frac > 5) and (pbar[-4:] != '....'):
                sub = pbar[-4:]
                if '%' in sub:
                    loc = int(sub.find('%'))
                    if loc == len(sub)-1:
                        dts = '.'*4
                    else:
                        cnt = sub[loc:].count('.')
                        dts = '.'*(4-cnt)
                else:
                    dts = '.'*4
                    
            l = dts+str(frac)+'%'
            if '100' in l:
                l += '\n'
            else:
                h = 0
            print( l, end='', sep='', file=sys.stdout, flush=True )
            # sys.stdout.write(l)
            # sys.stdout.flush()
            return pbar+l
    elif (frac%2 == 0) and (frac%10 != 0):
        h = eval(str(frac)[-1])

        if pbar[int(-h/2):] == '.'*int(h/2):
            return pbar
        else:
            l = '.'
            print( l, end='', sep='', file=sys.stdout, flush=True )
            # sys.stdout.write(l)
            # sys.stdout.flush()
            return pbar+l
    else:
        return pbar


def calc_dv_from_df(df, z, femit):
    """
    Use formula in Gordon, Baars and Cocke to calculate velocity channel widths
    given frequency channel widths (in MHz)
    and redshift
    Input:
    	df: the channel width in MHz
    	z: redshift of the galaxy
    	femit: the rest frequency of the emission/absorption line
    """
    c =  astcon.c.to('km/s')
    df = df.to('MHz')
    dv = df* c*(1.+z)/femit
    return dv


def calc_df_from_dv(dv, z, femit):
    """
    Use formula in Gordon, Baars and Cocke to calculate velocity channel widths
    given frequency channel widths (in MHz)
    and redshift
    Input:
    	df: the channel width in MHz
    	z: redshift of the galaxy
    	femit: the rest frequency of the emission/absorption line
    """
    c =  astcon.c.to('km/s').value
    df = dv / ( c*(1.+z)/femit)
    return df


def checkpath(file_path):
    if os.path.exists(file_path):
        return
    else:
        os.makedirs(file_path)
        return


def strlinelog(x, m, c):
    y = m/np.sqrt(x) + c
    return y

## Functions needed for fitting Gaussian - using the astropy frame work for creating douple and single gaussians

SingleGaussian = astmod.models.Gaussian1D
DoubleGaussian = (astmod.models.Gaussian1D + astmod.models.Gaussian1D).rename('DoubleGaussian')

# @astmod.models.custom_model
def singleGaussian(x, amp=1, cen=0, sig=1):
    y = amp * np.exp( -0.5*(x - cen)**2/(sig**2) )
    return y

def doubleGaussian(x, a1=1, b1=0, c1=1, a2=1, b2=0, c2=1):
    y = a1 * np.exp( -0.5*(x - b1)**2/(c1**2) ) + a2 * np.exp( -0.5*(x - b2)**2/(c2**2) )
    return y

def lorentzian(x, amp, center, width):
    """  A model based on a Lorentzian or Cauchy-Lorentz distribution function. Definition taken from lmfit module """
    s1 = width/np.pi
    s2 = width/( (x - center)**2 + width**2 )
    y = amp*s1*s2
    return y

# @astmod.models.custom_model
def VoigtModel(x, amplitude=1.0, center=0.0, gausswid=1.0, lorwid=0.):
    """1 dimensional voigt function.
    http://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
    https://www.princeton.edu/cefrc/Files/2013 Lecture Notes/Hanson/pLecture6.pdf
    """
    sigma = gausswid / np.sqrt(2 * np.log(2))
    vm = amplitude*(np.real(wofz((x + 1j*lorwid)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi))
    return vm

@astmod.models.custom_model
def fixed_gauss(x, amplitude=1., sigma=1.):
    """ Single Gaussian function with the mean fixed at 0, for the purpose of determing the p-value """
    g = amplitude * np.exp( -0.5*x**2 / sigma**2 )
    return g

# @astmod.models.custom_model
def breit_wigner(x, amplitude=1.0, center=0.0, sigma=1.0):
    """http://www-pnp.physics.ox.ac.uk/~barra/teaching/subatomic.pdf
    """
    bw = amplitude / ((x-center)**2 + sigma**2/4.)
    # gam = sigma/2.0
    # bw = amplitude*(q*gam + x - center)**2 / (gam*gam + (x-center)**2) 
    return  bw


# @astmod.models.custom_model
def busyfunction(x, a=0.4, b1=1, b2=1, xe=0, xp=0, c=1, w=6):
    """ Input parameters:
    a = amplitude scaling factor
    b1 = steepness of left peak; b-->0: slope flanks approach 0
    b2 = steepness of right peak; b-->infty: steep slope
    c = amplitute of central trough [c>0: increasing trough, c=0: no trough]  
    w = half width of the profile
    n = degree of trough polynomial
    xe, xp = centre of error functions
    negative values for a,b,c & w are unphysical for emission lines
    """
    n=2
    t1 = (a/4.)
    t2 = (erf(b1 * (w + x - xe)) + 1.)
    t3 = (erf(b2 * (w + xe - x)) + 1.)
    t4 = (c * (x - xp)**n + 1.)
    bf = t1*t2*t3*t4
    return bf

# @astmod.models.custom_model
def busyfunctionmodel(x, a=0.4, b1=1, b2=1, xe=0, w=6):
    """ Input parameters:
    a = amplitude scaling factor
    b1 = steepness of left peak; b-->0: slope flanks approach 0
    b2 = steepness of right peak; b-->infty: steep slope
    c = amplitute of central trough [c>0: increasing trough, c=0: no trough]  
    w = half width of the profile
    n = degree of trough polynomial
    xe, xp = centre of error functions
    negative values for a,b,c & w are unphysical for emission lines
    """
    c = 0
    n=2
    t1 = (a/4.)
    t2 = (erf(b1 * (w + x - xe)) + 1.)
    t3 = (erf(b2 * (w + xe - x)) + 1.)
    t4 = 1#(c * (x - xp)**n + 1.)
    bf = t1*t2*t3*t4
    return bf


# @astmod.models.custom_model
def gausshermite3(x, a=1, b=0, c=0.5, h3=0):
    y = (x-b)/c
    gh3 = a * np.exp( y**2/(-2.) ) * ( 1. + h3/np.sqrt(6) * ( 2*np.sqrt(2)*y**3 - 3.*np.sqrt(2)*y ) )
    return gh3

