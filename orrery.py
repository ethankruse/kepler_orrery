import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import datetime as dt
from diverging_map import diverge_map
import matplotlib.font_manager as fm
import pandas as pd
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic

from curved_text import CurvedText

# what KOI file to use
cd = os.path.abspath(os.path.dirname(__file__))
#koilist = os.path.join(cd, 'KOI_List.txt')
koilist = os.path.join(cd, 'download_toi.txt')
#koilist = os.path.join(cd, 'KOI_List_old.txt')

# are we loading in system locations from a previous file (None if not)
#lcenfile = os.path.join(cd, 'orrery_centers.txt')
#lcenfile = os.path.join(cd, 'orrery_centers_old.txt')
#lcenfile = None
lcenfile = os.path.join(cd, 'tess_centers.txt')
# if we're not loading a centers file,
# where do we want to save the one generated (None if don't save)
#scenfile = os.path.join(cd, 'tess_centers.txt')
scenfile = None

# add in the solar system to the plots
addsolar = True
# put it at a fixed location? otherwise use posinlist to place it
fixedpos = True
# fixed x and y positions (in AU) to place the Solar System
# if addsolar and fixedpos are True
ssx = -10.55
ssy = -1.5
# fraction of the way through the planet list to treat the solar system
# if fixedpos is False.
# 0 puts it first and near the center, 1 puts it last on the outside
posinlist = 0.2

# making rstart smaller or maxtry bigger takes longer but tightens the
# circle
# Radius of the circle (AU) to initially try placing a system
# when generating locations
rstart = 0.5
# number of tries to randomly place a system at a given radius
# before expanding the circle
maxtry = 50
# minimum spacing between systems (AU)
spacing = 0.15

# which font to use for the text
fontfile = os.path.join(cd, 'Avenir-Black.otf')
fontfam = 'normal'
fontcol = 'white'

# font sizes at various resolutions
fszs1 = {480: 12, 720: 14, 1080: 22}
fszs2 = {480: 15, 720: 17, 1080: 27}

# background color
bkcol = 'black'

# color and alpha for the circular orbit paths
orbitcol = '#424242'
orbitalpha = 1.

# add a background to the legend to distinguish it?
legback = True
# if so, use this color and alpha
legbackcol = bkcol
legalpha = 0.7

# are we making the png files for a movie or gif
makemovie = True
# resolution of the images. Currently support 480, 720 or 1080.
reso = 1080

# output directory for the images in the movie
# (will be created if it doesn't yet exist)
#outdir = os.path.join(cd, 'orrery-40s/')
outdir = os.path.join(cd, 'tess-movie/')

# number of frames to produce
# using ffmpeg with the palette at (sec * frames/sec)
#nframes = 15 * 20
nframes = 40 * 30

# times to evaluate the planets at
# Kepler observed from 120.5 to 1591
tstep = 0.2
times = np.arange(1795, 1795 + nframes*tstep, tstep)

# setup for the custom zoom levels
inds = np.arange(len(times))
nmax = inds[-1]
zooms = np.zeros_like(times) - 1.
x0s = np.zeros_like(times) + np.nan
y0s = np.zeros_like(times) + np.nan
startx, starty = 0, 0
endx, endy = 0, 0
# what zoom level each frame is at (1. means default with everything)

"""
# zoom out once
zooms[inds < 0.25 * nmax] = 0.35
zooms[inds > 0.7 * nmax] = 1.
zooms[zooms < 0.] = np.interp(inds[zooms < 0.], inds[zooms > 0.],
                              zooms[zooms > 0.])
"""
# zoom out then back in
zooms[inds < 0.25 * nmax] = 1.04
x0s[inds < 0.25 * nmax] = startx
y0s[inds < 0.25 * nmax] = starty
zooms[(inds > 0.5 * nmax) & (inds < 0.6 * nmax)] = 1.04
zooms[inds > 0.85 * nmax] = 1.04
x0s[inds > 0.85 * nmax] = endx
y0s[inds > 0.85 * nmax] = endy
zooms[zooms < 0.] = np.interp(inds[zooms < 0.], inds[zooms > 0.],
                              zooms[zooms > 0.])
x0s[~np.isfinite(x0s)] = np.interp(inds[~np.isfinite(x0s)], inds[np.isfinite(x0s)],
                              x0s[np.isfinite(x0s)])
y0s[~np.isfinite(y0s)] = np.interp(inds[~np.isfinite(y0s)], inds[np.isfinite(y0s)],
                              y0s[np.isfinite(y0s)])

# AU = parsecs * dscale
dscale = 1./26
# radius where a distance of 0 would go
zerodist = 0.2
# ===================================== #

# reference time for the Kepler data
#time0 = dt.datetime(2009, 1, 1, 12)
time0 = dt.datetime(2014, 12, 8, 12)

# the KIC number given to the solar system
kicsolar = -5

data = pd.read_csv(koilist)

# things that don't have a disposition get PC
data.rename(columns={'TFOPWG Disposition': 'disposition'}, inplace=True)
data['disposition'].replace(np.nan, 'PC', inplace=True)
# change this to the status we want to report
data['disposition'].replace('PC', 'Candidate', inplace=True)
data['disposition'].replace('KP', 'Confirmed', inplace=True)
data['disposition'].replace('CP', 'Confirmed', inplace=True)
data['disposition'].replace('APC', 'Candidate', inplace=True)
data['disposition'].replace('FA', 'False Positive', inplace=True)
data['disposition'].replace('FP', 'False Positive', inplace=True)
assert np.unique(data['disposition']).size == 3

kics = data['TIC ID'].values
pds = data['Period (days)'].values
it0s = data['Epoch (BJD)'].values
idists = data['Stellar Distance (pc)'].values
radius = data['Planet Radius (R_Earth)'].values
inc = data['Planet Insolation (Earth Flux)'].values
srad = data['Stellar Radius (R_Sun)'].values
stemp = data['Stellar Eff Temp (K)'].values
iteqs = data['Planet Equil Temp (K)'].values
tra = data['RA'].values
tdec = data['Dec'].values
slum = (srad**2) * ((stemp/5770)**4)
semi = np.sqrt((slum / inc))

ra = []
dec = []
for ii in np.arange(tra.size):
    ira = tra[ii]
    idec = tdec[ii]
    
    hh, mm, ss = ira.split(':')
    frac = int(hh) + int(mm)/60 + float(ss)/3600
    ra.append(frac * 360/24)
    
    hh, mm, ss = idec.split(':')
    frac = int(hh)
    if int(hh) < 0:
        frac -= int(mm)/60 + float(ss)/3600
    else:
        frac += int(mm)/60 + float(ss)/3600
    dec.append(frac)
ra = np.array(ra)
dec = np.array(dec)

# load in the data from the KOI list
#kics, pds, it0s, radius, iteqs, semi = np.genfromtxt(
#    koilist, unpack=True, usecols=(1, 5, 8, 20, 26, 23), delimiter=',')

# grab non-FPs
good = data['disposition'] != 'False Positive'
multikics, nct = np.unique(kics[good], return_counts=True)
multikics = multikics[nct > 1]

for ikic in multikics:
    srch = np.where(kics == ikic)[0]
    for isrch in srch:
        good = (np.isfinite(semi[isrch]) & np.isfinite(pds[isrch]) & (pds[isrch] > 0.) &
                np.isfinite(radius[isrch]) & np.isfinite(idists[isrch]) & np.isfinite(inc[isrch]) & np.isfinite(iteqs[isrch]))
        if not good:
            if ikic in [55652896, 120896927, 254113311, 260647166, 269701147, 278683844, 279741379, 425997655]:
                if ikic == 269701147:
                    pds[isrch] = 38.3561
                if ikic == 279741379:
                    pds[isrch] = 35.6125
                oth = srch[srch != isrch][0]
                tmpmass = (semi[oth]**3) / ((pds[oth] / 365.256)**2)
                tmpau = (((pds[isrch] / 365.256)**2) * tmpmass)**(1./3.)
                semi[isrch] = tmpau
                inc[isrch] = slum[isrch] * (tmpau**-2)
                iteqs[isrch] = (inc[isrch]**0.25)*255
            if ikic in [31852980]:
                idists[isrch] = 140
            
        good = (np.isfinite(semi[isrch]) & np.isfinite(pds[isrch]) & (pds[isrch] > 0.) &
                np.isfinite(radius[isrch]) & np.isfinite(idists[isrch]) & np.isfinite(inc[isrch]) & np.isfinite(iteqs[isrch]))
        if ikic not in [207425167, 328933398, 347332255, 420645189]:
            assert good


# grab the KICs with known parameters and not false positives
good = (np.isfinite(semi) & np.isfinite(pds) & (pds > 0.) &
        np.isfinite(radius) & np.isfinite(idists) & np.isfinite(inc) & np.isfinite(iteqs) & 
        (data['disposition'] != 'False Positive'))

kics = kics[good]
pds = pds[good]
it0s = it0s[good]
semi = semi[good]
radius = radius[good]
idists = idists[good]
inc = inc[good]
iteqs = iteqs[good]

# if we've already decided where to put each system, load it up
if lcenfile is not None:
    multikics, xcens, ycens, maxsemis = np.loadtxt(lcenfile, unpack=True)
    nplan = len(multikics)
# otherwise figure out how to fit all the planets into a nice distribution
else:
    # we only want to plot multi-planet systems
    multikics, nct = np.unique(kics, return_counts=True)
    multikics = multikics[nct > 1]
    maxsemis = multikics * 0.
    maxdists = multikics * 0.
    maxdecs = multikics * 0.
    maxras = multikics * 0.
    nplan = len(multikics)

    # the maximum size needed for each system
    for ii in np.arange(len(multikics)):
        maxsemis[ii] = np.max(semi[np.where(kics == multikics[ii])[0]])
        maxdists[ii] = np.max(idists[np.where(kics == multikics[ii])[0]])
        maxras[ii] = np.max(ra[np.where(kics == multikics[ii])[0]])
        maxdecs[ii] = np.max(dec[np.where(kics == multikics[ii])[0]])

    inds = np.argsort(maxdists)
    # reorder to place them
    maxsemis = maxsemis[inds]
    multikics = multikics[inds]
    maxdists = maxdists[inds]
    maxras = maxras[inds]
    maxdecs = maxdecs[inds]
    
    useecl = False
    
    if useecl:
        icrs = SkyCoord(ra=maxras, dec=maxdecs, frame='icrs', unit='deg')
        ecliptic = icrs.transform_to(BarycentricTrueEcliptic)
        maxrasecl = ecliptic.lon.value * 1
        maxdecsecl = ecliptic.lat.value * 1
    
    # add in the solar system if desired
    if addsolar:
        nplan += 1
        # we know where we want the solar system to be placed, place it first
        if fixedpos:
            insind = 0
        # otherwise treat it as any other system
        # and place it at this point through the list
        else:
            insind = int(posinlist * len(maxsemis))
    
        maxsemis = np.insert(maxsemis, insind, 1.524)
        multikics = np.insert(multikics, insind, kicsolar)
        maxdists = np.insert(maxdists, insind, 0)
        if useecl:
            maxrasecl = np.insert(maxrasecl, insind, 0)
            maxdecsecl = np.insert(maxdecsecl, insind, 0)
        else:
            maxras = np.insert(maxras, insind, 0)
            maxdecs = np.insert(maxdecs, insind, 0)
    
    if useecl:
        phase = maxdecsecl * 1
        phase[maxrasecl > 180] = 180 - phase[maxrasecl > 180]
    else:
        phase = maxdecs * 1
        phase[maxras > 180] = 180 - phase[maxras > 180]
    
    # XXX: special cases
    # the two tan planets in the bottom left
    phase[multikics==288636342] = 210
    # the two planets in the top right
    phase[multikics==280031353] = 50
    # the red, big planets near the 750 ly text
    phase[multikics==309257814] = 215
    # to avoid overlap with 50 ly text
    phase[multikics==307210830] = 195
    
    xcens = np.array([])
    ycens = np.array([])
    # place all the planets without overlapping or violating aspect ratio
    for ii in np.arange(nplan):
        # reset the counters
        repeat = True
        phaseoff = 0
        ispace = spacing * 1
        
        #if maxdists[ii] < 50:
        #    ispace = 0
    
        # progress bar
        if (ii % 20) == 0:
            print('Placing {0} of {1} planets'.format(ii, nplan))
    
        # put the solar system at its fixed position if desired
        if multikics[ii] == kicsolar and fixedpos:
            xcens = np.concatenate((xcens, [ssx]))
            ycens = np.concatenate((ycens, [ssy]))
            repeat = False
        else:
            xcens = np.concatenate((xcens, [0.]))
            ycens = np.concatenate((ycens, [0.]))
    
        # repeat until we find an open location for this system
        while repeat:
            iphase = (phase[ii] + phaseoff) * np.pi/180
            xcens[ii] = np.cos(iphase) * (maxdists[ii] * dscale + zerodist)
            ycens[ii] = np.sin(iphase) * (maxdists[ii] * dscale + zerodist)
    
            # how far apart are all systems
            rdists = np.sqrt((xcens - xcens[ii]) ** 2. +
                            (ycens - ycens[ii]) ** 2.)
            rsum = maxsemis + maxsemis[ii]
    
            # systems that overlap
            bad = np.where(rdists < rsum[:ii + 1] + ispace)
    
            # either the systems overlap or we've placed a lot and
            # the aspect ratio isn't good enough so try again
            if len(bad[0]) == 1:
                repeat = False
                
            if phaseoff == 0:
                phaseoff = 2
            elif phaseoff > 0:
                phaseoff *= -1
            else:
                phaseoff *= -1
                phaseoff += 2
    
            if phaseoff > 180:
                raise Exception('bad')
        #print(phaseoff)

    # save this placement distribution if desired
    if scenfile is not None:
        np.savetxt(scenfile,
                   np.column_stack((multikics, xcens, ycens, maxsemis)),
                   fmt=['%d', '%f', '%f', '%f'])

plt.close('all')

# make a diagnostic plot showing the distribution of systems
fig = plt.figure()
plt.xlim((xcens - maxsemis).min(), (xcens + maxsemis).max())
plt.ylim((ycens - maxsemis).min(), (ycens + maxsemis).max())
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('AU')
plt.ylabel('AU')

for ii in np.arange(nplan):
    c = plt.Circle((xcens[ii], ycens[ii]), maxsemis[ii], clip_on=False,
                   alpha=0.3)
    fig.gca().add_artist(c)

# all of the parameters we need for the plot
t0s = np.array([])
periods = np.array([])
semis = np.array([])
radii = np.array([])
teqs = np.array([])
dists = np.array([])
usedkics = np.array([])
fullxcens = np.array([])
fullycens = np.array([])
incs = np.array([])

for ii in np.arange(nplan):
    # known solar system parameters
    if addsolar and multikics[ii] == kicsolar:
        usedkics = np.concatenate((usedkics, np.ones(4) * kicsolar))
        # always start the outer solar system in the same places
        # for optimial visibility
        t0s = np.concatenate((t0s, [85., 192., 266., 180.]))#,
                                   # times[0] + 0.1 * 4332.8,
                                   # times[0] - 22. / 360 * 10755.7,
                                   # times[0] - 30687 * 145. / 360,
                                   # times[0] - 60190 * 202. / 360]))
        periods = np.concatenate((periods, [87.97, 224.70, 365.26, 686.98]))#,
                                            #4332.8, 10755.7, 30687, 60190]))
        semis = np.concatenate((semis, [0.387, 0.723, 1.0, 1.524]))#, 5.203,
                                       # 9.537, 19.19, 30.07]))
        radii = np.concatenate((radii, [0.383, 0.95, 1.0, 0.53]))#, 10.86, 9.00,
                                        #3.97, 3.86]))
        dists = np.concatenate((dists, np.ones(4)*0.01))
        fullxcens = np.concatenate((fullxcens, np.zeros(4) + xcens[ii]))
        fullycens = np.concatenate((fullycens, np.zeros(4) + ycens[ii]))
        incs = np.concatenate((incs, [6.68, 1.91, 1, 0.43]))#, 0.037, 0.011,
                                      #0.0027, 0.0011]))
        teqs = np.concatenate((teqs, [409, 299, 255, 206]))#, 200,
                                      #200, 200, 200]))
        continue

    fd = np.where(kics == multikics[ii])[0]
    # get the values for this system
    usedkics = np.concatenate((usedkics, kics[fd]))
    t0s = np.concatenate((t0s, it0s[fd]))
    periods = np.concatenate((periods, pds[fd]))
    semis = np.concatenate((semis, semi[fd]))
    radii = np.concatenate((radii, radius[fd]))
    incs = np.concatenate((incs, inc[fd]))
    teqs = np.concatenate((teqs, iteqs[fd]))
    dists = np.concatenate((dists, idists[fd]))
    fullxcens = np.concatenate((fullxcens, np.zeros(len(fd)) + xcens[ii]))
    fullycens = np.concatenate((fullycens, np.zeros(len(fd)) + ycens[ii]))
    

# sort by radius so that the large planets are on the bottom and
# don't cover smaller planets
rs = np.argsort(radii)[::-1]
usedkics = usedkics[rs]
t0s = t0s[rs]
periods = periods[rs]
semis = semis[rs]
radii = radii[rs]
incs = incs[rs]
teqs = teqs[rs]
dists = dists[rs]
fullxcens = fullxcens[rs]
fullycens = fullycens[rs]

if makemovie:
    plt.ioff()
else:
    plt.ion()

# create the figure at the right size (this assumes a default pix/inch of 100)
figsizes = {480: (8.54, 4.8), 720: (8.54, 4.8), 1080: (19.2, 10.8)}
fig = plt.figure(figsize=figsizes[reso])

# make the plot cover the entire figure with the right background colors
ax = fig.add_axes([0.0, 0, 1, 1])
ax.axis('off')
fig.patch.set_facecolor(bkcol)
ax.patch.set_facecolor(bkcol)

# don't count the orbits of the outer solar system in finding figure limits
ns = np.where(usedkics != kicsolar)[0]

# this section manually makes the aspect ratio equal
#  but completely fills the figure

# need this much buffer zone so that planets don't get cut off
buffsx = (fullxcens[ns].max() - fullxcens[ns].min()) * 0.007
buffsy = (fullycens[ns].max() - fullycens[ns].min()) * 0.007
# current limits of the figure
xmax = (fullxcens[ns] + semis[ns]).max() + buffsx
xmin = (fullxcens[ns] - semis[ns]).min() - buffsx
ymax = (fullycens[ns] + semis[ns]).max() + buffsy
ymin = (fullycens[ns] - semis[ns]).min() - buffsy

# figure aspect ratio
sr = 16. / 9.

# make the aspect ratio exactly right
if (xmax - xmin) / (ymax - ymin) > sr:
    plt.xlim(xmin, xmax)
    plt.ylim((ymax + ymin) / 2. - (xmax - xmin) / (2. * sr),
             (ymax + ymin) / 2. + (xmax - xmin) / (2. * sr))
else:
    plt.ylim(ymin, ymax)
    plt.xlim((xmax + xmin) / 2. - (ymax - ymin) * sr / 2.,
             (xmax + xmin) / 2. + (ymax - ymin) * sr / 2.)

lws = {480: 1, 720: 1, 1080: 2}
sslws = {480: 2, 720: 2, 1080: 4}
# plot the orbital circles for every planet
for ii in np.arange(len(t0s)):
    # solid, thinner lines for normal planets
    ls = 'solid'
    zo = 0
    lw = lws[reso]
    # dashed, thicker ones for the solar system
    if usedkics[ii] == kicsolar:
        ls = 'dashed'
        zo = -3
        lw = sslws[reso]

    c = plt.Circle((fullxcens[ii], fullycens[ii]), semis[ii], clip_on=False,
                   alpha=orbitalpha, fill=False,
                   color=orbitcol, zorder=zo, ls=ls, lw=lw)
    fig.gca().add_artist(c)
    
    
    
    
fsz1 = fszs1[reso]
fsz2 = fszs2[reso]
prop = fm.FontProperties(fname=fontfile)
    
# plot the distance markers
pldists = [50, 250, 500, 750, 1000]
txtangles = [270, 215, 210, 210, 207]
for ii in np.arange(len(pldists)):
    # solid, thinner lines for normal planets
    ls = 'solid'
    zo = -5
    lw = sslws[reso]
    
    
    idist = zerodist + (pldists[ii]/3.26156) * dscale

    c = plt.Circle((0, 0), idist, clip_on=False,
                   alpha=orbitalpha, fill=False,
                   color='#888888', zorder=zo, ls=':', lw=lw)
    fig.gca().add_artist(c)
    
    if ii == 1:
        ang = txtangles[ii] * np.pi/180
        #plt.text(idist * np.cos(ang), idist * np.sin(ang), 'Distance from the\nSolar System',
        #         color=fontcol, family=fontfam, fontproperties=prop, fontsize=fsz1,
        #         horizontalalignment='center', verticalalignment='center', rotation=txtangles[ii]-90)
        
        itxt = f'{pldists[ii]} light-years'
    else:
        ang = txtangles[ii] * np.pi/180
        itxt = f'{pldists[ii]} ly'
    
    iang = txtangles[ii]-90
    if iang > 90:
        iang -= 180
    
    if ii == 0:
        idist = 37 * dscale / 3.26156
        plt.text(idist * np.cos(ang), idist * np.sin(ang), itxt,
         color=fontcol, fontproperties=prop, fontsize=fsz1,
         horizontalalignment='center', verticalalignment='center', rotation=iang, zorder=-3)
    else:
        idist += 10 * dscale / 3.26156    
        # make the text curve as well    
        xa = idist * np.cos(np.linspace(ang, ang + 2*np.pi, 500))
        ya = idist * np.sin(np.linspace(ang, ang + 2*np.pi, 500))
        
        text = CurvedText(x=xa, y=ya, text=itxt, va='top',
                          axes=plt.gca(), color=fontcol, fontproperties=prop,
                          fontsize=fsz1)
    
    

# set up the planet size scale
sscales = {480: 12., 720: 30., 1080: 50.}
sscale = sscales[reso]

rearth = 1.
rnep = 3.856
rjup = 10.864
rmerc = 0.383
# for the planet size legend
solarsys = np.array([rearth, rnep, rjup])[::-1]
pnames = ['Earth', 'Neptune', 'Jupiter'][::-1]
csolar = np.array([255, 46, 112])[::-1]

# keep the smallest planets visible and the largest from being too huge
solarsys = np.clip(solarsys, 0.8, 1.3 * rjup)
solarscale = sscale * solarsys

radii = np.clip(radii, 0.8, 1.3 * rjup)
pscale = sscale * radii

# color bar temperature tick values and labels
ticks = np.array([250, 500, 750, 1000, 1250])
labs = ['250', '500', '750', '1000', '1250']

# blue and red colors for the color bar
RGB1 = np.array([1, 185, 252])
RGB2 = np.array([220, 55, 19])

# create the diverging map with a white in the center
mycmap = diverge_map(RGB1=RGB1, RGB2=RGB2, numColors=15)

# just plot the planets at time 0. for this default plot
phase = 2. * np.pi * (0. - t0s) / periods
tmp = plt.scatter(fullxcens + semis * np.cos(phase),
                  fullycens + semis * np.sin(phase), marker='o',
                  edgecolors='none', lw=0, s=pscale, c=teqs, vmin=ticks.min(),
                  vmax=ticks.max(), zorder=3, cmap=mycmap, clip_on=False)

# create the 'Solar System' text identification
if addsolar:
    loc = np.where(usedkics == kicsolar)[0][0]
    plt.text(fullxcens[loc], fullycens[loc] - 1.65, 'Inner Solar System', zorder=-2,
             color=fontcol, fontproperties=prop, fontsize=fsz1,
             horizontalalignment='center', verticalalignment='top')

# if we're putting in a translucent background behind the text
# to make it easier to read
if legback:
    box1starts = {480: (0., 0.445), 720: (0., 0.46), 1080: (0., 0.25)}
    box1widths = {480: 0.19, 720: 0.147, 1080: 0.244}
    box1heights = {480: 0.555, 720: 0.54, 1080: 0.75}

    box2starts = {480: (0.79, 0.8), 720: (0.83, 0.84), 1080: (0.86, 0.84)}
    box2widths = {480: 0.21, 720: 0.17, 1080: 0.14}
    box2heights = {480: 0.2, 720: 0.16, 1080: 0.16}
    
    box3starts = {480: (0.79, 0.8), 720: (0.83, 0.84), 1080: (0.82, 0.45)}
    box3widths = {480: 0.21, 720: 0.17, 1080: 0.18}
    box3heights = {480: 0.2, 720: 0.16, 1080: 0.1}

    # create the rectangles at the right heights and widths
    # based on the resolution
    c = plt.Rectangle(box1starts[reso], box1widths[reso], box1heights[reso],
                      alpha=legalpha, fc=legbackcol, ec='none', zorder=-4,
                      transform=ax.transAxes)
    d = plt.Rectangle(box2starts[reso], box2widths[reso], box2heights[reso],
                      alpha=legalpha, fc=legbackcol, ec='none', zorder=-4,
                      transform=ax.transAxes)
    e = plt.Rectangle(box3starts[reso], box3widths[reso], box3heights[reso],
                      alpha=legalpha, fc=legbackcol, ec='none', zorder=-4,
                      transform=ax.transAxes)
    ax.add_artist(c)
    ax.add_artist(d)
    ax.add_artist(e)

# appropriate spacing from the left edge for the color bar
#cbxoffs = {480: 0.09, 720: 0.07, 1080: 0.074}
cbxoffs = {480: 0.09, 720: 0.07, 1080: 0.09}
cbxoff = cbxoffs[reso]

# plot the solar system planet scale
ax.scatter(np.zeros(len(solarscale)) + cbxoff,
           1. - 0.47 + 0.03 * np.arange(len(solarscale)), s=solarscale,
           c=csolar, zorder=5, marker='o',
           edgecolors='none', lw=0, cmap=mycmap, vmin=ticks.min(),
           vmax=ticks.max(), clip_on=False, transform=ax.transAxes)

# put in the text labels for the solar system planet scale
for ii in np.arange(len(solarscale)):
    ax.text(cbxoff - 0.01, 1. - 0.47 - 0.002 + 0.03 * ii,
            pnames[ii], color=fontcol, 
            fontproperties=prop, fontsize=fsz1, zorder=5,
            transform=ax.transAxes, verticalalignment='center', horizontalalignment='right')

# colorbar axis on the left centered with the planet scale
ax2 = fig.add_axes([cbxoff - 0.005, 0.62, 0.01, 0.3])
ax2.set_zorder(2)
cbar = plt.colorbar(tmp, cax=ax2, extend='both', ticks=ticks)
ax3 = cbar.ax.twinx()
ax3.set_ylim(cbar.ax.get_ylim())
# remove the white/black outline around the color bar
cbar.outline.set_linewidth(0)
# allow two different tick scales
cbar.ax.minorticks_on()
# turn off tick lines and put the physical temperature scale on the left
cbar.ax.tick_params(axis='y', which='major', color=fontcol, width=2,
                    left=False, right=False, length=5, labelleft=False,
                    labelright=True, zorder=5)
# turn off tick lines and put the physical temperature approximations
# on the right
cbar.ax.tick_params(axis='y', which='minor', color=fontcol, width=2,
                    left=False, right=False, length=5, labelleft=True,
                    labelright=False, zorder=5)


# add another layer of labels
ax3.tick_params(axis='y', which='major', color=fontcol, width=2,
                    left=False, right=False, length=5, labelleft=False,
                    labelright=True, zorder=5, pad=60)

ax3.tick_params(axis='y', which='minor', color=fontcol, width=2,
                    left=False, right=False, length=5, labelleft=False,
                    labelright=False, zorder=5)
ax3.yaxis.set_ticks([255, 409, 730, 1200])
ax3.set_yticklabels(['Earth', 'Mercury', 'Surface\nof Venus', 'Lava'],
                        color=fontcol, 
                        fontproperties=prop, fontsize=fsz1)
for label in ax3.get_yticklabels(which='both'):
    label.set_fontproperties(prop)
    label.set_fontsize(fsz1)

# eq temp = (incident flux)**0.25 * 255
# say where to put the physical temperature approximations and give them labels
cbar.ax.yaxis.set_ticks((np.array([1, 10, 100, 300, 500])**0.25)*255, minor=True)
cbar.ax.set_yticklabels(labs, color=fontcol, 
                        fontproperties=prop, fontsize=fsz1, zorder=5)
cbar.ax.set_yticklabels(['1', '10', '100', '300', '500'],
                        minor=True, color=fontcol, 
                        fontproperties=prop, fontsize=fsz1)
for label in cbar.ax.get_yticklabels(which='both'):
    label.set_fontproperties(prop)
    label.set_fontsize(fsz1)
#cbar.ax.yaxis.set_label('Insolation\n(Earths)')
#cbar.ax.yaxis.set_label('Light year', minor=True)

clab = ''
# add the overall label at the bottom of the color bar
cbar.ax.set_xlabel(clab, color=fontcol, fontproperties=prop,
                   size=fsz1, zorder=5, labelpad=fsz1*1.5)

# switch back to the main plot
plt.sca(ax)
plt.text(cbxoff + 0.01, 0.92 + 0.01, 'Planet Equilibrium\nTemperature (K)', transform=ax.transAxes,color=fontcol,
                        fontproperties=prop, fontsize=fsz1, zorder=5,horizontalalignment='left', verticalalignment='bottom')
plt.text(cbxoff - 0.01, 0.92 + 0.01, 'Insolation\n(Earths)', transform=ax.transAxes,color=fontcol,
                        fontproperties=prop, fontsize=fsz1, zorder=5,horizontalalignment='right', verticalalignment='bottom')

# upper right credit and labels text offsets
txtxoffs = {480: 0.01, 720: 0.01, 1080: 0.01}
txtyoffs1 = {480: 0.10, 720: 0.08, 1080: 0.08}
txtyoffs2 = {480: 0.18, 720: 0.144, 1080: 0.144}

txtxoff = txtxoffs[reso]
txtyoff1 = txtyoffs1[reso]
txtyoff2 = txtyoffs2[reso]

# put in the credits in the top right
text = plt.text(1. - txtxoff, 1. - txtyoff1,
                time0.strftime('TESS Orrery I\n%d %b %Y'), color=fontcol,
                fontproperties=prop,
                fontsize=fsz2, zorder=5, transform=ax.transAxes, horizontalalignment='right')
plt.text(1. - txtxoff, 1. - txtyoff2, 'By Ethan Kruse\n@ethan_kruse',
         color=fontcol, 
         fontproperties=prop, fontsize=fsz1,
         zorder=5, transform=ax.transAxes, horizontalalignment='right')

# the center of the figure
x0 = np.mean(plt.xlim())
y0 = np.mean(plt.ylim())

# width of the figure
xdiff = np.diff(plt.xlim()) / 2.
ydiff = np.diff(plt.ylim()) / 2.


# add hemisphere text

#xl = plt.xlim()

orig = (0 - plt.ylim()[0])/np.diff(plt.ylim())

plt.text(0.995, orig + 0.01, 'Northern Hemisphere',
     color=fontcol, fontproperties=prop, fontsize=fsz1,
     horizontalalignment='right', verticalalignment='bottom', zorder=-3,
     transform=ax.transAxes)

plt.text(0.995, orig - 0.01, 'Southern Hemisphere',
     color=fontcol, fontproperties=prop, fontsize=fsz1,
     horizontalalignment='right', verticalalignment='top', zorder=-3,
     transform=ax.transAxes)

plt.plot([0.83, 1], [orig,orig], lw=3, c=fontcol, transform=ax.transAxes)
#plt.xlim(xl)

# create the output directory if necessary
if makemovie and not os.path.exists(outdir):
    os.mkdir(outdir)

if makemovie:
    # get rid of all old png files so they don't get included in a new movie
    oldfiles = glob(os.path.join(outdir, '*png'))
    for delfile in oldfiles:
        os.remove(delfile)

    # go through all the times and make the planets move
    for ii, time in enumerate(times):
        # remove old planet locations and dates
        tmp.remove()
        text.remove()

        # re-zoom to appropriate level
        plt.xlim([x0s[ii] - xdiff * zooms[ii], x0s[ii] + xdiff * zooms[ii]])
        plt.ylim([y0s[ii] - ydiff * zooms[ii], y0s[ii] + ydiff * zooms[ii]])

        newt = time0 + dt.timedelta(time)
        # put in the credits in the top right
        text = plt.text(1. - txtxoff, 1. - txtyoff1,
                        newt.strftime('TESS Orrery I\n%d %b %Y'),
                        color=fontcol, 
                        fontproperties=prop,
                        fontsize=fsz2, zorder=5, transform=ax.transAxes, horizontalalignment='right')
        # put the planets in the correct location
        phase = 2. * np.pi * (time - t0s) / periods
        tmp = plt.scatter(fullxcens + semis * np.cos(phase),
                          fullycens + semis * np.sin(phase),
                          marker='o', edgecolors='none', lw=0, s=pscale, c=teqs,
                          vmin=ticks.min(), vmax=ticks.max(),
                          zorder=3, cmap=mycmap, clip_on=False)

        fig.savefig(os.path.join(outdir, 'fig{0:04d}.png'.format(ii)),
                    facecolor=fig.get_facecolor(), edgecolor='none')
        if not (ii % 10):
            print('{0} of {1} frames'.format(ii, len(times)))








