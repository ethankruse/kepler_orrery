import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import datetime as dt
from diverging_map import diverge_map
import matplotlib.font_manager as fm
from matplotlib.ticker import FixedLocator as FL

# what planet parameter file to use
cd = os.path.abspath(os.path.dirname(__file__))
koilist = os.path.join(cd, 'luvoir_orrery.txt')

# are we loading in system locations from a previous file (None if not)
lcenfile = os.path.join(cd, 'luvoir_centers3.txt')
#lcenfile = None
# if we're not loading a centers file,
# where do we want to save the one generated (None if don't save)
#scenfile = os.path.join(cd, 'luvoir_centers4.txt')
scenfile = None

# add in the solar system to the plots
addsolar = True
# put it at a fixed location? otherwise use posinlist to place it
fixedpos = True
# fixed x and y positions (in AU) to place the Solar System
# if addsolar and fixedpos are True
ssx = 50.
ssy = 0.
# fraction of the way through the planet list to treat the solar system
# if fixedpos is False.
# 0 puts it first and near the center, 1 puts it last on the outside
posinlist = 0.2

# making rstart smaller or maxtry bigger takes longer but tightens the
# circle
# Radius of the circle (AU) to initially try placing a system
# when generating locations
rstart = 10.
# number of tries to randomly place a system at a given radius
# before expanding the circle
maxtry = 50
# minimum spacing between systems (AU)
spacing = 2.5

# which font to use for the text
fontfile = os.path.join(cd, 'Avenir-Black.otf')
fontfam = 'normal'
fontcol = 'white'


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

# don't plot any of the image text
notext = False


# font sizes at various resolutions
fszs1 = {480: 12, 720: 14, 1080: 22, 'mag': 22*3.125}
fszs2 = {480: 15, 720: 17, 1080: 27, 'mag': 27*3.125}

figsizes = {480: (8.54, 4.8), 720: (8.54, 4.8), 1080: (19.2, 10.8), 'mag':(51.75, 33.75)}

# line widths and solar system line widths
lws = {480: 1, 720: 1, 1080: 2, 'mag': 2}
sslws = {480: 2, 720: 2, 1080: 4, 'mag': 4}
# planet size scales
sscales = {480: 12., 720: 30., 1080: 50., 'mag': 50*3.125}

# properties of translucent boxes behind text
box1starts = {480: (0., 0.445), 720: (0., 0.46), 1080: (0., 0.47), 'mag': (0., 0.47)}
box1widths = {480: 0.19, 720: 0.147, 1080: 0.153, 'mag': 0.153}
box1heights = {480: 0.555, 720: 0.54, 1080: 0.53, 'mag': 0.53}

box2starts = {480: (0.79, 0.8), 720: (0.83, 0.84), 1080: (0.83, 0.84), 'mag': (0.83, 0.84)}
box2widths = {480: 0.21, 720: 0.17, 1080: 0.17, 'mag': 0.17}
box2heights = {480: 0.2, 720: 0.16, 1080: 0.16, 'mag': 0.16}

# scaling from left edge for color bar
cbxoffs = {480: 0.09, 720: 0.07, 1080: 0.074, 'mag': 0.074}

# upper right credit and labels text offsets
txtxoffs = {480: 0.2, 720: 0.16, 1080: 0.16, 'mag': 0.16}
txtyoffs1 = {480: 0.10, 720: 0.08, 1080: 0.08, 'mag': 0.08}
txtyoffs2 = {480: 0.18, 720: 0.144, 1080: 0.144, 'mag': 0.144}

# output directory for the images in the movie
# (will be created if it doesn't yet exist)
#outdir = os.path.join(cd, 'orrery-40s/')
outdir = os.path.join(cd, 'luvoir/')

# number of frames to produce
# using ffmpeg with the palette at (sec * frames/sec)
# nframes = 40 * 20
nframes = 30 * 30

# times to evaluate the planets at
# Kepler observed from 120.5 to 1591
# 11420 = April 9, 2040
times = np.arange(11220, 11220 + nframes * 5., 5.)

# setup for the custom zoom levels
inds = np.arange(len(times))
nmax = inds[-1]
zooms = np.zeros_like(times) - 1.
x0s = np.zeros_like(times) + np.nan
y0s = np.zeros_like(times) + np.nan
startx, starty = 5, -2
endx, endy = -5, -2

earthcol = '#00cb00'
# what zoom level each frame is at (1. means default with everything)

"""
# zoom out once
zooms[inds < 0.25 * nmax] = 0.35
zooms[inds > 0.7 * nmax] = 1.
zooms[zooms < 0.] = np.interp(inds[zooms < 0.], inds[zooms > 0.],
                              zooms[zooms > 0.])
"""
# zoom out then back in
zooms[inds < 0.25 * nmax] = 0.45
x0s[inds < 0.05 * nmax] = startx
y0s[inds < 0.05 * nmax] = starty
zooms[(inds > 0.5 * nmax) & (inds < 0.6 * nmax)] = 1.
zooms[inds > 0.85 * nmax] = 0.45
x0s[inds > 0.95 * nmax] = endx
y0s[inds > 0.95 * nmax] = endy

zooms[zooms < 0.] = np.interp(inds[zooms < 0.], inds[zooms > 0.],
                              zooms[zooms > 0.])
x0s[~np.isfinite(x0s)] = np.interp(inds[~np.isfinite(x0s)], inds[np.isfinite(x0s)],
                              x0s[np.isfinite(x0s)])
y0s[~np.isfinite(y0s)] = np.interp(inds[~np.isfinite(y0s)], inds[np.isfinite(y0s)],
                              y0s[np.isfinite(y0s)])

zooms = zooms * 0. + 1.01

# ===================================== #

# reference time for the Kepler data
time0 = dt.datetime(2009, 1, 1, 12)


# the KIC number given to the solar system
kicsolar = -5

# load in the data from the planet list
# kics, pds, it0s, radius, iteqs, semi = np.genfromtxt(
#     koilist, unpack=True, usecols=(1, 5, 8, 20, 26, 23), delimiter=',')

# star_ID,Period(yrs),Radius(REarth),a(AU),BBtemp(K),planet_type
kics, pds, radius, semi, iteqs, ptypes = np.genfromtxt(
    koilist, unpack=True, delimiter=',')

np.random.seed(42)
# sort out the luvoir parameters and give random t0s
pds *= 365.
it0s = np.random.rand(pds.size) * pds


# grab the KICs with known parameters
good = (np.isfinite(semi) & np.isfinite(pds) &
        np.isfinite(radius) & np.isfinite(iteqs))
# remove the luvoir overlaps
vals = np.where((ptypes == 1) | (ptypes == 4))[0]
toremove = np.random.choice(vals, size=54, replace=False)
good[toremove] = False


kics = kics[good]
pds = pds[good]
it0s = it0s[good]
semi = semi[good]
radius = radius[good]
iteqs = iteqs[good]
ptypes = ptypes[good]

# convert to insolation flux (assume albedo of 0.3)
sbconst = 5.67e-8
earthinsol = 1.3615e3
iinsol = (iteqs**4.) * 4. * sbconst / 0.7
iinsol /= earthinsol


# if we've already decided where to put each system, load it up
if lcenfile is not None:
    multikics, xcens, ycens, maxsemis = np.loadtxt(lcenfile, unpack=True)
    nplan = len(multikics)
# otherwise figure out how to fit all the planets into a nice distribution
else:
    np.random.seed()
    # we only want to plot multi-planet systems
    multikics, nct = np.unique(kics, return_counts=True)
    # multikics = multikics[nct > 1]
    maxsemis = multikics * 0.
    nplan = len(multikics)

    # the maximum size needed for each system
    for ii in np.arange(len(multikics)):
        maxsemis[ii] = np.max(semi[np.where(kics == multikics[ii])[0]])

    # place the smallest ones first, but add noise
    # so they aren't perfectly in order
    inds = np.argsort(maxsemis + np.random.randn(len(maxsemis)) * 0.5)[::-1]
    
    # place the smallest ones first, but add uniform noise
    # so they aren't perfectly in order
    #inds = np.argsort(maxsemis + np.random.uniform(low=-2, high=2, size=len(maxsemis)))[::-1]
    
    # purely random order
    #np.random.shuffle(inds)
    
    # biggest ones first, small ones fill in
    #bigs = np.where(maxsemis > 1.)[0]
    #smalls = np.where(maxsemis <= 1.)[0]
    #np.random.shuffle(smalls)
    #inds = np.concatenate((bigs, smalls))

    # reorder to place them
    maxsemis = maxsemis[inds]
    multikics = multikics[inds]

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

        # maxsemis = np.insert(maxsemis, insind, 1.524)
        maxsemis = np.insert(maxsemis, insind, 30.07)
        multikics = np.insert(multikics, insind, kicsolar)

    # ratio = x extent / y extent
    # what is the maximum and minimum aspect ratio of the final placement
    maxratio = 16.5 / 9
    minratio = 15.5 / 9

    xcens = np.array([])
    ycens = np.array([])
    # place all the planets without overlapping or violating aspect ratio
    for ii in np.arange(nplan):
        # reset the counters
        repeat = True
        r0 = rstart * 1.
        ct = 0
        ratio = 1.

        # progress bar
        if (ii % 20) == 0:
            print(f'Placing {ii} of {nplan} planets')

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
            # pick a random radius (up to our limit) and angle
            r = np.random.rand() * r0
            theta = np.random.rand() * 2. * np.pi
            xcens[ii] = r * np.cos(theta)
            ycens[ii] = r * np.sin(theta)

            # check what the aspect ratio would be if we place it here
            xex = (xcens + maxsemis[:ii + 1]).max() - \
                  (xcens - maxsemis[:ii + 1]).min()
            yex = (ycens + maxsemis[:ii + 1]).max() - \
                  (ycens - maxsemis[:ii + 1]).min()
            ratio = xex / yex

            # how far apart are all systems
            dists = np.sqrt((xcens - xcens[ii]) ** 2. +
                            (ycens - ycens[ii]) ** 2.)
            rsum = maxsemis + maxsemis[ii]

            # systems that overlap
            bad = np.where(dists < rsum[:ii + 1] + spacing)

            # either the systems overlap or we've placed a lot and
            # the aspect ratio isn't good enough so try again
            if len(bad[0]) == 1 and (
                        (minratio <= ratio <= maxratio) or ii < 20):
                repeat = False

            # if we've been trying to place this system but can't get it
            # at this radius, expand the search zone
            if ct > maxtry:
                ct = 0
                # add equal area every time
                r0 = np.sqrt(rstart ** 2. + r0 ** 2.)
                #if r0 > 100:
                #    print(f'bad {ii}')
            ct += 1

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
insols = np.array([])
usedkics = np.array([])
fullxcens = np.array([])
fullycens = np.array([])
exoearths = np.array([]).astype(bool)

for ii in np.arange(nplan):
    # known solar system parameters
    if addsolar and multikics[ii] == kicsolar:
        usedkics = np.concatenate((usedkics, np.ones(8) * kicsolar))
        # always start the outer solar system in the same places
        # for optimial visibility
        t0s = np.concatenate((t0s, [85., 192., 266., 180.,
                                    times[0] - 3. * 4332.8 / 4,
                                    times[0] - 22. / 360 * 10755.7,
                                    times[0] - 30687 * 145. / 360,
                                    times[0] - 60190 * 202. / 360]))
        periods = np.concatenate((periods, [87.97, 224.70, 365.26, 686.98,
                                            4332.8, 10755.7, 30687, 60190]))
        semis = np.concatenate((semis, [0.387, 0.723, 1.0, 1.524, 5.203,
                                        9.537, 19.19, 30.07]))
        radii = np.concatenate((radii, [0.383, 0.95, 1.0, 0.53, 10.86, 9.00,
                                        3.97, 3.86]))
        teqs = np.concatenate((teqs, [409, 299, 255, 206, 100,
                                      80, 60, 50]))
        insols = np.concatenate((insols, [6.67, 1.91, 1.0, 0.43, 0.037,
                                          0.011, 0.0027, 0.0011]))
        exoearths = np.concatenate((exoearths, [False]*8))
        fullxcens = np.concatenate((fullxcens, np.zeros(8) + xcens[ii]))
        fullycens = np.concatenate((fullycens, np.zeros(8) + ycens[ii]))
        continue

    fd = np.where(kics == multikics[ii])[0]
    # get the values for this system
    usedkics = np.concatenate((usedkics, kics[fd]))
    t0s = np.concatenate((t0s, it0s[fd]))
    periods = np.concatenate((periods, pds[fd]))
    semis = np.concatenate((semis, semi[fd]))
    radii = np.concatenate((radii, radius[fd]))
    teqs = np.concatenate((teqs, iteqs[fd]))
    insols = np.concatenate((insols, iinsol[fd]))
    exoearths = np.concatenate((exoearths, ptypes[fd] == 15))
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
teqs = teqs[rs]
insols = insols[rs]
exoearths = exoearths[rs]
fullxcens = fullxcens[rs]
fullycens = fullycens[rs]

if makemovie:
    plt.ioff()
else:
    plt.ion()

# create the figure at the right size (this assumes a default pix/inch of 100)
fig = plt.figure(figsize=figsizes[reso])

# make the plot cover the entire figure with the right background colors
ax = fig.add_axes([0.0, 0, 1, 1])
ax.axis('off')
fig.patch.set_facecolor(bkcol)
plt.gca().patch.set_facecolor(bkcol)

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
#sr = 16. / 9.
sr = figsizes[reso][0] / figsizes[reso][1]
# make the aspect ratio exactly right
if (xmax - xmin) / (ymax - ymin) > sr:
    plt.xlim(xmin, xmax)
    plt.ylim((ymax + ymin) / 2. - (xmax - xmin) / (2. * sr),
             (ymax + ymin) / 2. + (xmax - xmin) / (2. * sr))
else:
    plt.ylim(ymin, ymax)
    plt.xlim((xmax + xmin) / 2. - (ymax - ymin) * sr / 2.,
             (xmax + xmin) / 2. + (ymax - ymin) * sr / 2.)

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

# set up the planet size scale
sscale = sscales[reso]

rearth = 1.
rnep = 3.856
rjup = 10.864
rmerc = 0.383
# for the planet size legend
solarsys = np.array([rmerc, rearth, rnep, rjup])
pnames = ['Mercury', 'Exo-Earth', 'Neptune', 'Jupiter']
# csolar = np.array([409, 255, 46, 112])
csolar = np.log10([6.67, 1, 0.0011, 0.037])

# keep the smallest planets visible and the largest from being too huge
solarsys = np.clip(solarsys, 0.8, 1.3 * rjup)
solarscale = sscale * solarsys

radii = np.clip(radii, 0.8, 1.3 * rjup)
pscale = sscale * radii

"""
# color bar temperature tick values and labels
ticks = np.array([250, 500, 750, 1000, 1250])
if notext:
    labs = ['']*len(ticks)
else:
    labs = ['250', '500', '750', '1000', '1250', '1500']
"""  

# color bar temperature tick values and labels
ticks = np.array([-1.5, -1, 0, 1, 1.5])
if notext:
    labs = ['']*len(ticks)
else:
    labs = ['0.03', '0.1', '1', '10', '30']

# blue and red colors for the color bar
RGB1 = np.array([1, 185, 252])
RGB2 = np.array([220, 55, 19])

# create the diverging map with a white in the center
mycmap = diverge_map(RGB1=RGB1, RGB2=RGB2, numColors=15)

# just plot the planets at time 0. for this default plot
phase = 2. * np.pi * (0. - t0s) / periods

ie = exoearths
tmp = plt.scatter(fullxcens[~ie] + semis[~ie] * np.cos(phase[~ie]),
                  fullycens[~ie] + semis[~ie] * np.sin(phase[~ie]), marker='o',
                  edgecolors='none', lw=0, s=pscale[~ie], c=np.log10(insols[~ie]), vmin=ticks.min(),
                  vmax=ticks.max(), zorder=3, cmap=mycmap, clip_on=False)

tmp2 = plt.scatter(fullxcens[ie] + semis[ie] * np.cos(phase[ie]),
                  fullycens[ie] + semis[ie] * np.sin(phase[ie]), marker='o',
                  edgecolors='none', lw=0, s=pscale[ie], c=earthcol, 
                  zorder=3, clip_on=False)


fsz1 = fszs1[reso]
fsz2 = fszs2[reso]
prop = fm.FontProperties(fname=fontfile)

# create the 'Solar System' text identification
if addsolar:
    loc = np.where(usedkics == kicsolar)[0][0]
    if not notext:
        plt.text(fullxcens[loc], fullycens[loc], 'Solar\nSystem', zorder=-2,
                 color=fontcol, family=fontfam, fontproperties=prop, fontsize=fsz1,
                 horizontalalignment='center', verticalalignment='center')

# if we're putting in a translucent background behind the text
# to make it easier to read
if legback:
    # create the rectangles at the right heights and widths
    # based on the resolution
    c = plt.Rectangle(box1starts[reso], box1widths[reso], box1heights[reso],
                      alpha=legalpha, fc=legbackcol, ec='none', zorder=4,
                      transform=ax.transAxes)
    d = plt.Rectangle(box2starts[reso], box2widths[reso], box2heights[reso],
                      alpha=legalpha, fc=legbackcol, ec='none', zorder=4,
                      transform=ax.transAxes)
    ax.add_artist(c)
    ax.add_artist(d)

# appropriate spacing from the left edge for the color bar
cbxoff = cbxoffs[reso]

# plot the solar system planet scale
ax.scatter(np.zeros(len(solarscale)) + cbxoff,
           1. - 0.13 + 0.03 * np.arange(len(solarscale)), s=solarscale,
           c=csolar, zorder=5, marker='o',
           edgecolors='none', lw=0, cmap=mycmap, vmin=ticks.min(),
           vmax=ticks.max(), clip_on=False, transform=ax.transAxes)

ax.scatter(np.zeros(1) + cbxoff,
           1. - 0.13 + 0.03 * np.array([1]), s=solarscale[1],
           c=earthcol, zorder=5, marker='o',
           edgecolors='none', lw=0, clip_on=False, transform=ax.transAxes)


# put in the text labels for the solar system planet scale
for ii in np.arange(len(solarscale)):
    if not notext:
        ax.text(cbxoff + 0.01, 1. - 0.14 + 0.03 * ii,
                pnames[ii], color=fontcol, family=fontfam,
                fontproperties=prop, fontsize=fsz1, zorder=5,
                transform=ax.transAxes)

# colorbar axis on the left centered with the planet scale
ax2 = fig.add_axes([cbxoff - 0.005, 0.54, 0.01, 0.3])
ax2.set_zorder(2)
cbar = plt.colorbar(tmp, cax=ax2, extend='both', ticks=ticks)
# remove the white/black outline around the color bar
cbar.outline.set_linewidth(0)
# allow two different tick scales
cbar.ax.minorticks_on()
# turn off tick lines and put the physical temperature scale on the left
cbar.ax.tick_params(axis='y', which='major', color=fontcol, width=2,
                    left=False, right=False, length=5, labelleft=True,
                    labelright=False, zorder=5)
# turn off tick lines and put the physical temperature approximations
# on the right
cbar.ax.tick_params(axis='y', which='minor', color=fontcol, width=2,
                    left=False, right=False, length=5, labelleft=False,
                    labelright=True, zorder=5)
# say where to put the physical temperature approximations and give them labels
cbar.ax.yaxis.set_minor_locator(FL(tmp.norm([255, 409, 730, 1200])))
cbar.ax.set_yticklabels(labs, color=fontcol, family=fontfam,
                        fontproperties=prop, fontsize=fsz1, zorder=5)
if not notext:
    cbar.ax.set_yticklabels(['Earth', 'Mercury', 'Surface\nof Venus', 'Lava'],
                            minor=True, color=fontcol, family=fontfam,
                            fontproperties=prop, fontsize=fsz1)
    # clab = 'Planet Equilibrium\nTemperature (K)'
    clab = 'Insolation'
    # add the overall label at the bottom of the color bar
    cbar.ax.set_xlabel(clab, color=fontcol, family=fontfam, fontproperties=prop,
                       size=fsz1, zorder=5)

# switch back to the main plot
plt.sca(ax)

txtxoff = txtxoffs[reso]
txtyoff1 = txtyoffs1[reso]
txtyoff2 = txtyoffs2[reso]

if not notext:
    # put in the credits in the top right
    text = plt.text(1. - txtxoff, 1. - txtyoff1,
                    time0.strftime('LUVOIR Orrery\n%d %b %Y'), color=fontcol,
                    family=fontfam, fontproperties=prop,
                    fontsize=fsz2, zorder=5, transform=ax.transAxes)
    plt.text(1. - txtxoff, 1. - txtyoff2, 'By Ethan Kruse\n@ethan_kruse',
             color=fontcol, family=fontfam,
             fontproperties=prop, fontsize=fsz1,
             zorder=5, transform=ax.transAxes)

# the center of the figure
x0 = np.mean(plt.xlim())
y0 = np.mean(plt.ylim())

# width of the figure
xdiff = np.diff(plt.xlim()) / 2.
ydiff = np.diff(plt.ylim()) / 2.

# create the output directory if necessary
if makemovie and not os.path.exists(outdir):
    os.mkdir(outdir)

if makemovie:
    # get rid of all old png files so they don't get included in a new movie
    oldfiles = glob(os.path.join(outdir, 'fig*png'))
    for delfile in oldfiles:
        os.remove(delfile)

    # go through all the times and make the planets move
    for ii, time in enumerate(times):
        # remove old planet locations and dates
        tmp.remove()
        tmp2.remove()
        

        # re-zoom to appropriate level
        plt.xlim([x0s[ii] - xdiff * zooms[ii], x0s[ii] + xdiff * zooms[ii]])
        plt.ylim([y0s[ii] - ydiff * zooms[ii], y0s[ii] + ydiff * zooms[ii]])

        newt = time0 + dt.timedelta(time)
        if not notext:
            # put in the credits in the top right
            text.remove()
            text = plt.text(1. - txtxoff, 1. - txtyoff1,
                            newt.strftime('LUVOIR Orrery\n%d %b %Y'),
                            color=fontcol, family=fontfam,
                            fontproperties=prop,
                            fontsize=fsz2, zorder=5, transform=ax.transAxes)
        # put the planets in the correct location
        phase = 2. * np.pi * (time - t0s) / periods        
        tmp = plt.scatter(fullxcens[~ie] + semis[~ie] * np.cos(phase[~ie]),
                  fullycens[~ie] + semis[~ie] * np.sin(phase[~ie]), marker='o',
                  edgecolors='none', lw=0, s=pscale[~ie], c=np.log10(insols[~ie]), vmin=ticks.min(),
                  vmax=ticks.max(), zorder=3, cmap=mycmap, clip_on=False)

        tmp2 = plt.scatter(fullxcens[ie] + semis[ie] * np.cos(phase[ie]),
                  fullycens[ie] + semis[ie] * np.sin(phase[ie]), marker='o',
                  edgecolors='none', lw=0, s=pscale[ie], c=earthcol, 
                  zorder=3, clip_on=False)
        

        # add back facecolor and edgecolor if don't want transparent
        plt.savefig(os.path.join(outdir, 'fig{0:04d}.png'.format(ii)),
                    transparent=False, facecolor=fig.get_facecolor(), edgecolor='none')
        if not (ii % 10):
            print(f'{ii} of {len(times)} frames')
