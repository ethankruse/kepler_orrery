#!/usr/bin/env python
#------------------------------------------------------------------------------
# Name:        colorMapCreator.py
# Purpose:     Generate reasonable diverging colormaps using the technique
#              presented in "Diverging Color Maps for Scientific Visualization
#              (Expanded)" by Kenneth Moreland.
#
# Author:      Carlo Barth
#
# Created:     22.10.2013
# Copyright:   (c) 2013
#------------------------------------------------------------------------------

# main() (diverge_map) function modified by Ethan Kruse 2015
# to return a colormap directly. Also found some bugs, but am hacking around
# that for now

# Imports
import numpy as np



# =============================================================================
# ====================== The Class ColorMapCreator ============================
# =============================================================================


class ColorMapCreator:
    """
    Class ColorMapCreator:
    Create diverging colormaps from RGB1 to RGB2 using the method of Moreland
    or a simple CIELAB-interpolation. numColors controls the number of color
    values to output (odd number) and divide gives the possibility to output
    RGB-values from 0.0-1.0 instead of 0-255. If a filename different than
    "" is given, the colormap will be saved to this file, otherwise a simple
    output using print will be given.
    """

    # ======================== Global Variables ===============================

    # Reference white-point D65
    Xn, Yn, Zn = [95.047, 100.0, 108.883] # from Adobe Cookbook

    # Transfer-matrix for the conversion of RGB to XYZ color space
    transM = np.array([[0.4124564, 0.2126729, 0.0193339],
                        [0.3575761, 0.7151522, 0.1191920],
                        [0.1804375, 0.0721750, 0.9503041]])


    # ============================= Functions =================================


    def __init__(self, RGB1, RGB2, numColors = 33., divide = 255.,
                  method = "moreland", filename = ""):

        # create a class variable for the number of colors
        self.numColors = numColors

        # assert an odd number of points
        assert np.mod(numColors,2) == 1, \
            "For diverging colormaps odd numbers of colors are desireable!"

        # assert a known method was specified
        knownMethods = ["moreland", "lab"]
        assert method in knownMethods, "Unknown method was specified!"

        if method == knownMethods[0]:
            #generate the Msh diverging colormap
            self.colorMap = self.generateColorMap(RGB1, RGB2, divide)
        elif method == knownMethods[1]:
            # generate the Lab diverging colormap
            self.colorMap = self.generateColorMapLab(RGB1, RGB2, divide)

        # print out the colormap of save it to file named filename
        if filename == "":
            for c in self.colorMap:
                    pass
                    # print "{0}, {1}, {2}".format(c[0], c[1], c[2])
        else:
            with open(filename, "w") as f:
                for c in self.colorMap:
                    f.write("{0}, {1}, {2}\n".format(c[0], c[1], c[2]))
    #-

    def rgblinear(self, RGB):
        """
        Conversion from the sRGB components to RGB components with physically
        linear properties.
        """

        # initialize the linear RGB array
        RGBlinear = np.zeros((3,))

        #  calculate the linear RGB values
        for i,value in enumerate(RGB):
            value = float(value) / 255.
            if value > 0.04045 :
                value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
            else :
                value = value / 12.92
            RGBlinear[i] = value * 100.
        return RGBlinear
    #-

    def sRGB(self, RGBlinear):
        """
        Back conversion from linear RGB to sRGB.
        """

        # initialize the sRGB array
        RGB = np.zeros((3,))

        #  calculate the sRGB values
        for i,value in enumerate(RGBlinear):
            value = float(value) / 100.

            if value > 0.00313080495356037152:
                value = (1.055 * np.power(value,1./2.4) ) - 0.055
            else :
                value = value * 12.92

            RGB[i] = round(value * 255.)
        return RGB
    #-

    def rgb2xyz(self, RGB):
        """
        Conversion of RGB to XYZ using the transfer-matrix
        """
        return np.dot(self.rgblinear(RGB), self.transM)
    #-

    def xyz2rgb(self, XYZ):
        """
        Conversion of RGB to XYZ using the transfer-matrix
        """
        #return np.round(np.dot(XYZ, np.array(np.matrix(transM).I)))
        return self.sRGB(np.dot(XYZ, np.array(np.matrix(self.transM).I)))
    #-

    def rgb2Lab(self, RGB):
        """
        Conversion of RGB to CIELAB
        """

        # convert RGB to XYZ
        X, Y, Z = (self.rgb2xyz(RGB)).tolist()

        # helper function
        def f(x):
            limit = 0.008856
            if x> limit:
                return np.power(x, 1./3.)
            else:
                return 7.787*x + 16./116.

        # calculation of L, a and b
        L = 116. * ( f(Y/self.Yn) - (16./116.) )
        a = 500. * ( f(X/self.Xn) - f(Y/self.Yn) )
        b = 200. * ( f(Y/self.Yn) - f(Z/self.Zn) )
        return np.array([L, a, b])
    #-

    def Lab2rgb(self, Lab):
        """
        Conversion of CIELAB to RGB
        """

        # unpack the Lab-array
        L, a, b = Lab.tolist()

        # helper function
        def finverse(x):
            xlim = 0.008856
            a = 7.787
            b = 16./116.
            ylim = a*xlim+b
            if x > ylim:
                return np.power(x, 3)
            else:
                return ( x - b ) / a

        # calculation of X, Y and Z
        X = self.Xn * finverse( (a/500.) + (L+16.)/116. )
        Y = self.Yn * finverse( (L+16.)/116. )
        Z = self.Zn * finverse( (L+16.)/116. - (b/200.) )

        # conversion of XYZ to RGB
        return self.xyz2rgb(np.array([X,Y,Z]))
    #-

    def Lab2Msh(self, Lab):
        """
        Conversion of CIELAB to Msh
        """

        # unpack the Lab-array
        L, a, b = Lab.tolist()

        # calculation of M, s and h
        M = np.sqrt(np.sum(np.power(Lab, 2)))
        s = np.arccos(L/M)
        h = np.arctan2(b,a)
        return np.array([M,s,h])
    #-

    def Msh2Lab(self, Msh):
        """
        Conversion of Msh to CIELAB
        """

        # unpack the Msh-array
        M, s, h = Msh.tolist()

        # calculation of L, a and b
        L = M*np.cos(s)
        a = M*np.sin(s)*np.cos(h)
        b = M*np.sin(s)*np.sin(h)
        return np.array([L,a,b])
    #-

    def rgb2Msh(self, RGB):
        """ Direct conversion of RGB to Msh. """
        return self.Lab2Msh(self.rgb2Lab(RGB))
    #-

    def Msh2rgb(self, Msh):
        """ Direct conversion of Msh to RGB. """
        return self.Lab2rgb(self.Msh2Lab(Msh))
    #-

    def adjustHue(self, MshSat, Munsat):
        """
        Function to provide an adjusted hue when interpolating to an
        unsaturated color in Msh space.
        """

        # unpack the saturated Msh-array
        Msat, ssat, hsat = MshSat.tolist()

        if Msat >= Munsat:
            return hsat
        else:
            hSpin = ssat * np.sqrt(Munsat**2 - Msat**2) / \
                    (Msat * np.sin(ssat))
            if hsat > -np.pi/3:
                return hsat + hSpin
            else:
                return hsat - hSpin
    #-

    def interpolateColor(self, RGB1, RGB2, interp):
        """
        Interpolation algorithm to automatically create continuous diverging
        color maps.
        """

        # convert RGB to Msh and unpack
        Msh1 = self.rgb2Msh(RGB1)
        M1, s1, h1 = Msh1.tolist()
        Msh2 = self.rgb2Msh(RGB2)
        M2, s2, h2 = Msh2.tolist()

        # If points saturated and distinct, place white in middle
        if (s1>0.05) and (s2>0.05) and ( np.abs(h1-h2) > np.pi/3. ):
            Mmid = max([M1, M2, 88.])
            if interp < 0.5:
                M2 = Mmid
                s2 = 0.
                h2 = 0.
                interp = 2*interp
            else:
                M1 = Mmid
                s1 = 0.
                h1 = 0.
                interp = 2*interp-1.

        # Adjust hue of unsaturated colors
        if (s1 < 0.05) and (s2 > 0.05):
            h1 = self.adjustHue(np.array([M2,s2,h2]), M1)
        elif (s2 < 0.05) and (s1 > 0.05):
            h2 = self.adjustHue(np.array([M1,s1,h1]), M2)

        # Linear interpolation on adjusted control points
        MshMid = (1-interp)*np.array([M1,s1,h1]) + \
                 interp*np.array([M2,s2,h2])

        return self.Msh2rgb(MshMid)
    #-

    def generateColorMap(self, RGB1, RGB2, divide):
        """
        Generate the complete diverging color map using the Moreland-technique
        from RGB1 to RGB2, placing "white" in the middle. The number of points
        given by "numPoints" controls the resolution of the colormap. The
        optional parameter "divide" gives the possibility to scale the whole
        colormap, for example to have float values from 0 to 1.
        """

        # calculate
        scalars = np.linspace(0., 1., self.numColors)
        RGBs = np.zeros((self.numColors, 3))
        for i,s in enumerate(scalars):
            RGBs[i,:] = self.interpolateColor(RGB1, RGB2, s)
        return RGBs/divide
    #-

    def generateColorMapLab(self, RGB1, RGB2, divide):
        """
        Generate the complete diverging color map using a transition from
        Lab1 to Lab2, transitioning true RGB-white. The number of points
        given by "numPoints" controls the resolution of the colormap. The
        optional parameter "divide" gives the possibility to scale the whole
        colormap, for example to have float values from 0 to 1.
        """

        # convert to Lab-space
        Lab1 = self.rgb2Lab(RGB1)
        Lab2 = self.rgb2Lab(RGB2)
        LabWhite = np.array([100., 0., 0.])

        # initialize the resulting arrays
        Lab = np.zeros((self.numColors ,3))
        RGBs = np.zeros((self.numColors ,3))
        N2 = np.floor(self.numColors/2.)

        # calculate
        for i in range(3):
            Lab[0:N2+1, i] = np.linspace(Lab1[i], LabWhite[i], N2+1)
            Lab[N2:, i] = np.linspace(LabWhite[i], Lab2[i], N2+1)
        for i,l in enumerate(Lab):
            RGBs[i,:] = self.Lab2rgb(l)
        return RGBs/divide
    #-

# =============================================================================
# ========================== The Main-Function ================================
# =============================================================================


# define the initial and final RGB-colors (low and high end of the diverging
# colormap
def diverge_map(RGB1=np.array([59, 76, 192]), RGB2=np.array([180, 4, 38]),
                numColors=101):
    # create a new instance of the ColorMapCreator-class using the desired
    # options
    colormap = ColorMapCreator(RGB1, RGB2, numColors=numColors)
    # there's clearly some bugs since it's possible to get values > 1
    # e.g. with starting values RGB1 = [1,185,252], RGB2 = [220, 55, 19],
    # numColors > 3
    # but this is good enough for now
    colormap.colorMap = np.clip(colormap.colorMap, 0, 1)

    cdict = {'red': [], 'green': [], 'blue': []}
    inds = np.linspace(0.,1.,numColors)

    # create a matplotlib colormap
    for ii, ind in enumerate(inds):
        cdict['red'].append([ind, colormap.colorMap[ii, 0],
                             colormap.colorMap[ii, 0]])
        cdict['green'].append([ind, colormap.colorMap[ii, 1],
                               colormap.colorMap[ii, 1]])
        cdict['blue'].append([ind, colormap.colorMap[ii, 2],
                              colormap.colorMap[ii, 2]])

    from matplotlib.colors import LinearSegmentedColormap
    mycmap = LinearSegmentedColormap('BlueRed1', cdict)

    return mycmap
