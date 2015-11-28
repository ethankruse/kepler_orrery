#!/bin/bash
# first argument is directory with input/output files
# second argument is the name of the output gif (without the directory)
# third argument is the frame rate (fps)


# GIFs can only have 256 colors. But this makes a custom palette of the 256
# best colors to use for this particular GIF rather than a generic color palette
ffmpeg -i $1fig%04d.png -vf palettegen $1palette.png
# make the GIF using a particular frame rate.
ffmpeg -framerate $3 -i $1fig%04d.png -i $1palette.png -lavfi paletteuse -r $3 $1$2
