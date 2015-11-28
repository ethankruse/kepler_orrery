# Kepler Orrery

This is the source code to create the Kepler orrery featured in
[this video](https://www.youtube.com/watch?v=_DnDeBa0KFc).

Everything can be run using python assuming the following packages are
installed (most are defaults in every python installation):
* `os`
* `datetime`
* `numpy`
* `matplotlib`
* `glob`

To create the movie or gif however, ffmpeg must also be installed to run
the *.sh files.

All appropriate settings for the movie creation are listed at the top of
orrery.py and should be documented.

The movie should be recreated with the default settings by running

`python orrery.py`

`./makeorrery_movie.sh movie/ orrery_movie.mp4 30`
