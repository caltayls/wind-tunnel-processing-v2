This work was carried out to support the development of an emerging wind tunnel measuring technique - pressure sensitive paint (PSP). 
PSP is used to measure forces across a body inside a wind tunnel. As its name suggests, the paint is sensitive
to pressure meaning that the intensity of light changes (which can be recorded with specialised cameras) at different pressures. The intensity
data that PSP provides can then be processed to give a transient pressure field across a body which in turn can be processed to give
a force on a body.

The main benefit of this technique is that it gives a continuous pressure field across the body rather than a discreet points which is
true of conventional measuring techniques. This greater depth of information can uncover more insights into the aerodynamic behaviour 
of a body that may have otherwise been missed.

To convert light intensity data to a pressure field, conventional pressure taps are used to measure the pressure directly at discreet
locations across the body. An average of light intensity can then be taken around each pressure tap which is then paired with the
pressure of the respective pressure tap. With enough data, a regressional machine learning model can be fitted with the data to allow
any light intensity value to be converted to a pressure.

The first aim of the code found in this repo is to automate the process of correctly labelling the pressure taps in the raw PSP 
cine files for each wind tunnel test carried out (locating_ptaps.ipynb). Over 100 tests were carried out so doing this manually 
would have been a gruelling task. The second aim was then to create rings around every pressure tap so that an average psp intensity 
could be taken (find_mean_luminosity.ipynb).   


Further information about files found in the repo can be found below.

find_mean_luminosity.ipynb:
This script contains the means of
	- locating pressure taps for all configurations
	- creating a dataset of mean luminosity for each ptap
	- validation of method used to obtain mean luminosity


locating_ptaps.ipynb:
Creates a dataset of ptap names and their distance from every
other ptap. Distances were used to identify ptaps.



ptap_avgs_across_frames_VALIDATION.p:
This file is a pickled python dictionary containing average ptap luminosity
at a series of speicified frames. The dictionary was used to validate methods
used.



