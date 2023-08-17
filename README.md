# Wind Tunnel Pressure Tap Labeller

A tool to automate pressure tap labelling for wind tunnel test data

## Motivation

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

The first aim of the tool is to automate the process of correctly labelling the pressure taps in the raw PSP 
cine files for each wind tunnel test carried out. Over 100 tests were carried out so doing this manually 
would have been a gruelling task. The second aim was then to create rings around every pressure tap so that an average psp intensity 
could be taken. The subsequent data were then fed into a supervised ML model (KNN) to produce an accurate (r^2 = 0.92) calibration model.   

## Usage

### Creating initial labelled dataset

To automate pressure tap labelling, a labelled dataset must first be created using data from one of the tests carried out.

Using the `LabelledDataset().create_labelled_dataset(file_paths=files, labelled_datset_file_path='store_in_labels.csv')` method, the user will first encounter an initial circle detection within a frame of the wind tunnel data (Figure 1).

<figure>
  <img src="snapshots\store_in_bay-default_circles.png" alt="Detecting circles">
  <figcaption>Figure 1</figcaption>
</figure>

Circles can be added or removed so that the circle finder does not return too many or too few (Figure 2).

<div style='display:flex'>
    <figure>
        <img src="snapshots\store_in_bay-insufficient_circles.png" alt="too few circles">
        <figcaption>Figure 2</figcaption>
    </figure>
    <figure>
        <img src="snapshots\store_in_bay-too_many_circles.png" alt="too many circles"/>
    </figure>
</div>

Once the user is happy with the circles detected, they will then be prompted to label each circle (Figure 3)

<figure>
    <img src="snapshots\store_in_bay-labelling_circles.png" alt="adding labels">
    <figcaption>Figure 3</figcaption>
</figure>

The user is then displayed the final result and has the option to correct any mislabelled circles (Figure 4)

<figure>
    <img src="snapshots\store_in_bay-labelled_circles.png" alt="adding labels">
    <figcaption>Figure 4</figcaption>
</figure>

Once labelling is completed, a csv file will be saved to the location specified within the method arguments. The csv file (Figure 5) contains each pressure tap label, coordinates, and distances from every other pressure tap - the latter is used within the algorithm used to automate pressure tap labelling for the remaining wind tunnel tests. 

<figure>
    <img src="snapshots\label_dataset.png" alt="adding labels">
<figcaption>Figure 5</figcaption>
</figure>



### Labelling remaining datasets

The `PressureTapLabeller` class can then be used to label pressure taps in every other wind tunnel test.
The `find_ptaps('62150-DPSP-Store-in-doors-on-sawtooth-LE-TE-Beta10')` method uses an algorithm to score the circles against each pressure tap. The highest scoring pressure tap is assigned to the circle. Any circles that have a score below a certain threshold are discarded. Figure 6 shows the scoring system where pressure tap IDs are columns and unlabelled circles are the indices. 

<figure>
    <img src="snapshots\scoring_circles.png" alt="scoring circles">
<figcaption>Figure 6</figcaption>
</figure>


This data is then used to generate a labelled pressure tap dataset, ready to be used for the next processing stage (Figure 7).

<figure>
    <img src="snapshots\predicted_ptap_labelling.png" alt="predicted pressure taps">
<figcaption>Figure 7</figcaption>
</figure>

A label plot is available during this process to ensure pressure taps are labelled correctly (Figure 8)

<figure>
    <img src="snapshots\predicted_ptaps.png" alt="predicted pressure taps">
<figcaption>Figure 8</figcaption>
</figure>
