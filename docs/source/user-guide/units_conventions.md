# Units and conventions

In principle, DiaBayes does not perform any unit conversions internally, meaning that the output values have the same units as the input values.
By sticking to SI conventions, you will get that all length scales are measured in metres, time in seconds, speeds in metres per seconds, etc.
You are free to choose your own dimensions, as long as all of the input values are consistent; if you want to represent slip rate in feet per hour (_who am I to judge..._), then consequently $D_c$ must have units of feet and $\theta$ units of hours.
This also applies to experimental data loaded into the GUI; the columns of the CSV file need to have units that are consistent with those entered into the user input boxes.