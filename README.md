# Descriptive Statistics and Hypothesis Tests

The goal of this library is to provide a common interface for existing hypothesis tests across several frameworks as well as implement some new ones.  Additionally this library is here to provide a set of descriptive statistics that may be overlooked.

## Implemented Notions

* Timeseries Data

This is data with a timeseries component, therefore this data is collected over time.  Time will be one of the variables collected, on some scale

* Geo Spatial Data

This is data with a geospatial component, therefore this data is collected with coordinates of some kind.  It is likely latitude and longitude or some variant there of will be present in data sets of this kind.

* Numeric Data 

This is data without time or geospatial structure.  For now it is the blanket for all other data.  If a better name is decided upon, it will be used in this names place.

* Image Data

This is data from images that has been transformed via some feature engineering to numbers or possibly the raw pixels in some scale, likely RGB or Grey scale.

* Text Data

This is textual data that has been transformed via some feature engineering to numbers.


## Known Installation Issues

If you get `ModuleNotFoundError: No module named 'tkinter'`:

Since this library relies on pysal, it relies on tkinter.  Tkinter is a weird library in that it isn't pip installable.  On some systems it comes pre-installed, on others it does not.  For ubuntu the fix is:

`sudo apt-get install idle3`

This is the only way I've found to install tkinter for python 3.  This library does not offer python 2 support so I won't discuss installation for that version.

