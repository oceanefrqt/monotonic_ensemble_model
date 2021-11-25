# Monotonic Models for Classification Tasks in Systems Biology

Author: Océane FOURQUET

This project aims to construct a metamodel composed of an ensemble of monotonic classification models[1](https://link.springer.com/article/10.1007/s00453-012-9628-4). In addition to reimplementing an established approach[2](https://academic.oup.com/jid/article/217/11/1690/4911472?login=true), it integrates new strategies to select the ensemble of classifiers and integrate a notion of uncertainty in the monotonic models.

## Python and librairies versions
- Python 3.9.1
- Pandas 1.2.3
- Numpy 1.19.2
- Matplotlib 3.3.4

## Run the code
The code is split in 3 stages. Stage 1: determine the optimal number of classifiers to construct the metamodel. Stage 2: compute an estimate of the AUC score of a metamodel constructed with k_opt classifiers. Stage 3: construct the final metamodel from the whole dataset. The three stages are coded in the py file stages.py, available in the Module.

To run the whole project, you can run full\_project.py with as follow:

python3 full\_project.py <dataset> <probeset> <outpath>  

ex : python3 full\_project.py dengue/dengue.csv dengue/probeset_annotations.txt Result/ > log.txt

To run only one stage, you can use the relevant py file among stage1.py, stage2.py and stage3.py.

Note that by default, this code uses the 3-classes classification (severe, non severe and uncertain). If you want to compute the metamodel by favoring severe or non severe (2-classes classification), you must change the parameter in the file full\_project.py. 


## Files in Module
- cost\_matrix\_uncertainty.py
- measures.py
- monotonic\_regression\_uncertainty.py
- optimal\_k\_aggregations.py
- selection\_algorithm.py
- show\_results.py
- stages.py
- tools.py
