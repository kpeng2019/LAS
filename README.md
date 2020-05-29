# LAS
This package is a brief wrap-up toolkit built based on 2 explanation packages: LIME and SHAP. The package contains 2 explainers: LIMEBAG and SHAP. It takes data and fitted models as input and returns explanations about feature importance ranks and/or weights. (etc. what attributes matter most within the prediction model).

## rq1.py
The demo runs LIMEBAG on a default dataset. It generates and presents explanations about feature importance ranks and weights for all testing data points.

## rq2.py
The demo uses the explanations returned from LIMEBAG to run an effect size test. A summary of feature importance ranks and weights will be generated and presented as output.
