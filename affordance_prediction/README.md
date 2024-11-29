# Apartment affordance model

The `scripts/affordance_model.py` script reads in data and trains a DNN model, saving the model parameters.
In order to sun correctly, it needs to be placed in the following folder structure:

```
- scripts
    - affordance_model.py
- data
    - input
        - shape.csv
        - edge_64.csv
        - vertex_64.csv
    - output
        - output.csv
- models
```
