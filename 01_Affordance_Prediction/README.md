## TODO: give a brief description of input files

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
- models  # model object is saved here
```

## File details

### `/input`

These files contain predictor features...

- `shape.csv`
- `edge_64.csv`
- `vertex_64.csv`

### `/output/output.csv`

This file contains features that the model learns to predict:

- `bld_flr_apt` - `char`; Unique apartment ID (not to be predicted)
- `number_of_rooms` - `int`; Number of living-, bed-, lounge- (_etc._) rooms
- `corridor_area_ratio` - `float`; Circulation efficiency (ratio of corridor area / total area)
- `largest_room_sunlight` - `float`; Simulated mean sunlight in the largest living room (in lux)
- `largest_room_noise` - `float`; Simulated mean noise level in the largest living room (in dB)
- `largest_room_kitchen_distance` - `float`; Distance between kitchen and the largest living room (in m)
- `kitchen_sunlight` - `float`; Simulated mean sunlight in the kitchen (in lux)
- `has_second_bathroom` - `[0, 1]`; Does the apartment have a second bathroom
- `bathroom_has_window` - `[0, 1]`; Does the apartment have at least one bathroom with a window
- `has_loggia` - `[0, 1]`; Does the apartment have a loggia
- `has_balcony` - `[0, 1]`; Does the apartment have a balcony
