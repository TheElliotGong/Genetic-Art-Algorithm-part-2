# Stained Glass Genetic Art Algorithm - Revised

This project is forked from Sebastian Proost's Voronoi-based Genetic Art Algorithm. We added image pre-processing, Voronoi shape outlining, and a divide-and-conquer approach.

## Running the code

To run the code in this repository, clone it, set up a virtual environment and install the required packages from 
requirements.txt

```bash

git clone https://github.com/4dcu-be/Genetic-Art-Algorithm-part-2
cd Genetic-Art-Algorithm-part-2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

Next you can run evolve.py and evolve_simple.py using

```bash

python evolve_voronoi.py

```

The paths to the target image and output directory are hard-coded, but can easily be changed. the lines are in the main
routine.

```python
    target_image_path = "./img/girl_with_pearl_earring_half.jpg"
    checkpoint_path = "./output/"
```

