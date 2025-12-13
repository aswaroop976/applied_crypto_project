# Applied-crypto final project repository

## To run detection avoidance attack:

- Put image dataset in src/images_in
- Outputted perturbed images stored in src/images_out

```
pip install -r requirements.txt

# To run pdq hash detection avoidance attack:
python3 src/pdq_gradient.py

# To run phash dectection avoidance attack:
python3 src/phash_gradient.py

```

## to run white box attacks
- two scripts, whitebox1, whitebox2
- whitebox1 contains functions necessary for whitebox2 to run image modification
- check paths of images in whitebox1/2, make sure they match where you want the images
- need to pip install the modules mentioned at header :)
- check location of whitebox1 functions when called in whitebox 2 to run
- email me if any quesitons bdschuch@purdue.edu
