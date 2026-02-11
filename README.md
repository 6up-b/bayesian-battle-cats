## Battle Cats Bayesian Seed Tracker

Seed tracking and prediction tool for Battle Cats gacha banners. Reconstruct possible RNG states from observed rolls and predicts future rolls using
- exact seed matching
- Bayesian posterior updates
- entropy based uncertainty estimation
- conditional prediciton from mixture

## About
Battle Cats uses deterministic RNG. Given enough observed rolls, the RNG seed is uniquely identifiable. Over time the posterior collapses into a single seed.

When seed tracking, the usual method is to brute force for the true seed. 
Sometimes though you can get more informative roll sequences that do not need to be brute forced. You can be informed on the value of the next cat from the pool of remaining possible seeds, without knowing the true seed value. This can help inform you early whether to invest more gold rare tickets into a given campaign to find your actual seed or find out early if all the possibilities are dogshit. 

You can use the remaining entropy to see how many gold rare tickets you would need to find your real seed value.

Note that the reroll logic depends on previous roll and posterior tracks seed only

## Installation
- python 3.9
- cython
- setuptools
```python
pip install cython setuptools

```
I included my cython c file for gnu/linux people. For windows you have to build the seed seeker using setup.py. 
```python
python setup.py build_ext --inplace
```
## Usage
Enter your rolls into rolls.txt and the paste of the gacha infomration (probabilities) into paste.txt (default naming). You can get the scraped information from BC godfat.

Once you save your posterior as a pickle using --use-pickle, rolls.txt will be used for the rolls following whatever you made to generate the posterior.

```python
usage: bc_bayes_exact.py [-h] [--paste PASTE] [--posterior-pkl POSTERIOR_PKL]
                         [--use-pickle] [--save-pickle] [--start START]
                         [--end END] [--max-found MAX_FOUND]
                         [--predict-k PREDICT_K] [--top-k TOP_K]
                         [--print-pos PRINT_POS] [--show-cat SHOW_CAT]
                         [--entropy] [--uber-top UBER_TOP]
                         [--observed-file OBSERVED_FILE] [--estimate-rolls]

options:
  -h, --help            show this help message and exit
  --paste PASTE
  --posterior-pkl POSTERIOR_PKL
  --use-pickle          Load prior posterior from pickle (if present).
  --save-pickle         Save updated posterior to pickle after this update.
  --start START         Start seed (inclusive) for seeking (only used if no
                        prior pickle).
  --end END             End seed (exclusive) for seeking (only used if no
                        prior pickle).
  --max-found MAX_FOUND
                        Stop after this many matches (0 = all in range).
  --predict-k PREDICT_K
                        Number of future rolls to predict
  --top-k TOP_K         Number of top outcomes to display per roll
  --print-pos PRINT_POS
                        How many future roll positions to print
  --show-cat SHOW_CAT   Print probability of a specific cat at each roll.
                        Repeatable. Special value "Ubers" prints P(any Uber) +
                        which Ubers.
  --entropy             Print Shannon entropy (bits) per roll to visualize
                        uncertainty collapse.
  --uber-top UBER_TOP   When using --show-cat Ubers, show this many top Ubers
                        per roll.
  --observed-file OBSERVED_FILE
                        Path to a text file containing observed rolls, comma-
                        separated. Example: "Gold Cat, Pirate Cat"
  --estimate-rolls      Estimate how many additional rolls are needed to
                        identify the exact seed, using posterior entropy H(S)
                        and next-roll entropy H(Y).```

```python python3 bc_bayes_exact.py   --start 0 --end 2000000000   --predict-k 30   --print-pos 20   --entropy   --estimate-rolls --show-cat "Uber" --paste paste_plat.txt ```




