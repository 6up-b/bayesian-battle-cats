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
                        and next-roll entropy H(Y).
```
## Example
```python
python3 bc_bayes_exact.py   --start 0 --end 2000000000   --predict-k 30   --print-pos 20   --entropy   --estimate-rolls --show-cat "Uber" --paste paste_plat.txt
```


## Technical slop

Battle Cats uses a deterministic pseudorandom number generator. Each roll is produced by advancing the seed and mapping values to: 
1. rarity
2. unit slot
3. reroll logic (for duplicate handling)

$$ s_{t+1} = f(s_t) $$

Where $f$ is the Xorshift funtion

$$s \oplus = s \ll 13 $$
$$ s \oplus = s \gg 17 $$
$$ s \oplus =s \ll 15 $$
This transformation is deterministic and invertible in the 32 bit state space.
$$ \text{roll outcome} = g(s_t) $$

Observed roll constraints:

$$g(f^{(1)}(s)) = y_1$$
$$g(f^{(2)}(s)) = y_2$$
$$\ldots$$
$$g(f^{(n)}(s)) = y_n $$

The seed finder searches for $` s \in [0,2^{32}]) : \text{given rolls} `$ to produce a set of candidate seeds $` S = \{s_1, s_2, ..., s_k \} `$
if $k=1$ the seed is uniquely determined and the value of all future rolls is deterministic.

To predict futur rolls, we have a mixture distribution $$P(Y_{t+k}) = \sum_{s \in S} P(Y_{t+k} | s) P(s) $$.

Since each seed produces exactly 1 outcome: 

$$P(Y_{t+k} = y) = \sum_{s : g(f^{(k)}(s))=y} P(s) $$ 

for next roll, roll+2, ..., roll+k

## Entropy as a measure of uncertainty

Posterior uncertanty is measured via
$$H(S) = -\sum_s P(s) \log_2 P(s) $$

Effective candidate count:
$$N_{\text{eff}} = 2^{H(S)} $$

Uncertainty in the next roll:

$$H(Y) = -\sum_y P(y) \log_2 P(y) $$
Mapping from seed to outcome is deterministic so each roll reveals information about the seed.

To approximate the rolls needed to determine the seed using bits of information gained from the previous rolls,
$$\text{rolls remaining} \\approx \frac{H(S)}{H(Y)} $$
assuming average case information gain

Predictions collapse quickly because observed rolls eliminate seeds exponentially
Each roll partitions the seed space
$$|S_{n+1}| \approx \frac{|S_n|}{\text{effective branching factor}} $$

rare or distinctive outcomes will prune more seeds.

System is like a degenerate HMM where the hidden state is the RNG true seed, we use rolled cats as observations with deterministic PRNG transitions, and a deterministic mapping emission. Inference usually reduces to filtering after around 4 rolls. 








