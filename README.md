# AIChallenge2022

## Installation

Install dependencies in the `requirements.txt` first then run `pip install -e . ` to install this library from source code.

## Calculate the similarity between a sentence and a paragraph

```python
from src.nlp.calc_sim import calc_sim

sentence = "An terrible accident happened in the crowded street."
paragraph = "This morning, a car crashed into a group of people, killing 5 people and injuring 10 others. The driver of the car was arrested by the police."

sim = calc_sim(sentence, paragraph)

print(f'Sencence: {sentence}')
print(f'Paragraph: {paragraph}')
print(f'Similarity: {sim}')
```