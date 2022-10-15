# AIChallenge2022

## Using this code to clone the repo
`git clone https://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@github.com/cubist38/AIChallenge2022.git`

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