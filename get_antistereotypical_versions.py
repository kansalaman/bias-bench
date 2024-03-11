source = '/home/ubuntu/224n/bias-bench/data/all_sentences.json'

import json
import os
from openai import OpenAI

client = OpenAI()

def get_deb_versions(sentences):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "user",
            "content": "<sentences>\n\neach of the sentences above might contain some stereotypical bias. If so, we want to convert it to another sentence that is almost identical to the sentence but is anti-stereotypical. It should not be neutral but should present an antistereotypical bias. you can only change nouns: proper or improper. the resultant sentence should make sense, it should not be rubbish. You can only make one change\n\nprint the output as a single json where key is the word that needs to be replaced (in any of the input sentences) and value is the word it needs to be replaced with to make the sentence anti-stereotypical"
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return json.loads(response.choices[0].message.content)


sentences = json.load(open(source))

max_tokens = 1000

batches = [[[],0]]
for i in range(0, len(sentences)):
    if batches[-1][1] + len(sentences[i].split()) > max_tokens:
        batches.append([[],0])
    else:
        batches[-1][0].append(sentences[i])
        batches[-1][1] += len(sentences[i].split())

print(len(batches))
for batch in batches:
    print(len(batch[0]))

debiased_sentences = {}
for batch in batches[:-1]:
    debiased_sentences.update(get_deb_versions(batch[0]))

json.dump(debiased_sentences, open('debiasing.json', 'w'), indent=2)