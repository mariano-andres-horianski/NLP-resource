import datetime
import json
import plac
import random
import spacy
#import warnings
#warnings.filterwarnings("ignore")


from pathlib import Path
from spacy.pipeline import EntityRecognizer, Tagger
from spacy.training import Example
from spacy.tokens import Doc
from collections import Counter, defaultdict

from train_data import TRAIN_DATA
from train_data import LABELS as NEW_LABELS



# remove existing labels from the model
for label in ["PER", "ORG", "LOC", "MISC"]:
    try:
        NEW_LABELS.remove(label)
    except KeyError:
        continue
print(f"Training set length: {len(TRAIN_DATA)}")

BATCH_SIZE = 10

@plac.annotations(
    model=("Model name. Defaults to blank 'es' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))


def main(model='es_core_news_lg',
         new_model_name='es_core_news_lg_ext',
         output_dir='es_core_news_lg_ext',
         n_iter=100):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Modelo cargado '%s'" % model)
    else:
        nlp = spacy.blank('es')  # create blank Language class
        print("Modelo en blanco 'es' creado")

    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last = True)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    for label in NEW_LABELS:
        ner.add_label(label)

    losses_max = 99999999999
    t0 = datetime.datetime.now()
    print("Start: ", t0)

    # load test text
    validation_data = list(filter(None, open('test_text.txt','r', encoding="utf-8").read().splitlines()))
    iteration = 0
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.resume_training()

        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            iteration = iteration + 1
            losses = {}
            t_it0 = datetime.datetime.now()
            #for raw_text, entity_offsets in TRAIN_DATA:
            #    nlp.update([raw_text], [entity_offsets], drop=0.25, sgd=optimizer, losses=losses)
            # ----------------------
            # for i in range(int(len(TRAIN_DATA) / 3)):
            #     raw_texts = TRAIN_DATA[i][0]
            #     entity_offsets = TRAIN_DATA[i][1]
            #     nlp.update(raw_texts, entity_offsets, drop=0.25, sgd=optimizer, losses=losses)
            # -----------------------
            # for i in range(int(len(TRAIN_DATA) / 3)):
            #     raw_texts = [TRAIN_DATA[i * 3][0], TRAIN_DATA[i * 3 + 1][0], TRAIN_DATA[i * 3 + 2][0]]
            #     entity_offsets = [TRAIN_DATA[i * 3][1], TRAIN_DATA[i * 3 + 1][1], TRAIN_DATA[i * 3 + 2][1]]
            #     nlp.update(raw_texts, entity_offsets, drop=0.25, sgd=optimizer, losses=losses)
            # -----------------------
            
            for i in range(int(len(TRAIN_DATA) / BATCH_SIZE)):
                raw_texts = [TRAIN_DATA[i * BATCH_SIZE + x][0] for x in range(BATCH_SIZE)]
                entity_offsets = [TRAIN_DATA[i * BATCH_SIZE + x][1] for x in range(BATCH_SIZE)]

                # print(raw_texts[0])
                # print(entity_offsets[0])
                # print()
                # print()
                # print()

                examples = [
                    Example.from_dict(nlp.make_doc(rt), eo)
                    for rt, eo in zip(raw_texts, entity_offsets)
                ]
                nlp.update(examples, drop=0.25, sgd=optimizer, losses=losses)
            t_it1 = datetime.datetime.now()
            print("Iteration:" + str(iteration))
            print(losses)
            #for text in validation_data:
            #    doc = nlp(text)
            #    ents = [(ent.text, ent.label_) for ent in doc.ents]
            #    for ent, label in ents:
            #        print(f'Found entity: "{ent}": "{label}",')
            print(f"Tiempo de iteración {t_it1 - t_it0}")
            # save model to output directory
            # take the most successful
            for a in losses.keys():
                if losses[a] < losses_max:
                    losses_max = losses[a]
                    if output_dir is not None:
                        output_dir = Path(output_dir)
                        if not output_dir.exists():
                            output_dir.mkdir()
                        nlp.meta['name'] = new_model_name  # rename model
                        nlp.to_disk(output_dir)
                        print("saved model: ", output_dir)
        
    # test the trained model
    print("Finished: ", datetime.datetime.now() - t0)
    print("\n\n\n")


if __name__ == '__main__':
    plac.call(main)
