import codecs
import json
import pprint
import spacy

from spacy.pipeline import merge_entities
from spacy.matcher import Matcher
from spacy.tokens import Span

nlp = spacy.load("es_core_news_lg")
matcher = Matcher(nlp.vocab)


def fix_label(label):
    """
    Fixes PERSON to PER label (so that doccano's PERSON matches spacy's PER)

    :param label: a label (str)
    :return: the correct label (str)
    """
    if label == "PERSON": return "PER"
    return label


def get_entities(labels, data, *, codec_decode="unicode_escape"):
    """
    Returns a dict of {text}:{label} for a labels list read from a doccano
    JSONL file.

    :param labels: the "label" key for a doccano-made JSONL file, it's a list
        of (start, end, label) entries
    :param data: the "data" key of a JSONL file, it's the whole text (and a JSON
        dictionary in itself) to which the "labels" key entries refer to
    :param codec_decode: (keyword only) controls which codec is used to decode
        the text from the "data" key. Use None if you don't want to perform
        codecs decoding
    :return: dict of entity:label to use with the other functions from this
        module to tag the texts
    """
    ret = {}
    for (start, end, label) in labels:
        text = data[start:end]
        if codec_decode is not None:
            text = codecs.decode(text, codec_decode)
        ret[text] = fix_label(label)
    return ret


def remove_overlapping(entities):
    """
    Removes overlapping entities from an entity list.

    :param entities: list of entities in a text
    :return: a list of entities, without overlapping entries
    """
    if not entities:
        return []  # a bit of a sanity check in case we get an empty list
    ret = []
    entities = sorted(entities, key=lambda x: x[0])
    prev_start, prev_end, prev_label = entities[0]
    ret.append((prev_start, prev_end, prev_label))
    for (start, end, label) in entities[1:]:
        # if start == prev_start:

        if start < prev_end:
            continue  # we don't want this one
        ret.append((start, end, label))
        prev_start, prev_end, prev_label = start, end, label
    return ret


def tag_data(data, entities):
    """
    Returns a list of tagged examples as spaCy requires, based on "data" (from
    doccano JSONL) and labels (from `get_labels()`).

    :param data: "data" key of doccano's JSONL
    :param entities: labels dictionary from `get_labels()`
    :return: list of tuples as expected by spaCy to train
    """
    ret = []
    for d in data:
        try:
            json_data = json.loads(d["data"])
        except TypeError:
            continue
        for k, v in json_data.items():
            for sublist in v:
                for phrase in sublist:
                    entry_ents = []
                    doc = merge_entities(nlp(phrase))
                    matches = matcher(doc)
                    for token in doc:
                        start = token.idx
                        end = start + len(token)
                        label = token.ent_type_
                        if not label:
                            continue
                        # if token.ent_type_ in {"PER", "LOC", "ORG"}:
                        #     label = token.ent_type_
                        # else:
                        #     # we try to find it in the tagged entities, falling
                        #     # back to "MISC" if we can't find it
                        #     label = entities.get(str(token))
                        #     if label is None:
                        #         continue
                        #     if token.ent_type_ == "MISC":
                        #         print("Tengo MISC...")
                        #     else:
                        #         print("Encontré por match...")
                        #     if label != "MISC":
                        #         print(f"    Cambiado por {label}.")
                        entry_ents.append((start, end, label))  
                    # for ent, label in entities.items():
                    #     pos = phrase.find(ent)
                    #     while pos >= 0:
                    #         start = pos
                    #         end = pos + len(ent)
                    #         entry_ents.append((start, end, label))
                    #         pos += 1
                    #         pos = phrase.find(ent, pos)
                    # entry_ents = remove_overlapping(entry_ents)
                    entry = (phrase, {"entities": entry_ents})
                    if len(entry_ents):
                        ret.append(entry)
    return ret

def make_entity_adder(entity_type):
    def add_entity(matcher, doc, i, matches):
        match_id, start, end = matches[i]
        print(f"Setting {doc[start:end]} to {entity_type}")
        entity = Span(doc, start, end, label=entity_type)
        # let's see if we have an overlapping entity...
        doc_ents = list(doc.ents)
        overlaps = []
        for idx, ent in enumerate(doc.ents):
            if ent.start <= start <= ent.end:
                overlaps.append(idx)
        overlaps.sort()
        for ol in overlaps[::-1]:
            del doc_ents[ol]
        doc.ents = tuple(doc_ents)
        doc.ents += (entity,)
    return add_entity

def main():
    with open("mariano-est5-pps.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # TODO: load the JSONL format instead
    # first we get the entities...
    entities = {}
    for d in data:
        entities.update( get_entities(d["label"], d["data"]) )
    # let's add the matchers...
    LABELS = set(l for (p, l) in entities.items())
    for label in LABELS:
        patterns = [k for k, v in entities.items() if v == label]
        adder = make_entity_adder(label)
        for p in patterns:
            pattern = [{"ORTH": w for w in (s for s in p.split() if s) }]
            matcher.add(p.replace(" ", ""), [pattern], on_match=adder)
    # now we get the entries
    entries = tag_data(data, entities)
    # and now we print the file
    with open("train_data.py", "w", encoding="utf-8") as fout:
        LABELS = set(l for (p, l) in entities.items())
        fout.write("LABELS = \\\n")
        fout.write( "\n".join( 
            f"    {line}" for line in pprint.pformat(LABELS).split("\n") )
        )
        fout.write("\n\n")
        fout.write("TRAIN_DATA = \\\n")
        fout.write( "\n".join( 
            f"    {line}" for line in pprint.pformat(entries).split("\n") )
        )

if __name__ == "__main__":
    main()