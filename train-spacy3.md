1) Annotate with doccano
2) Convert the JSONL- to biluo with convert.py
3) Run `python -m spacy convert annotation_iob.json --lang es .`
4) Start a new base model: `python -m spacy init fill-config base_config.cfg config.cfg`
5) Train with: `python -m spacy train config.cfg --paths.train annotation_iob.spacy --paths.dev dev.spacy`