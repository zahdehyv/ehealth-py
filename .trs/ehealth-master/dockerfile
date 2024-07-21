FROM python:3.8

RUN pip install torch torchvision

RUN pip install streamlit

RUN pip install spacy
RUN pip install -U spacy-lookups-data
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download es_core_news_md

RUN pip install transformers
RUN wget https://users.dcc.uchile.cl/~jperez/beto/cased_2M/pytorch_weights.tar.gz 
RUN wget https://users.dcc.uchile.cl/~jperez/beto/cased_2M/vocab.txt 
RUN wget https://users.dcc.uchile.cl/~jperez/beto/cased_2M/config.json 
RUN tar -xzvf pytorch_weights.tar.gz
RUN mv pytorch/ configs/beto.
RUN mv config.json configs/beto.
RUN mv vocab.txt configs/beto.