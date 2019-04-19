#!/bin/sh

echo "Downloading the Dataset"
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip
mv cornell\ movie-dialogs\ corpus/* .
rm cornell_movie_dialogs_corpus.zip
rm -rf cornell\ movie-dialogs\ corpus/
