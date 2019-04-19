
#!/bin/sh

echo "Downloading the word2vec"
wget http://bioasq.lip6.fr/tools/BioASQword2vec/
tar -zxvf index.html
mv word2vecTools/* .
rm index.html
rm -rf word2vecTools
