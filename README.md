# CSE 848 Project

Experiments using autoencoders to learn evolvable encodings for the *n*-legged table problem.

## *Learning an Evolvable Genotype-Phenotype Map*

Experiments reported in this paper employed    [v2.0.2](https://github.com/mmore500/cse-848-project/tree/v2.0.2) of this software.

data, tutorials, and writeup @ [https://osf.io/n92c7/](https://osf.io/n92c7/)

Accepted to GECCO 2018.

> We present AutoMap, a pair of methods for automatic generation of evolvable genotype-phenotype mappings.
Both use an artificial neural network autoencoder trained on phenotypes harvested from fitness peaks as the basis for a genotype-phenotype mapping.
In the first, the decoder segment of a bottlenecked autoencoder serves as the genotype-phenotype mapping.
In the second, a denoising autoencoder serves as the genotype-phenotype mapping.
Automatic generation of evolvable genotype-phenotype mappings are demonstrated on the $n$-legged table problem, a toy problem that defines a simple rugged fitness landscape, and the Scrabble string problem, a more complicated problem that serves as a rough model for linear genetic programming.
For both problems, the automatically generated genotype-phenotype mappings are found to enhance evolvability.

## Software Authorship

Matthew Andres Moreno

`mmore500@msu.edu`

## Credits

This implementation draws on several open-source packages, most notably Distributed Evolutionary Algorithms for Python (DEAP).
Both the package and (adapted) example usage from the package's documentation were employed in this implementation.
