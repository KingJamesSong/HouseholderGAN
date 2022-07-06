Codes for StyleGAN

##TO-DO LISTS

Reduce Number of Semantic Attributes (10 --> 5).
Adjust Traversal Range (e.g., -1010 or -55).
Try more and all layers.
Increase Training Steps (1k to 10k).
Weight Initialization at 0k steps (Nearest-Orthogonal Mapping).
Study Related Work (SeFA, GANSpace)

2020.7.1

1) ortho all layers + training partial parameters + with loading d

2) ortho all layers + training partial parameters + without loading d

3) #ortho all layers + training all parameters + with loading d (Preferred)#

2020.7.6:

Discussion:

1) Limited semantic discovery.

2) G(a + n * d1) = G(a + n * d2) or

  G(a + n * d1) = G(a - n * d2)
  
3) loading Discriminator is necessary.

4) Training all parameters will improve the quality of generated samples and preserve the identity, but discovers
the limited discovery.

Experiments: 

1) Increasing BatchSize from 2 to 48 (useful, significantly increase diversity and hierachy).

2) Loss to increase num of semantics.

2) FID and VP.

3) Rank of the matrix: 5, 10, 20.

4) ortho gradient.
