# The Option-Critic Architecture

#### (Bacon PL, Harb J, Precup D)

This is a TensorFlow implementation of the option critic architecture](https://arxiv.org/pdf/1609.05140.pdf) by Bacon PL, Harb J and Precup D. 
The release of the code was influenced by the recent Baseline initiative from [OpenAI](https://github.com/openai/baselines) and is aimed to serve as a starting point in Hierarchical RL, more specifically in options learning.

Providing a unified common ground through the adoption of a single framework will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of as as recently pointed out by J. Schulman. This implementation of the option-critic architecture has been influenced by the original Theano implemntation by [J. Haarb](https://github.com/jeanharb/option_critic) as well as the extremely helpful repository of [D. Britz](https://github.com/dennybritz/reinforcement-learning) and the very engaging posts by [A. Juliani](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0).

We hope to be providing the community with a useful tensorflow implementation of a popular options learning architecture. This, however, is a research-oriented implementation thus it might not comply with the most efficient code practices or be fully functional across platforms.

You can download it by typing:

```bash
git clone git@github.com:yadrimz/option-critic.git
```

## Requirements:
- tensorflow-1.1.0
- python 2.7
- gym[atari]
