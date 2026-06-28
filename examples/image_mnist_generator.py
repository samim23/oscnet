"""Run the MNIST oscillator generator experiment."""

from oscnet.experiments.mnist_generator import RECOMMENDED_GENERATOR_PRESET, main


if __name__ == "__main__":
    main(default_preset=RECOMMENDED_GENERATOR_PRESET)
