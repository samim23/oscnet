#!/usr/bin/env python3
"""Spoken-digit RFB→HORN classification (local entrypoint).

Examples:

    python examples/audio_digit_classification.py --frontend resonator
    python examples/audio_digit_classification.py --sweep --epochs 20
"""

from oscnet.experiments.audio_digit.cli import main

if __name__ == "__main__":
    main()
