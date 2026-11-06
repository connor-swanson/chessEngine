"""Shared code for the training notebooks (encoding, model, data, search, evaluation).

Notebooks keep the *algorithm* they teach (the loss, the training loop) in the
notebook itself and import only the plumbing from here, so every notebook uses
the exact same board encoding, network and evaluation.
"""
from .encoding import (ACTION_SIZE, NUM_PLANES, encode_board, planes_to_float, move_to_index,
                       index_to_move, legal_moves_and_indices, legal_mask)
from .model import (ChessNet, wdl_to_scalar, masked_log_softmax, count_params, default_device,
                    save_checkpoint, load_checkpoint)
