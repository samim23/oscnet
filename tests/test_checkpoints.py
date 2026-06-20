import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from oscnet.models import OscillatoryAutoencoder
from oscnet.utils import load_equinox_checkpoint, save_equinox_checkpoint


def _make_model(**hyperparams):
    return OscillatoryAutoencoder(key=jax.random.PRNGKey(0), **hyperparams)


def _make_optimizer():
    return optax.adam(1e-3)


def test_equinox_checkpoint_round_trip(tmp_path):
    hyperparams = {
        "input_dim": 4,
        "hidden_dim": 6,
        "latent_dim": 3,
        "sequence_length": 2,
        "decoder_mode": "repeat",
    }
    model = _make_model(**hyperparams)
    optimizer = _make_optimizer()
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    inputs = jnp.ones((2, 1, 4))
    expected = model(inputs)

    checkpoint_path = save_equinox_checkpoint(
        model=model,
        opt_state=opt_state,
        epoch=1,
        metrics={"loss": 0.5},
        output_dir=tmp_path,
        hyperparams=hyperparams,
    )

    loaded_model, loaded_opt_state, metadata, loaded_hyperparams = load_equinox_checkpoint(
        checkpoint_path,
        make_model_fn=_make_model,
        make_optimizer_fn=_make_optimizer,
    )

    assert loaded_hyperparams == hyperparams
    assert metadata["epoch"] == 1
    assert metadata["metrics"]["loss"] == 0.5
    assert loaded_opt_state is not None
    assert jnp.allclose(loaded_model(inputs), expected)
