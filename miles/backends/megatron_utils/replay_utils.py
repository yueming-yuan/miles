from miles.utils.replay_base import BaseReplayManager, IndexerReplayManager, RoutingReplayManager


def _register_replay_list_moe(replay_list, replay_data, models):
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    layer_indices = []
    replay_idx = 0
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if isinstance(config.moe_layer_freq, int):
                if layer_id % config.moe_layer_freq != 0:
                    continue
            elif isinstance(config.moe_layer_freq, list):
                assert len(config.moe_layer_freq) == config.num_layers
                if config.moe_layer_freq[layer_id] == 0:
                    continue
            layer_indices.append(layer_id)

    for replay_idx, layer_idx in enumerate(layer_indices):
        layer_data = replay_data[:, layer_idx]
        replay_list[replay_idx].record(layer_data)


def _register_replay_list_sequential(replay_list, replay_data, _models):
    if replay_data.shape[1] != len(replay_list):
        raise AssertionError(
            f"replay data has {replay_data.shape[1]} streams, but {len(replay_list)} modules registered replay"
        )

    for replay_idx, replay in enumerate(replay_list):
        replay.record(replay_data[:, replay_idx])


def get_register_replay_list_func(manager: BaseReplayManager):
    if isinstance(manager, RoutingReplayManager):
        return _register_replay_list_moe
    elif isinstance(manager, IndexerReplayManager):
        return _register_replay_list_sequential
    else:
        raise ValueError(f"Unsupported manager type: {type(manager)}")
