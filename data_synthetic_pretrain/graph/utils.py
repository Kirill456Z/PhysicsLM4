from data_synthetic_pretrain.graph.models import NodeWord, SpecialToken


def break_up_sequence_into_words(tokens: list[int], base_vocab_size: int) -> list[NodeWord | SpecialToken]:
    result = []
    cur_word = []
    for token in tokens:
        if token > 2 * base_vocab_size:
            result.append(SpecialToken(token=token))
            cur_word = []
        else:
            cur_word.append(token)
            if token > base_vocab_size:
                result.append(NodeWord(tokens=tuple(cur_word)))
                cur_word = []
    return result, cur_word