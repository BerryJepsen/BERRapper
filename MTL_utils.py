from transformers import AutoTokenizer, BertConfig
from MTL_hparam import model_name, label_all_tokens
import transformers


def tokenize_and_align_data(examples):
    # tokenizing
    #tokenizer = AutoTokenizer.from_pretrained('./pretrained_models/')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding='max_length', max_length=128)

    durations_tmp = []
    for i, dur in enumerate(examples["durations"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        durations_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                durations_ids.append(0)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                durations_ids.append(dur[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                durations_ids.append(dur[word_idx] if label_all_tokens else 0)
            previous_word_idx = word_idx

        durations_tmp.append(durations_ids)
    tokenized_inputs["durations"] = durations_tmp

    labels_tmp = []
    for i, label in enumerate(examples["beats"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels_tmp.append(label_ids)

    tokenized_inputs["labels"] = labels_tmp

    pitches_tmp = []
    for i, label in enumerate(examples["pitches"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        pitch_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                pitch_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                pitch_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                pitch_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        pitches_tmp.append(pitch_ids)

    tokenized_inputs["pitches"] = pitches_tmp
    return tokenized_inputs


