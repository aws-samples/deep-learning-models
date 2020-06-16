from colorama import Fore, Style


def colorize(token: str, color: str) -> str:
    return f"{color}{token}{Style.RESET_ALL}"


def colorize_gen(tokenizer, true_ids, gen_ids, mask):
    gen_ids = gen_ids.numpy().flatten()
    true_ids = true_ids.numpy().flatten()
    mask = mask.numpy().flatten()
    tokens = tokenizer.convert_ids_to_tokens(gen_ids)
    styled_tokens = tokens.copy()
    for i in range(len(tokens)):
        if mask[i]:
            styled_tokens[i] = colorize(
                tokens[i], color=Fore.GREEN if (true_ids[i] == gen_ids[i]) else Fore.RED
            )
        else:
            styled_tokens[i] = tokens[i]
    return " ".join(styled_tokens)


def colorize_dis(tokenizer, gen_ids, dis_preds):
    gen_ids = gen_ids.numpy().flatten()
    dis_preds = dis_preds.numpy().flatten()
    tokens = tokenizer.convert_ids_to_tokens(gen_ids)
    styled_tokens = tokens.copy()
    for i in range(len(tokens)):
        if dis_preds[i]:
            styled_tokens[i] = colorize(tokens[i], color=Fore.YELLOW)
        else:
            styled_tokens[i] = tokens[i]
    return " ".join(styled_tokens)
