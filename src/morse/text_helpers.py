import torch
import torch.nn.functional as F
import MorseCodePy as mcp

class Vectorizer:
    def __init__(self, token_to_index, index_to_token):
        self.text_to_nums = token_to_index
        self.nums_to_text = index_to_token
        pass

    def to_tensor(self, text: str):
        return torch.tensor([self.text_to_nums[ch] for ch in text])
    
    def from_tensor(self, tensor: torch.Tensor):
        assert tensor.ndim == 1
        return ''.join([self.nums_to_text[num.item()] for num in tensor])
    
    def text_transform(self, text: str):
        '''override if needed'''
        return self.to_tensor(text)
    
    def batch_text_transform(self, texts: list[str], pad_value):
        vecs = []
        for t in texts:
            vecs.append(self.text_transform(t))
        lengths = torch.tensor([len(v) for v in vecs])
        desired_length = torch.max(lengths)
        vecs = [F.pad(v, (0, desired_length - len(v)), value=pad_value) for v in vecs]
        batch = torch.stack(vecs, dim=0)
        return batch, lengths

def encode_to_morse(text: str, remove_separator_pad=False):
    encoded = mcp.encode(text, language='russian')
    sharp_code = '--.--'    # #
    hard_code = '.--.-.'
    common_hard_code = '--.--'  # Ъ
    encoded = encoded.replace(common_hard_code, hard_code)
    encoded = encoded.replace('*', sharp_code)

    if remove_separator_pad:
        encoded = encoded.replace('/ ', '/')
        encoded = encoded.replace(' /', '/')
    return encoded

def decode_from_morse(morse: str, separator_pads_removed=False):
    sharp_code = '--.--'    # #
    hard_code = '.--.-.'
    common_hard_code = '--.--'  # Ъ
    # # -> ? -> #
    question_mark_code = '..--..'   # ?
    code = morse.replace(sharp_code, question_mark_code)
    code = code.replace(hard_code, common_hard_code)
    if separator_pads_removed:
        code = code.replace('/', ' / ').strip(' ')
    text = mcp.decode(code, language='russian')
    text = text.replace('?', '#').upper()
    return text
