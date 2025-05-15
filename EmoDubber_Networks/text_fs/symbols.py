from text_fs import cmudict

_silences = ["@sp", "@spn", "@sil"]
_arpabet = ["@" + s for s in cmudict.valid_symbols]
symbols = (
    _arpabet
    + _silences
)

