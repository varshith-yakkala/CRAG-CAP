from typing import List

def decompose_passage(psg: str, mode: str = "selection") -> List[str]:
    """
    Decomposes retrieved passage into sub-strips based on selected mode:
    - 'fixed_num': Segments passage into fixed word windows (e.g. 50 words)
    - 'excerption': Segments passage based on sentence boundaries (period / question mark)
    - 'selection': Retains passage intact (no splitting)
    """
    if mode == "selection":
        return [psg]

    elif mode == "fixed_num":
        window_length = 50
        words = psg.split(" ")
        final_strips = []
        buf = []
        for w in words:
            buf.append(w)
            if len(buf) == window_length:
                final_strips.append(" ".join(buf))
                buf = []
        if buf:
            if final_strips and len(buf) < 10:
                final_strips[-1] += (" " + " ".join(buf))
            else:
                final_strips.append(" ".join(buf))
        return final_strips

    elif mode == "excerption":
        num_concatenate_strips = 3
        question_strips = psg.split("?")
        origin_strips = []
        for qs in question_strips:
            origin_strips += qs.split(". ")
        strips = []
        for s in origin_strips:
            s_clean = s.strip()
            if not s_clean:
                continue
            if s_clean in strips:
                continue
            if not strips:
                strips.append(s_clean)
            else:
                if len(s_clean.split()) > 5:
                    strips.append(s_clean)
                else:
                    strips[-1] += (" " + s_clean)
        final_strips = []
        buf = []
        for strip in strips:
            buf.append(strip)
            if len(buf) == num_concatenate_strips:
                final_strips.append(" ".join(buf))
                buf = []
        if buf:
            final_strips.append(" ".join(buf))
        return final_strips

    return [psg]
