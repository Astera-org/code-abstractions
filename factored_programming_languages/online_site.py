"""Online residual-stream site for the TCI ladder runners (`--site online`; authors, 2026-09-25, the Mistral counter-example).

The suffix site writes a factor's K/V-row contrast into the recipient's prompt memory once. The online site is the
PER-TOKEN COUNTERFACTUAL of run_online_factors.py / run_online_steer.py: every donor context (the same LOO pairs the K/V
battery pools) is teacher-forced along the recipient's own output, and at each step the recipient's residual stream at the
token being written gets
    h_l[:, -1, :] += alpha * delta_l,   delta_l = sum over factor groups of mean over pairs (x, y) of h_l(x, y_<t) - h_l(y, y_<t)
over the model's certified band layers (models.band). A "delta" carried through the runners is therefore a recipe, not a
tensor: {"_groups": [[(plus_key, minus_key), ...], ...]} (one group per factor; compose sums the groups), optionally
{"_rand": seed} (norm-matched random direction per layer, fixed per cell, matched to ||delta_l|| at every step) and
{"_match": groups} (the random-labelling control rescaled per step and per layer to the norm of the reference contrast).
Severing of the recipient (attention mask on the specification rows) is unchanged; the contexts are never masked.
"""
import torch


class OnlineSite:
    def __init__(self, layers, band_rows, mode="live"):
        self.band = list(range(band_rows[0] - 1, band_rows[1]))       # certified rows (1-based, inclusive) -> layer indices
        self.mode, self.alpha, self.cap, self.delta, self.fired = "off", 0.0, {}, {}, 0
        self.vmode = mode                     # "live": per-token counterfactual; "vector": pooled generation vector, frozen
        self.programs, self.vectors = {}, {}   # vector mode caches: prompt key -> greedy program ids; recipe -> per-layer vector
        self.hooks = [layers[i].register_forward_hook(self._hook(i)) for i in self.band]
        self.printed = False

    def _hook(self, i):
        def fn(mod, inp, out):
            tup = isinstance(out, tuple)
            h = out[0] if tup else out
            if self.mode == "capture":
                self.cap[i] = h[:, -1, :].detach().float()
                return None
            if self.mode == "capture_all":
                self.cap[i] = h.detach().float()
                return None
            if self.mode == "steer" and i in self.delta:
                h = h.clone()
                h[:, -1, :] += self.alpha * self.delta[i].to(h.device, h.dtype)
                self.fired += 1
                return (h,) + tuple(out[1:]) if tup else h
            return None
        return fn

    def info(self):
        return {"mode": self.vmode, "band_layers": [self.band[0], self.band[-1]], "n_layers": len(self.band),
                "delta": "per token: sum over groups of mean over LOO pairs of h_l(x, y_<t) - h_l(y, y_<t), contexts teacher-forced",
                "apply": "h_l[:, -1, :] += alpha * delta_l at the launch step and every generated token",
                "controls": "rand = fixed random direction per layer, norm-matched to delta_l every step; match = rescaled per step to the reference groups' norm"}

    # ---- pooled generation vector (the literature's fixed direction, read at the continuation positions) ---------------
    @torch.inference_mode()
    def _greedy(self, model, tok, ids, stop, stop_strings, max_new, dev):
        """the model's own unsteered program for a prompt (token ids), cached per prompt key by the caller"""
        L = ids.shape[1]
        cache = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
        nxt_ids, out = ids[:, L - 1:], []
        for _ in range(max_new):
            step = model(input_ids=nxt_ids, past_key_values=cache, use_cache=True)
            cache = step.past_key_values
            nxt = int(step.logits[:, -1].argmax(-1))
            if nxt in stop:
                break
            out.append(nxt)
            if stop_strings and any(s_ in tok.decode(out[-24:], skip_special_tokens=True) for s_ in stop_strings):
                break
            nxt_ids = torch.tensor([[nxt]], device=dev)
        return out

    @torch.inference_mode()
    def vector_for(self, groups, model, tok, enc_of, padded, stop, stop_strings, max_new, dev):
        """sum over groups of mean over pairs (x, y) of mean over the program's positions of h_l(x, y+_<t) - h_l(y, y+_<t),
        y+ = x's own greedy program teacher-forced through BOTH contexts (paired: same tokens, only the prompt differs)"""
        key = tuple(tuple(tuple(map(str, pr)) for pr in grp) for grp in groups)
        if key in self.vectors:
            return self.vectors[key]
        total = {i: None for i in self.band}
        for grp in groups:
            acc = {i: None for i in self.band}
            for x, y in grp:
                if x not in self.programs:
                    self.programs[x] = self._greedy(model, tok, enc_of(padded[x]), stop, stop_strings, max_new, dev)
                prog = self.programs[x]
                if not prog:
                    continue
                px, py = enc_of(padded[x]), enc_of(padded[y])
                L = px.shape[1]
                pt = torch.tensor([prog], device=dev)
                bids = torch.cat([torch.cat([px, pt], 1), torch.cat([py, pt], 1)], 0)
                self.cap = {}; self.mode = "capture_all"
                model(input_ids=bids, use_cache=False)
                self.mode = "off"
                for i in self.band:                                   # positions that write the program: launch .. last-1
                    h = self.cap[i][:, L - 1:L - 1 + len(prog), :]
                    d = (h[0] - h[1]).mean(0)
                    acc[i] = d if acc[i] is None else acc[i] + d
            n = sum(1 for x, _ in grp if self.programs.get(x))
            for i in self.band:
                if acc[i] is not None:
                    g = acc[i] / max(n, 1)
                    total[i] = g if total[i] is None else total[i] + g
        dim = model.get_input_embeddings().weight.shape[1]
        vec = {i: (total[i] if total[i] is not None else torch.zeros(dim, device=dev)) for i in self.band}
        self.vectors[key] = vec
        return vec

    @torch.inference_mode()
    def generate_vector(self, model, tok, enc_of, padded, cache, ids, amask, delta, alpha, stop, stop_strings, max_new, dev):
        """vector mode: one frozen per-layer vector for the cell, added at the launch step and every generated token"""
        groups = delta["_groups"]; match = delta.get("_match"); rand_seed = delta.get("_rand")
        D = dict(self.vector_for(groups, model, tok, enc_of, padded, stop, stop_strings, max_new, dev))
        if match:
            E = self.vector_for(match, model, tok, enc_of, padded, stop, stop_strings, max_new, dev)
            D = {i: D[i] * (E[i].norm() / (D[i].norm() + 1e-9)) for i in self.band}
        if rand_seed is not None:
            g = torch.Generator().manual_seed(int(rand_seed))
            D = {i: (r := torch.randn(D[i].shape, generator=g).to(D[i].device)) * (D[i].norm() / (r.norm() + 1e-9)) for i in self.band}
        L = ids.shape[1]
        self.fired = 0
        nxt_ids = ids[:, L - 1:]
        out = []
        steps = 0
        for _ in range(max_new):
            self.delta, self.alpha, self.mode = {i: D[i].unsqueeze(0) for i in self.band}, alpha, "steer"
            steps += 1
            step = model(input_ids=nxt_ids, past_key_values=cache, use_cache=True, attention_mask=amask)
            self.mode = "off"
            cache = step.past_key_values
            nxt = int(step.logits[:, -1].argmax(-1))
            if nxt in stop:
                break
            out.append(nxt)
            if stop_strings and any(s_ in tok.decode(out[-24:], skip_special_tokens=True) for s_ in stop_strings):
                break
            if amask is not None:
                amask = torch.cat([amask, torch.ones(1, 1, dtype=amask.dtype, device=amask.device)], 1)
            nxt_ids = torch.tensor([[nxt]], device=dev)
        self.delta = {}
        if not self.printed:
            self.printed = True
            nv = float(sum(D[i].norm() for i in self.band))
            print(f"[online:vector] {len(self.programs)} programs cached; |v| summed over {len(self.band)} layers {nv:.1f}; fired {self.fired} adds over {steps} steps", flush=True)
            assert self.fired == steps * len(self.band), (self.fired, steps, len(self.band))
        return tok.decode(out, skip_special_tokens=True)

    @torch.inference_mode()
    def generate_online(self, model, tok, enc_of, padded, cache, ids, amask, delta, alpha, stop, stop_strings, max_new, dev):
        if self.vmode == "vector":
            return self.generate_vector(model, tok, enc_of, padded, cache, ids, amask, delta, alpha, stop, stop_strings, max_new, dev)
        groups = delta["_groups"]; match = delta.get("_match"); rand_seed = delta.get("_rand")
        keys = []
        for grp in list(groups) + (list(match) if match else []):
            for x, y in grp:
                for k in (x, y):
                    if k not in keys:
                        keys.append(k)
        idx = {k: j for j, k in enumerate(keys)}
        bids = torch.cat([enc_of(padded[k]) for k in keys], 0)          # padded prompts share one length
        assert bids.shape[1] == ids.shape[1], (bids.shape, ids.shape)
        L = ids.shape[1]
        # contexts: prefill up to the launch token and capture there; the recipient's launch step then reads this first delta
        self.cap = {}; self.mode = "capture"
        bcache = model(input_ids=bids, use_cache=True).past_key_values
        self.mode = "off"; H = {i: self.cap[i].clone() for i in self.band}
        gen = torch.Generator().manual_seed(int(rand_seed)) if rand_seed is not None else None
        rdir = {i: torch.randn(H[i].shape[1], generator=gen) for i in self.band} if gen is not None else None

        def contrast(grps):
            out = {}
            for i in self.band:
                acc = None
                for grp in grps:
                    d = torch.stack([H[i][idx[x]] - H[i][idx[y]] for x, y in grp]).mean(0)
                    acc = d if acc is None else acc + d
                out[i] = acc
            return out

        def delta_now():
            D = contrast(groups)
            if match:
                E = contrast(match)
                D = {i: D[i] * (E[i].norm() / (D[i].norm() + 1e-9)) for i in self.band}
            if rdir is not None:
                D = {i: rdir[i].to(D[i].device) * (D[i].norm() / (rdir[i].norm() + 1e-9)) for i in self.band}
            return D

        self.fired = 0
        nxt_ids = ids[:, L - 1:]
        out = []
        steps = 0
        for _ in range(max_new):
            self.delta, self.alpha, self.mode = delta_now(), alpha, "steer"
            steps += 1
            step = model(input_ids=nxt_ids, past_key_values=cache, use_cache=True, attention_mask=amask)
            self.mode = "off"
            cache = step.past_key_values
            nxt = int(step.logits[:, -1].argmax(-1))
            if nxt in stop:
                break
            out.append(nxt)
            if stop_strings and any(s_ in tok.decode(out[-24:], skip_special_tokens=True) for s_ in stop_strings):
                break
            if amask is not None:
                amask = torch.cat([amask, torch.ones(1, 1, dtype=amask.dtype, device=amask.device)], 1)
            nxt_ids = torch.tensor([[nxt]], device=dev)
            self.cap = {}; self.mode = "capture"                      # teacher-force the contexts with the recipient's token
            bcache = model(input_ids=nxt_ids.expand(bids.shape[0], -1), past_key_values=bcache, use_cache=True).past_key_values
            self.mode = "off"; H = {i: self.cap[i].clone() for i in self.band}
        self.delta = {}
        if not self.printed:
            self.printed = True
            print(f"[online] {len(keys)} contexts x {len(self.band)} layers; fired {self.fired} adds over {steps} steps", flush=True)
            assert self.fired == steps * len(self.band), (self.fired, steps, len(self.band))   # steps, not len(out)+1: a max-length run has no stop step
        return tok.decode(out, skip_special_tokens=True)
