import numpy as np
from collections import Counter
from .base_scan import BaseScan, ScanFinding

class BackdoorScan(BaseScan):
    """
    Detect potential backdoor triggers using two signals:

    1. **Rare-token frequency analysis** — backdoor triggers are often
       uncommon tokens/phrases that appear disproportionately with one label.

    2. **Loss-trajectory outliers** — samples with a trigger converge
       unusually fast because the model learns the shortcut quickly.
    """
    name = "backdoor"

    def run(self, model, tokenizer, dataset, train_losses, cfg) -> list[ScanFinding]:
        findings = []
        texts  = dataset[cfg.text_column]
        labels = dataset[cfg.label_column] if cfg.label_column else [None]*len(texts)

        # --- Signal 1: rare token–label correlation ---
        token_label: dict[str, list] = {}
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            for tok in set(tokens):                       # unique per sample
                token_label.setdefault(tok, []).append(labels[i])

        suspicious_tokens = []
        for tok, lbls in token_label.items():
            if len(lbls) < 3:
                continue
            most_common_label, count = Counter(lbls).most_common(1)[0]
            ratio = count / len(lbls)
            if ratio > 0.95 and len(lbls) > 5:
                suspicious_tokens.append((tok, most_common_label, ratio, len(lbls)))

        if suspicious_tokens:
            # Find which training rows contain these tokens
            sus_token_set = {t[0] for t in suspicious_tokens}
            affected = [i for i, t in enumerate(texts)
                        if sus_token_set & set(t.lower().split())]
            findings.append(ScanFinding(
                scan_name=self.name,
                severity="high",
                description=(
                    f"Found {len(suspicious_tokens)} token(s) with >95% "
                    f"label correlation — possible backdoor triggers: "
                    f"{[t[0] for t in suspicious_tokens[:10]]}"
                ),
                affected_indices=affected,
                confidence=0.75,
                metadata={"suspicious_tokens": suspicious_tokens[:20]},
            ))

        # --- Signal 2: loss-trajectory outliers ---
        if train_losses:
            # Average loss per sample across batches (rough approximation).
            # Training batches can be uneven (last batch smaller), so we can't
            # directly convert to a rectangular numpy array.
            max_batch_len = max(len(batch) for batch in train_losses)
            per_position_losses = []
            for pos in range(max_batch_len):
                # Collect the loss for this position in each batch (if present).
                vals = [batch[pos] for batch in train_losses if pos < len(batch)]
                if vals:
                    per_position_losses.append(np.mean(vals))

            if per_position_losses:
                mean_loss = np.array(per_position_losses)
                # Samples with anomalously LOW loss learned a "shortcut"
                threshold = np.percentile(mean_loss, 5)
                fast_learners = np.where(mean_loss < threshold)[0].tolist()

                if fast_learners:
                    findings.append(ScanFinding(
                        scan_name=self.name,
                        severity="medium",
                        description=(
                            f"{len(fast_learners)} samples converged unusually "
                            f"fast — potential shortcut/trigger learning."
                        ),
                        affected_indices=fast_learners,
                        confidence=0.60,
                    ))

        return findings