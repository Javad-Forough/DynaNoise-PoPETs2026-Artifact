import torch
import torch.nn.functional as F
import math

class DynaNoise:
    def __init__(self, base_variance=0.1, lambda_scale=1.0, temperature=1.0, ensemble_size=1):
        """
        :param base_variance: Base noise variance (sigma^2_0).
        :param lambda_scale:  Lambda (λ) scales noise with sensitivity.
        :param temperature:   Softmax temperature smoothing.
        :param ensemble_size: M for probabilistic smoothing (averaging M samples).
        """
        self.base_variance = float(base_variance)
        self.lambda_scale = float(lambda_scale)
        self.temperature = float(temperature)
        self.ensemble_size = int(ensemble_size)

    def sensitivity_score(self, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            eps = 1e-9
            entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)  # (batch,)
            k = logits.shape[1]
            max_entropy = math.log(k + eps)
            R_q = 1.0 - (entropy / max_entropy)
        return R_q

    def inject_noise(self, logits: torch.Tensor) -> torch.Tensor:
        R_q = self.sensitivity_score(logits)  # (batch,)
        sigma_sq = self.base_variance * (1.0 + self.lambda_scale * R_q)  # (batch,)

        std = torch.sqrt(sigma_sq).unsqueeze(1)  # (batch,1)
        noise = torch.randn_like(logits) * std
        return logits + noise

    def smooth_output(self, noisy_logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(noisy_logits / self.temperature, dim=1)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.ensemble_size <= 1:
            noisy = self.inject_noise(logits)
            return self.smooth_output(noisy)

        probs_accum = 0.0
        for _ in range(self.ensemble_size):
            noisy = self.inject_noise(logits)
            probs_accum = probs_accum + self.smooth_output(noisy)
        return probs_accum / float(self.ensemble_size)
