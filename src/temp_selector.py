import json
import math
import os
import re
from collections import Counter


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _parse_temperature(text):
    tagged = re.search(r"<temp>\s*([0-9]*\.?[0-9]+)\s*</temp>", text)
    if tagged:
        return float(tagged.group(1))

    fallback = re.search(r"([0-9]*\.?[0-9]+)", text)
    if fallback:
        return float(fallback.group(1))
    return None


class StaticTemperatureSelector:
    """
    Legacy static selector that preserves the original target_round/target_agent behavior.
    """

    def __init__(self, args, num_agents):
        self.args = args
        self.num_agents = num_agents

    def start_sample(self, sample_idx):
        return

    def get_temperatures(self, round_idx):
        temperatures = [1.0] * self.num_agents
        if round_idx == self.args.target_round and 0 <= self.args.target_agent_idx < self.num_agents:
            temperatures[self.args.target_agent_idx] = self.args.target_temp
        return temperatures

    def observe_round(self, round_idx, signals):
        return

    def metadata(self):
        return {
            "mode": "off",
            "selector": "static",
            "target_round": self.args.target_round,
            "target_agent_idx": self.args.target_agent_idx,
            "target_temp": self.args.target_temp,
        }

    def sample_state(self):
        return {}

    def save_state(self):
        return


class AdaptiveTemperatureSelector:
    """
    Adaptive selector that can optionally use TextGrad for policy updates.

    Signals are self-supervised only: response disagreement/entropy, empty-answer rate,
    and response length stats. No ground-truth labels are used for temperature updates.
    """

    def __init__(self, args, num_agents):
        self.args = args
        self.num_agents = num_agents
        self.min_temp = args.temp_selector_min_temp
        self.max_temp = args.temp_selector_max_temp
        self.step = args.temp_selector_step
        self.apply_to = args.temp_selector_apply_to
        self.initial_temp = _clamp(float(args.target_temp), self.min_temp, self.max_temp)
        self.current_temp = self.initial_temp
        self.current_direction = 1.0
        self.previous_reward = None
        self.round_history = []
        self.state_path = args.temp_selector_state_path.strip()
        self.sample_idx = None

        self._tg = None
        self._tg_enabled = False
        self._tg_error = None
        self._tg_policy = None
        self._tg_optimizer = None

        self._load_state()
        if args.temp_selector == "textgrad":
            self._init_textgrad()

    def _init_textgrad(self):
        try:
            import textgrad as tg

            self._tg = tg
            if self.args.textgrad_backward_engine:
                try:
                    tg.set_backward_engine(self.args.textgrad_backward_engine, override=True)
                except TypeError:
                    tg.set_backward_engine(self.args.textgrad_backward_engine)

            self._tg_policy = tg.Variable(
                f"Tune MAD temperature policy. Output a single value in the format <temp>{self.current_temp:.2f}</temp>.",
                requires_grad=True,
                role_description="temperature policy for multi-agent debate",
            )
            self._tg_optimizer = tg.TGD(parameters=[self._tg_policy])
            self._tg_enabled = True
        except Exception as exc:
            self._tg_enabled = False
            self._tg_error = str(exc)

    def _load_state(self):
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r") as f:
                payload = json.load(f)
            saved_temp = payload.get("current_temp", self.current_temp)
            self.current_temp = _clamp(float(saved_temp), self.min_temp, self.max_temp)
            saved_direction = payload.get("current_direction", self.current_direction)
            self.current_direction = float(saved_direction)
        except Exception:
            return

    def start_sample(self, sample_idx):
        self.sample_idx = sample_idx
        self.current_temp = self.initial_temp
        self.current_direction = 1.0
        self.previous_reward = None
        self.round_history = []

    def get_temperatures(self, round_idx):
        if self.apply_to == "target_agent":
            temperatures = [1.0] * self.num_agents
            if 0 <= self.args.target_agent_idx < self.num_agents:
                temperatures[self.args.target_agent_idx] = self.current_temp
            else:
                temperatures = [self.current_temp] * self.num_agents
        else:
            temperatures = [self.current_temp] * self.num_agents
        return temperatures

    def _compute_signals(self, signals):
        responses = list(signals.get("responses", {}).values())
        final_answers = signals.get("final_answers", [])
        usable_answers = [str(x).strip() for x in final_answers if str(x).strip() != ""]

        if not usable_answers:
            agreement = 0.0
            entropy = 1.0
            empty_ratio = 1.0
        else:
            counts = Counter(usable_answers)
            total = sum(counts.values())
            probs = [c / total for c in counts.values()]
            agreement = max(probs)

            if len(probs) <= 1:
                entropy = 0.0
            else:
                raw_entropy = -sum(p * math.log(p + 1e-12) for p in probs)
                entropy = raw_entropy / math.log(len(probs))

            empty_ratio = 1.0 - (len(usable_answers) / max(1, len(final_answers)))

        lengths = [len(str(r).split()) for r in responses]
        avg_len = float(sum(lengths) / max(1, len(lengths)))

        reward = agreement - 0.35 * entropy - 0.15 * empty_ratio
        return {
            "agreement": float(agreement),
            "vote_entropy": float(entropy),
            "empty_ratio": float(empty_ratio),
            "avg_response_len": avg_len,
            "proxy_reward": float(reward),
        }

    def _heuristic_next_temp(self, metrics):
        reward = metrics["proxy_reward"]

        if self.previous_reward is None:
            direction = 1.0 if metrics["vote_entropy"] > 0.4 else -1.0
        else:
            direction = self.current_direction if reward >= self.previous_reward else -self.current_direction

        if metrics["empty_ratio"] > 0.5:
            direction = -1.0

        candidate = self.current_temp + (direction * self.step)
        candidate = _clamp(candidate, self.min_temp, self.max_temp)

        self.current_direction = direction
        self.previous_reward = reward
        return candidate

    def _textgrad_next_temp(self, metrics):
        if not self._tg_enabled:
            return None

        try:
            instruction = (
                "You are optimizing a temperature policy for multi-agent debate. "
                "Use only self-supervised signals from the previous round. "
                f"Current temp={self.current_temp:.3f}, "
                f"agreement={metrics['agreement']:.3f}, "
                f"entropy={metrics['vote_entropy']:.3f}, "
                f"empty_ratio={metrics['empty_ratio']:.3f}, "
                f"avg_len={metrics['avg_response_len']:.1f}. "
                f"Output one temperature in [{self.min_temp:.2f}, {self.max_temp:.2f}] "
                "as exactly one tag: <temp>X.YY</temp>."
            )

            loss_fn = self._tg.TextLoss(instruction)
            if hasattr(self._tg_optimizer, "zero_grad"):
                self._tg_optimizer.zero_grad()
            loss = loss_fn(self._tg_policy)
            loss.backward()
            self._tg_optimizer.step()

            parsed = _parse_temperature(self._tg_policy.value)
            if parsed is None:
                return None
            return _clamp(float(parsed), self.min_temp, self.max_temp)
        except Exception as exc:
            self._tg_enabled = False
            self._tg_error = str(exc)
            return None

    def observe_round(self, round_idx, signals):
        metrics = self._compute_signals(signals)
        next_temp = self._textgrad_next_temp(metrics)
        if next_temp is None:
            next_temp = self._heuristic_next_temp(metrics)

        self.round_history.append(
            {
                "round": int(round_idx),
                "selected_temp": float(self.current_temp),
                "metrics": metrics,
                "next_temp": float(next_temp),
                "textgrad_used": bool(self._tg_enabled),
            }
        )
        self.current_temp = float(next_temp)

    def metadata(self):
        return {
            "mode": "adaptive",
            "selector": self.args.temp_selector,
            "apply_to": self.apply_to,
            "min_temp": self.min_temp,
            "max_temp": self.max_temp,
            "step": self.step,
            "initial_temp": self.initial_temp,
            "textgrad_enabled": self._tg_enabled,
            "textgrad_backward_engine": self.args.textgrad_backward_engine,
            "textgrad_error": self._tg_error,
        }

    def sample_state(self):
        return {
            "sample_idx": self.sample_idx,
            "round_history": self.round_history,
            "final_temp": self.current_temp,
        }

    def save_state(self):
        if not self.state_path:
            return

        payload = {
            "version": 1,
            "current_temp": self.current_temp,
            "current_direction": self.current_direction,
            "metadata": self.metadata(),
        }

        state_dir = os.path.dirname(self.state_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)

        with open(self.state_path, "w") as f:
            json.dump(payload, f, indent=2)


def build_temperature_selector(args, num_agents):
    if args.temp_selector_mode == "off":
        return StaticTemperatureSelector(args, num_agents)
    if args.temp_selector_mode == "adaptive":
        return AdaptiveTemperatureSelector(args, num_agents)
    raise ValueError(f"Unknown temp selector mode: {args.temp_selector_mode}")